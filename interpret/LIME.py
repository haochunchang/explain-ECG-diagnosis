# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#
# LIME: Local Interpretable Model-agnostic Explanations
#
# Ref:
# 1. https://arxiv.org/abs/1602.04938
# 2. https://github.com/marcotcr/lime

from functools import partial

import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from torch import Tensor

import tqdm


class Lime:
    """
    Explains predictions on ECG signal data.

    (e.g. Data with 1-D time series with multiple channels)
    """

    def __init__(
        self,
        model,
        kernel_width=.25,
        kernel=None,
        verbose=False,
        feature_selection="auto",
        random_state=None
    ):
        """
        Arguments
        ---------
        model: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.
        Other parameters see LimeBase for detail. 
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.model = model

    def __call__(
        self,
        instance,
        labels=(1,),
        max_features=100000,
        num_samples=100,
        batch_size=10,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
        progress_bar=False
    ):
        """
        Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance.
        We then learn locally weighted linear models on this neighborhood data
        to explain each of the classes.

        Arguments
        ---------
            instance: numpy.ndarray of shape (1, num_channels, time_duration).
            labels: iterable with labels to be explained.
            max_features: maximum number of features present in explanation.
            num_samples: number of neighborhood data for learning the linear model.
            batch_size: number of neighborhood data to feed in ```self.model```
                at each label-generation step (in ```generate_data_labels```).
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression in LimeBase.
                Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns
        -------
            An ECG signal Explanation object with the corresponding explanations.
        """
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        segments = self.segment_signals(instance, num_segments=10)

        data, labels = self.generate_data_labels(
            instance, segments, num_samples,
            batch_size=batch_size,
            progress_bar=progress_bar
        )

        distances = pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        res = SignalExplanation(instance, segments)
        for label in range(labels.shape[1]):
            (
                res.intercept[label],
                res.local_exp[label],
                res.score,
                res.local_pred
            ) = self.base.explain_instance_with_data(
                data, labels, distances, label, max_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection
            )
        return res

    def segment_signals(self, instance, num_segments):
        """
        Segment signal instance into fixed time windows.

        Returns
        -------
        segments: np.ndarray, integer mask with same shape as ```instance```.
        """
        segments = np.zeros_like(instance)
        segment_size = instance.shape[-1] // num_segments
        for i in range(1, num_segments + 1):
            segments[0, :, i * segment_size:(i + 1) * segment_size] = i

        last_segment = segments[0, :, num_segments * segment_size:]
        if last_segment.shape[0] > 0:
            last_segment = num_segments + 1

        return segments

    def generate_data_labels(
        self,
        instance,
        segments,
        num_samples,
        batch_size=10,
        progress_bar=True
    ):
        """
        Generates signals and predictions in the neighborhood of this instance.
        By masking some parts of the instance, specified by segments.

        Returns
        -------
            A tuple (data, labels), where:
                data: integer array indicating which parts of the instance is masked with
                    shape (num_samples, num_segments).
                labels: prediction probabilities matrix with shape (num_samples, num_classes).
        """
        num_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, size=(num_samples, num_features))

        # First sample is the original data point, so disable all masks.
        data[0, :] = 1

        labels = []
        signals = []
        samples = tqdm(data) if progress_bar else data
        for sample in samples:
            temp = np.copy(instance)
            zeros = np.where(sample == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)

            for z in zeros:
                mask[segments == z] = True

            temp[mask] = 0
            signals.append(temp)

            if len(signals) == batch_size:
                try:
                    preds = self.model(np.array(signals))
                except TypeError:
                    preds = self.model(Tensor(signals))
                    preds = preds.detach().numpy()

                labels.extend(preds)
                signals = []

        labels = np.array(labels)
        return data, labels


class LimeBase:
    """
    Base class for learning a locally linear sparse model from perturbed data
    """
    def __init__(self, kernel_fn, verbose=False, random_state=None):
        """
        Arguments
        ---------
            kernel_fn: function that transforms an array of distances into an
                       array of proximity values (floats). (For sampling neighborhood data)
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection="auto",
        model_regressor=None
    ):
        """
        Takes perturbed data, labels and distances, returns explanation.

        Arguments
        ---------
        neighborhood_data: perturbed data, 2d array.
                        first element is assumed to be the original data point.
        neighborhood_labels: corresponding perturbed labels. should have as
                            many columns as the number of possible labels.
        distances: distances to original data point.
        label: int, label for which we want an explanation.
        num_features: int, maximum number of features in explanation.
        feature_selection: str, how to select ```num_features```.
            Options are:
            "forward_selection": iteratively add features to the model.
                This is costly when ```num_features``` is high.
            "highest_weights": selects the features that have the highest
                product of absolute weight * original data point when
                learning with all the features.
            "none": uses all features, ignores ```num_features```.
            "auto": uses forward_selection if ```num_features``` <= 6, and
                "highest_weights" otherwise. (Default)
        model_regressor: sklearn regressor to use in explanation.
            Defaults to Ridge regression if None. Must have
            model_regressor.coef_ and 'sample_weight' as a parameter
            to model_regressor.fit()

        Returns
        -------
        (intercept, exp, score, local_pred):
            intercept: float
            exp: a sorted list of tuples, where each tuple (x,y) corresponds
                to the feature id (x) and the local weight (y).
                The list is sorted by decreasing absolute value of y.
            score: float, the R^2 value of the returned explanation.
            local_pred: the prediction of the explanation model on the original instance.
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection
        )
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)

        easy_model = model_regressor
        easy_model.fit(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )
        original_sample = neighborhood_data[0, used_features].reshape(1, -1)
        local_pred = easy_model.predict(original_sample)

        if self.verbose:
            print("Intercept: ", easy_model.intercept_)
            print("Local prediction: ", local_pred)
            print("True label: ", neighborhood_labels[0, label])
        return (
            easy_model.intercept_,
            sorted(
                zip(used_features, easy_model.coef_),
                key=lambda x: np.abs(x[1]),
                reverse=True
            ),
            prediction_score,
            local_pred
        )

    def feature_selection(self, data, labels, weights, num_features, method):
        """
        Selects features for the model. see explain_instance_with_data to
           understand the parameters.
        """
        if method == "none":
            return np.array(range(data.shape[1]))
        elif method == "forward_selection":
            return self.forward_selection(data, labels, weights, num_features)
        elif method == "highest_weights":
            return self.highest_weights(data, labels, weights, num_features)
        elif method == "auto":
            if num_features <= 6:
                n_method = "forward_selection"
            else:
                n_method = "highest_weights"
            return self.feature_selection(
                data, labels, weights,
                num_features, n_method
            )
        else:
            raise ValueError("Unrecognized feature selection method: {}".format(method))

    def forward_selection(self, data, labels, weights, num_features):
        """
        Adds one feature (with the highest improvement) at a time.

        Returns an array of selected feature indexes.
        """
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(
                    data[:, used_features + [feature]],
                    labels,
                    sample_weight=weights
                )
                score = clf.score(
                    data[:, used_features + [feature]],
                    labels,
                    sample_weight=weights
                )
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def highest_weights(self, data, labels, weights, num_features):
        """
        Selects the features that have the highest
            ```absolute weight * original data point``` when learning with all the features.

        Returns
        -------
        An array of selected feature indexes.
        """
        clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
        clf.fit(data, labels, sample_weight=weights)

        coef = clf.coef_
        weighted_data = coef * data[0]
        feature_weights = sorted(
            zip(range(data.shape[1]), weighted_data),
            key=lambda x: np.abs(x[1]),
            reverse=True
        )
        return np.array([x[0] for x in feature_weights[:num_features]])


class SignalExplanation:
    """
    Class for holding ECG signal explanation.
    """
    def __init__(self, instance, segments):
        self.instance = instance
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.score = 0
        self.local_pred = None

    def get_instance_and_mask(
        self,
        label,
        positive_only=True,
        negative_only=False,
        num_features=5,
        min_weight=0.
    ):
        """
        Process and output time masks to explain the important parts of the signal.

        Arguments
        ---------
            label: int, label to explain.
            positive_only: if True, only take segments that positively contribute to
                the prediction of the label.
            negative_only: if True, only take segments that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time.
            num_features: number of segments to include in explanation.
            min_weight: minimum weight of the segments to include in explanation.

        Returns
        -------
            (instance, mask), where
                instance and mask both are 2d numpy array with shape (num_channel, duration).
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only and negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")

        segments = self.segments
        instance = self.instance
        exp = self.local_exp[label]

        mask = np.zeros(segments.shape, segments.dtype)

        if isinstance(instance, Tensor):
            instance = instance.numpy()

        temp = instance.copy()
        if positive_only:
            fs = [index for index, weight in exp
                  if weight > 0 and weight > min_weight][:num_features]
        if negative_only:
            fs = [index for index, weight in exp
                  if weight < 0 and abs(weight) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = instance[segments == f].copy()
                mask[segments == f] = 1
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = instance[segments == f].copy()
                temp[segments == f, c] = np.max(instance)
        return temp[0], mask[0]
