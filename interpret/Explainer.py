# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#

import sys
from importlib import import_module
from os.path import dirname
sys.path.insert(1, dirname(__file__))


class Explainer:
    """
    Interface for model explainations
    """
    def __init__(self, explain_method, **kwargs):
        """
        Arguments
        ---------
        explain_method: str, names for dynamically initialize specified explainer.
        **kwargs:
            Other keyword arguments for initialize the explainer.
        """
        self.__available_methods = {
            "GradCam",
            "Lime"
        }
        self.__check_explain_methods(explain_method)
        module = import_module(explain_method)
        self.__explainer = getattr(module, explain_method)(**kwargs)

    @property
    def explainer(self):
        return self.__explainer

    @property
    def available_methods(self):
        return self.__available_methods

    def __check_explain_methods(self, method):
        if method not in self.__available_methods:
            raise NotImplementedError(
                print("Only the following methods are available: {}".format(self.__available_methods))
            )

    def explain_instance(self, instance, **kwargs):
        """
        Call the specified explainer to get explanation of given instance.

        Arguments
        ---------
        instance: numpy.ndarray or torch.Tensor depends on the explainer.
            Has shape of (1, num_channels, durations)
        **kwargs:
            Other keyword arguments for explaining the instance.

        Returns
        -------
        explanations of the given instance from the explainer.
        """
        return self.__explainer(instance, **kwargs)
