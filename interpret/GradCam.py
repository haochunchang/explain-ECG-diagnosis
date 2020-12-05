# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#
# GradCam: Gradient-weighted Class Activation Mapping
#
# Ref: https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py

import numpy as np
import torch
from torchvision import transforms


class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.model.eval()
        self.extractor = ModelOutputs(
            model=model,
            feature_module=feature_module,
            target_layers=target_layer_names
        )
        self.feature_module = feature_module

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        features, output = self.extractor(input)
        if index is None:
            index = np.argmax(output.data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].data.numpy()
        # shape: (n_sample, n_channel, length)
        target = features[-1]
        # shape: (n_sample, n_channel, length)
        target = target.data.numpy()

        #TODO: Fix warning of non-writeable tensors.
        weights = np.mean(grads_val, axis=2)[0, :]
        cam = weights.reshape((-1, 1)) * target[0, :]
        resize_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((15, 2048)),
            transforms.ToTensor()
        ])
        cam = resize_transforms(cam).numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers.
    """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, x = self.feature_extractor(x)

        for name, module in self.model._modules.items():
            if name not in self.feature_extractor.model._modules:
                x = module(x)
        return target_activations, x
