# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#
# ref: https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py

import torch
from torch.autograd import Function

import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor():
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


class ModelOutputs():
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


class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
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
        target = features[-1]
        target = target.data.numpy()[0, :]

        weights = np.mean(grads_val, axis=2)[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        cam.resize((2048,))
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            input,
            positive_mask
        )
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input),
                grad_output,
                positive_mask_1
            ),
            positive_mask_2
        )

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input)

        if index is None:
            index = np.argmax(output.data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input.grad.data.numpy()
        output = output[0, :, :]

        return output


def preprocess_signals(signals):
    """
    Preprocess signals as Torch Tensors
    """
    preprocessed_signals = signals.squeeze(0).view((signals.shape[0], 15, -1))
    return preprocessed_signals.requires_grad_(True)


def show_cam_on_image(sample, mask):
    # FIXME

    cam = mask + np.float32(sample)
    cam = (cam / np.max(cam) * 16)[0]
    fig, ax = plt.subplots(5, 3, figsize=(12, 8))
    ax = ax.flatten()
    for i in range(15):
        ax[i].plot(moving_average(cam[i, :], n=100))
    plt.savefig("./cam.jpg")


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
