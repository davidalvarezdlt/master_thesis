"""
Module containing the VGG feature extractor.
"""
import torch
import torch.hub
import torch.nn as nn
import torchvision.models
import torchvision.models.vgg


class VGGFeatures(torchvision.models.VGG):
    """Implementation of the VGG Features extractor.

    Attributes:
        vgg_mean: Tensor containing the mean for image normalization.
        vgg_std: Tensor containing the standard deviation for image
            normalization.
    """
    def __init__(self, features, device):
        super().__init__(features)
        self.vgg_mean = torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1) \
            .to(device)
        self.vgg_std = torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1) \
            .to(device)

    def forward(self, x, normalize_input=True):
        """Forward pass through the VGG Features extractor.

        Args:
            x: Tensor of size (B,3,H,W) containing input images.
            normalize_input: Whether or not to normalize the input images
                ``x``.

        Returns:
            List of tensors containing VGG features.
        """
        if normalize_input:
            x = (x - self.vgg_mean) / self.vgg_std
        pool_feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                pool_feats.append(x.detach())
        return pool_feats

    @staticmethod
    def get_pretrained_model(device):
        """Returns an instance of ``VGGFeatures`` with loaded state.

        Args:
            device: Identifier of the device where the model should be moved.

        Returns:
            Instance of ``VGGFeatures`` with loaded state.
        """
        vgg_cfgs = torchvision.models.vgg.cfgs['D']
        model = VGGFeatures(
            torchvision.models.vgg.make_layers(vgg_cfgs, batch_norm=False),
            device
        )

        state_dict = torch.hub.load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg16']
        )
        model.load_state_dict(state_dict)

        return model
