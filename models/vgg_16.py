import ssl
import torch
import torch.nn as nn
import torchvision.models
import torchvision.models.utils
from torchvision.models.vgg import cfgs, make_layers, model_urls

ssl._create_default_https_context = ssl._create_unverified_context


class VGGFeatures(torchvision.models.VGG):

    def __init__(self, features, device):
        super().__init__(features)
        self.vgg_mean = torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.vgg_std = torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def forward(self, x, normalize_input=True):
        if normalize_input:
            x = (x - self.vgg_mean) / self.vgg_std
        pool_feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                pool_feats.append(x.detach())
        return pool_feats


def get_pretrained_model(device):
    model = VGGFeatures(make_layers(cfgs['D'], batch_norm=False), device)
    state_dict = torchvision.models.utils.load_state_dict_from_url(model_urls['vgg16'])
    model.load_state_dict(state_dict)
    return model.to(device)
