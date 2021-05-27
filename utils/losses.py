from models.vgg_16 import get_pretrained_model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LossesUtils:
    model_vgg = None
    grad_horz = None
    grad_vert = None

    def __init__(self, model_vgg, device):
        self.model_vgg = model_vgg
        self.grad_horz = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat((3, 1, 1, 1)).to(device)
        self.grad_vert = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat((3, 1, 1, 1)).to(device)

    def init_vgg(self, device):
        self.model_vgg = get_pretrained_model(device)
        for param in self.model_vgg.features.parameters():
            param.requires_grad = False

    def masked_l1(self, input, target, mask, batch_mask=None, reduction='mean', weight=1):
        if batch_mask is not None and not any(batch_mask):
            return torch.zeros(1).to(input.device)
        elif batch_mask is not None:
            input, target, mask = input[batch_mask], target[batch_mask], mask[batch_mask]
        masked_l1_loss = F.l1_loss(input * mask, target * mask, reduction=reduction)
        return weight * masked_l1_loss / (torch.sum(mask) + 1E-9 if reduction == 'sum' else 1)

    def bce(self, input, target, mask, batch_mask=None, reduction='mean', weight=1):
        if batch_mask is not None:
            input, target, mask = input[batch_mask], target[batch_mask], mask[batch_mask]
        bce_loss = F.binary_cross_entropy(input * mask, target * mask, reduction=reduction)
        return weight * bce_loss / (torch.sum(mask) if reduction == 'sum' else 1)

    def perceptual(self, input, target, weight=1):
        input_vgg = self.model_vgg(input.contiguous())
        target_vgg = self.model_vgg(target.contiguous())
        loss_perceptual = 0
        for p in range(len(input_vgg)):
            loss_perceptual += F.l1_loss(input_vgg[p], target_vgg[p])
        return loss_perceptual * weight / len(input_vgg), input_vgg, target_vgg

    def style(self, x_vgg, x_hat_vgg, weight=1):
        loss_style = 0
        for p in range(len(x_vgg)):
            b, c, h, w = x_vgg[p].size()
            g_x = torch.mm(x_vgg[p].view(b * c, h * w), x_vgg[p].view(b * c, h * w).t())
            g_x_comp = torch.mm(x_hat_vgg[p].view(b * c, h * w), x_hat_vgg[p].view(b * c, h * w).t())
            loss_style += F.l1_loss(g_x_comp / (b * c * h * w), g_x / (b * c * h * w))
        return loss_style * weight / len(x_vgg)

    def tv(self, x_hat, weight=1):
        loss_tv_h = (x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]).pow(2).sum()
        loss_tv_w = (x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]).pow(2).sum()
        loss_tv = (loss_tv_h + loss_tv_w) / (x_hat.size(0) * x_hat.size(1) * x_hat.size(2) * x_hat.size(3))
        return loss_tv * weight

    def grad(self, input, target, reduction, weight=1):
        input_grads = torch.cat((self._grad_horizontal(input), self._grad_vertical(input)), dim=1)
        target_grads = torch.cat((self._grad_horizontal(target), self._grad_vertical(target)), dim=1)
        mask = torch.ones_like(input_grads).to(input.device)
        return self.masked_l1(input_grads, target_grads, mask, batch_mask=None, reduction=reduction, weight=weight)

    def _grad_horizontal(self, x):
        return F.conv2d(x, padding=1, weight=self.grad_horz, groups=3)

    def _grad_vertical(self, x):
        return F.conv2d(x, padding=1, weight=self.grad_vert, groups=3)
