import numpy as np
import torch
import skimage.metrics
import models.lpips
import math


class UtilsMeasures:
    model_lpips = None

    def init_lpips(self, device):
        self.model_lpips = models.lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu='cuda' in device)
        for param in self.model_lpips.parameters():
            param.requires_grad = False

    def destroy_lpips(self):
        self.model_lpips = None

    def psnr(self, input, target):
        """Computes the PSNR between two images.

        Args:
            input (torch.FloatTensor): tensor of size (C,F,H,W) containing predicted images.
            target (torch.FloatTensor): tensor of size (C,F,H,W) containing ground-truth images.
        """
        items_psnr = []
        for f in range(target.size(1)):
            items_psnr.append(skimage.metrics.peak_signal_noise_ratio(
                target[:, f].numpy(), input[:, f].numpy())
            )
        items_psnr = [100 if math.isnan(i_psnr) else i_psnr for i_psnr in items_psnr]
        return np.mean([item_psnr for item_psnr in items_psnr if not np.isinf(item_psnr) and not np.isnan(item_psnr)])

    def ssim(self, input, target):
        """Computes the SSIM between two images.

        Args:
            input (torch.FloatTensor): tensor of size (C,F,H,W) containing predicted images.
            target (torch.FloatTensor): tensor of size (C,F,H,W) containing ground-truth images.
        """
        items_ssim = []
        for f in range(target.size(1)):
            items_ssim.append(skimage.metrics.structural_similarity(
                target[:, f].permute(1, 2, 0).numpy(), input[:, f].permute(1, 2, 0).numpy(), multichannel=True)
            )
        return np.mean(items_ssim)

    def lpips(self, input, target):
        """Computes the LPIPS between two images.

        Args:
            input (torch.FloatTensor): tensor of size (C,F,H,W) containing predicted images.
            target (torch.FloatTensor): tensor of size (C,F,H,W) containing ground-truth images.
        """
        with torch.no_grad():
            return np.mean(self.model_lpips.forward(
                input.transpose(0, 1), target.transpose(0, 1), normalize=True
            ).flatten().cpu().tolist())
