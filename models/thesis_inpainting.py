import torch
import torch.nn as nn
import models.rrdb_net


class ThesisInpaintingVisible(nn.Module):

    def __init__(self, in_c=9):
        super(ThesisInpaintingVisible, self).__init__()
        self.nn = models.rrdb_net.RRDBNet(in_nc=in_c, out_nc=3, nb=20)
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x_target, v_target, x_refs_aligned, v_refs_aligned, v_maps):
        b, c, f, h, w = x_refs_aligned.size()

        # Unsqueeze dimensions
        x_target = x_target.unsqueeze(2).repeat(1, 1, f, 1, 1)
        v_target = v_target.unsqueeze(2).repeat(1, 1, f, 1, 1)

        # Normalize the input images
        x_target_norm = (x_target - self.mean) / self.std
        x_ref_aligned_norm = (x_refs_aligned - self.mean) / self.std

        # Predict output depending on the NN
        nn_input = torch.cat([x_target_norm, x_ref_aligned_norm, v_target, v_refs_aligned, v_maps], dim=1)
        nn_output = self.nn(
            nn_input.transpose(1, 2).reshape(b * f, 9, h, w)
        ).reshape(b, f, c, h, w).transpose(1, 2)

        # Propagate data through the NN
        y_hat = torch.clamp(nn_output * self.std + self.mean, 0, 1)
        y_hat_comp = v_target * x_target + (1 - v_target) * y_hat

        # Return the data
        return y_hat, y_hat_comp
