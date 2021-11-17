"""
Module containing the ``pytorch_lightning.LightningModule`` implementation of
the of the Copy-and-Paste Network (CPN).

Check the original implementation in:
https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPN(nn.Module):
    """Implementation of the Copy-and-Paste Network (CPN).

    Attributes:
        A_Encoder: Instance of a ``A_Encoder`` layer.
        A_Regressor: Instance of a ``A_Regressor`` layer.
        Encoder: Instance of a ``Encoder`` layer.
        CM_Module: Instance of a ``CM_Module`` layer.
        Decoder: Instance of a ``Decoder`` layer.
    """
    def __init__(self):
        super(CPN, self).__init__()
        self.A_Encoder = A_Encoder()
        self.A_Regressor = A_Regressor()
        self.Encoder = Encoder()
        self.CM_Module = CM_Module()
        self.Decoder = Decoder()

    def align(self, x_target, m_target, x_refs, m_refs):
        """Aligns the images ``x_refs`` with respect to the image ``x_target``.

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the target image.
            m_target: Tensor of size ``(B,C,H,W)`` containing the mask of the
                target image.
            x_refs: Tensor of size ``(B,C,F,H,W)`` containing the reference
                images.
            m_refs: Tensor of size ``(B,1,F,H,W)`` containing the masks of the
                reference images.

        Returns:
            Tuple of three positions containing:
                - Tensor of size ``(B,C,F,H,W)`` containing the aligned
                    reference images.
                - Tensor of size ``(B,1,F,H,W)`` containing the aligned
                    visibility maps.
                - Tensor of size ``(B,1,F,H,W)`` containing a map indicating
                    which areas of the target frame are visible in the
                    reference frames.
        """
        b, c, ref_n, h, w = x_refs.size()

        x_target_feats = self.A_Encoder(x_target, m_target)
        x_refs_feats = self.A_Encoder(
            x_refs.transpose(1, 2).reshape(-1, c, h, w),
            m_refs.transpose(1, 2).reshape(-1, 1, h, w),
        ).reshape(
            b, ref_n, x_target_feats.size(1), x_target_feats.size(2),
            x_target_feats.size(3),
        ).transpose(1, 2)

        theta_rt = self.A_Regressor(
            x_target_feats.unsqueeze(2).repeat(1, 1, ref_n, 1, 1)
            .transpose(1, 2).reshape(
                -1, x_refs_feats.size(1), x_refs_feats.size(3),
                x_refs_feats.size(4)
            ),
            x_refs_feats.transpose(1, 2).reshape(
                -1, x_refs_feats.size(1), x_refs_feats.size(3),
                x_refs_feats.size(4)
            )
        )
        grid_rt = F.affine_grid(
            theta_rt, [theta_rt.size(0), c, h, w], align_corners=False
        )

        x_aligned = F.grid_sample(
            x_refs.transpose(1, 2).reshape(-1, c, h, w),
            grid_rt,
            align_corners=False,
        ).reshape(b, ref_n, c, h, w).transpose(1, 2)
        v_aligned = (F.grid_sample(
            1 - m_refs.transpose(1, 2).reshape(-1, 1, h, w),
            grid_rt,
            align_corners=False,
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2) > 0.5).float()
        v_maps = (v_aligned - (1 - m_target.unsqueeze(2))).clamp(0, 1)

        return x_aligned, v_aligned, v_maps

    @staticmethod
    def init_model_with_state(checkpoint_path, device='cpu'):
        """Returns an instance of the ``CPN`` network with loaded state.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Identifier of the device where the model should be
                allocated.

        Returns:
            Instance of a ``CPN`` network with loaded state.
        """
        model = CPN()
        checkpoint_data = dict(
            torch.load(checkpoint_path, map_location=device)
        )

        model_state = model.state_dict()
        for ck_item, k_data in checkpoint_data.items():
            if ck_item.replace('module.', '') in model_state:
                model_state[ck_item.replace('module.', '')].copy_(k_data)
        model.load_state_dict(model_state)

        for param in model.parameters():
            param.requires_grad = False
        return model.to(device)


class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, 5, 2, 2, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, 3, 1, 1, activation=nn.ReLU())
        self.conv23 = Conv2d(64, 128, 3, 2, 1, activation=nn.ReLU())
        self.conv3 = Conv2d(128, 128, 3, 1, 1, activation=nn.ReLU())
        self.conv34 = Conv2d(128, 256, 3, 2, 1, activation=nn.ReLU())
        self.conv4a = Conv2d(256, 256, 3, 1, 1, activation=nn.ReLU())
        self.conv4b = Conv2d(256, 256, 3, 1, 1, activation=nn.ReLU())
        self.register_buffer(
            'mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.upsample(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        return self.conv4b(x)


class A_Regressor(nn.Module):
    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, 3, 2, 1, activation=nn.ReLU())
        self.conv5a = Conv2d(512, 512, 3, 1, 1, activation=nn.ReLU())
        self.conv5b = Conv2d(512, 512, 3, 1, 1, activation=nn.ReLU())
        self.conv56 = Conv2d(512, 512, 3, 2, 1, activation=nn.ReLU())
        self.conv6a = Conv2d(512, 512, 3, 1, 1, activation=nn.ReLU())
        self.conv6b = Conv2d(512, 512, 3, 1, 1, activation=nn.ReLU())
        self.fc = nn.Linear(512, 6)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])
        return self.fc(x).view(-1, 2, 3)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, 5, 2, 2, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, 3, 1, 1, activation=nn.ReLU())
        self.conv23 = Conv2d(64, 128, 3, 2, 1, activation=nn.ReLU())
        self.conv3 = Conv2d(128, 128, 3, 1, 1, activation=nn.ReLU())
        self.value3 = Conv2d(128, 128, 3, 1, 1, activation=None)
        self.register_buffer(
            'mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        return self.value3(x)


class CM_Module(nn.Module):
    def __init__(self):
        super(CM_Module, self).__init__()

    def forward(self, c_feats, v_t, v_aligned):
        b, c_c, f, h, w = c_feats.size()
        v_t_resized = (F.interpolate(
            v_t, size=(h, w), mode='bilinear', align_corners=False
        ) > 0.5).float()

        cos_sim, vr_map = [], []
        for r in range(f - 1):
            v_r = (F.interpolate(
                v_aligned[:, :, r], size=(h, w), mode='bilinear',
                align_corners=False
            ) > 0.5).float()
            vr_map.append(v_r)

            vmap = v_t_resized * v_r
            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            gs_norm = v_sum * c_c
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]) \
                .sum(-1).sum(-1).sum(-1) / gs_norm
            gs[v_sum_zeros] = 0
            cos_sim.append(
                torch.ones((b, c_c, h, w)).to(c_feats.device) *
                gs.view(b, 1, 1, 1)
            )

        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        c_match = CM_Module.masked_softmax(cos_sim, vr_map, dim=2)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        c_mask = torch.sum(c_match * vr_map, 2)
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)

        return torch.cat([c_feats[:, :, 0], c_out, c_mask], dim=1), c_mask

    @staticmethod
    def masked_softmax(vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = masked_sums < 1e-4
        masked_sums += zeros.float()
        return masked_exps / masked_sums


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(257, 257, 3, 1, 1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(257, 257, 3, 1, 1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(257, 257, 3, 1, 1, activation=nn.ReLU())
        self.convA4_1 = Conv2d(257, 257, 3, 1, 2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(257, 257, 3, 1, 4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(257, 257, 3, 1, 8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(257, 257, 3, 1, 16, D=16, activation=nn.ReLU())
        self.conv3c = Conv2d(257, 257, 3, 1, 1, activation=nn.ReLU())
        self.conv3b = Conv2d(257, 128, 3, 1, 1, activation=nn.ReLU())
        self.conv3a = Conv2d(128, 128, 3, 1, 1, activation=nn.ReLU())
        self.conv32 = Conv2d(128, 64, 3, 1, 1, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, 3, 1, 1, activation=nn.ReLU())
        self.conv21 = Conv2d(64, 3, 5, 1, 2, activation=None)
        self.register_buffer(
            'mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = self.conv4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.convA4_1(x)
        x = self.convA4_2(x)
        x = self.convA4_3(x)
        x = self.convA4_4(x)
        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv21(x)
        return (x * self.std) + self.mean


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), D=(1, 1), activation=None):
        super(Conv2d, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, D)
            )

    def forward(self, x):
        return self.conv(x)
