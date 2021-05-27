import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, D=1, activation=nn.ReLU()):
        super(Conv2d, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


# Alignment Encoder
class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv34 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 8
        self.conv4a = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 8
        self.conv4b = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 8
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        return x


# Alignment Regressor
class A_Regressor(nn.Module):
    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 16
        self.conv5a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 16
        self.conv5b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 16
        self.conv56 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 32
        self.conv6a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 32
        self.conv6b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 32
        init_He(self)

        self.fc = nn.Linear(512, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

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

        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)

        return theta


# Encoder (Copy network)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.value3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None)  # 4
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        v = self.value3(x)
        return v


# Decoder (Paste network)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())

        # dilated convolution blocks
        self.convA4_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(257, 257, kernel_size=3, stride=1, padding=8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=16, D=16, activation=nn.ReLU())

        self.conv3c = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv3b = Conv2d(257, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv3a = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv32 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv21 = Conv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None)  # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
        x = F.upsample(x, scale_factor=2, mode='nearest')  # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')  # 2
        x = self.conv21(x)

        p = (x * self.std) + self.mean
        return p


# Context Matching Module
class CM_Module(nn.Module):
    def __init__(self):
        super(CM_Module, self).__init__()

    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums < 1e-4)
        masked_sums += zeros.float()
        return masked_exps / masked_sums

    def forward(self, values, tvmap, rvmaps):
        B, C, T, H, W = values.size()
        # t_feat: target feature
        t_feat = values[:, :, 0]
        # r_feats: refetence features
        r_feats = values[:, :, 1:]

        B, Cv, T, H, W = r_feats.size()
        # vmap: visibility map
        # tvmap: target visibility map
        # rvmap: reference visibility map
        # gs: cosine similarity
        # c_m: c_match
        gs_, vmap_ = [], []
        tvmap_t = (F.upsample(tvmap, size=(H, W), mode='bilinear', align_corners=False) > 0.5).float()
        for r in range(T):
            rvmap_t = (F.upsample(rvmaps[:, :, r], size=(H, W), mode='bilinear', align_corners=False) > 0.5).float()
            # vmap: visibility map
            vmap = tvmap_t * rvmap_t
            gs = (vmap * t_feat * r_feats[:, :, r]).sum(-1).sum(-1).sum(-1)
            # valid sum
            v_sum = vmap[:, 0].sum(-1).sum(-1)
            zeros = (v_sum < 1e-4)
            gs[zeros] = 0
            v_sum += zeros.float()
            gs = gs / v_sum / C
            gs = torch.ones(t_feat.shape).float().to(t_feat.device) * gs.view(B, 1, 1, 1)
            gs_.append(gs)
            vmap_.append(rvmap_t)

        gss = torch.stack(gs_, dim=2)
        vmaps = torch.stack(vmap_, dim=2)

        # weighted pixelwise masked softmax
        c_match = self.masked_softmax(gss, vmaps, dim=2)
        c_out = torch.sum(r_feats * c_match, dim=2)

        # c_mask
        c_mask = (c_match * vmaps)
        c_mask = torch.sum(c_mask, 2)
        c_mask = 1. - (torch.mean(c_mask, 1, keepdim=True))

        return torch.cat([t_feat, c_out, c_mask], dim=1), c_mask


class CPNOriginal(nn.Module):
    def __init__(self):
        super(CPNOriginal, self).__init__()
        self.A_Encoder = A_Encoder()  # Align
        self.A_Regressor = A_Regressor()  # output: alignment network
        self.Encoder = Encoder()  # Merge
        self.CM_Module = CM_Module()
        self.Decoder = Decoder()

    def forward(self, x_target, m_target, x_refs, m_refs):
        x_refs_aligned, v_refs_aligned, v_maps = self.align(x_target, m_target, x_refs, m_refs)
        y_hat, y_hat_comp = self.copy_and_paste(x_target, 1 - m_target, x_refs_aligned, v_refs_aligned)
        return y_hat, y_hat_comp

    def align(self, x_target, m_target, x_refs, m_refs):
        b, c, ref_n, h, w = x_refs.size()

        # Get alignment features
        x_target_feats = self.A_Encoder(x_target, m_target)
        x_refs_feats = self.A_Encoder(
            x_refs.transpose(1, 2).reshape(-1, c, h, w), m_refs.transpose(1, 2).reshape(-1, 1, h, w)
        ).reshape(b, ref_n, x_target_feats.size(1), x_target_feats.size(2), x_target_feats.size(3)).transpose(1, 2)

        # Get alignment grid
        theta_rt = self.A_Regressor(
            x_target_feats.unsqueeze(2).repeat(1, 1, ref_n, 1, 1).transpose(1, 2).reshape(
                -1, x_refs_feats.size(1), x_refs_feats.size(3), x_refs_feats.size(4)
            ),
            x_refs_feats.transpose(1, 2).reshape(-1, x_refs_feats.size(1), x_refs_feats.size(3), x_refs_feats.size(4))
        )
        grid_rt = F.affine_grid(theta_rt, (theta_rt.size(0), c, h, w), align_corners=False)

        # Align data
        x_aligned = F.grid_sample(
            x_refs.transpose(1, 2).reshape(-1, c, h, w), grid_rt, align_corners=False
        ).reshape(b, ref_n, c, h, w).transpose(1, 2)
        v_aligned = (F.grid_sample(
            1 - m_refs.transpose(1, 2).reshape(-1, 1, h, w), grid_rt, align_corners=False
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2) > 0.5).float()

        # Compute visibility map
        v_maps = (v_aligned - (1 - m_target.unsqueeze(2))).clamp(0, 1)

        # Return data
        return x_aligned, v_aligned, v_maps

    def copy_and_paste(self, x_target, v_target, x_aligned, v_aligned):
        b, c, f_ref, h, w = x_aligned.size()

        # Get c_features of everything
        c_feats = self.Encoder(
            torch.cat([x_target.unsqueeze(2), x_aligned], dim=2).transpose(1, 2).reshape(-1, c, h, w),
            torch.cat([1 - v_target.unsqueeze(2), v_aligned], dim=2).transpose(1, 2).reshape(-1, 1, h, w)
        )
        c_feats = c_feats.reshape(b, f_ref + 1, c_feats.size(1), c_feats.size(2), c_feats.size(3)).transpose(1, 2)

        # Apply Content-Matching Module
        p_in, c_mask = self.CM_Module(c_feats, v_target, v_aligned)

        # Upscale c_mask to match the size of the mask
        c_mask = (F.interpolate(c_mask, size=(h, w), mode='bilinear', align_corners=False)).detach()

        # Obtain the predicted output y_hat. Clip the output to be between [0, 1]
        y_hat = torch.clamp(self.Decoder(p_in), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = v_target * x_target + (1 - v_target) * y_hat

        # Return everything
        return y_hat, y_hat_comp
