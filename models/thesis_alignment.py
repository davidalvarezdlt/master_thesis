import torch.nn as nn
import torch.nn.functional as F
import torch
import utils.movement
import utils.correlation


class SeparableConv4d(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super(SeparableConv4d, self).__init__()
        self.conv_1 = nn.Sequential(
            torch.nn.Conv2d(in_c, 128, 3, padding=1), nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1)
        )
        self.conv_2 = nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            torch.nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            torch.nn.Conv2d(128, out_c, 3, padding=1)
        )

    def forward(self, corr):
        corr = corr.unsqueeze(4)
        b, t, h, w, c, *_ = corr.size()

        # reshape (b*t*H*W, c, H, W)
        # shape is b, t, H*W, inter_dim, H*W then permute
        x2_bis = self.conv_1(corr.reshape(-1, c, h, w))
        x2_bis = x2_bis.reshape(b, t, h * w, x2_bis.size(1), h * w).permute(0, 1, 4, 3, 2)

        # reshape (b*t*H*W, inter_dim, H, W)
        x3_bis = self.conv_2(x2_bis.reshape(-1, x2_bis.size(3), h, w))
        x3_bis = x3_bis.reshape(b, t, h, w, x3_bis.size(1), h, w).squeeze(4)

        # Return last layer
        return x3_bis.permute(0, 1, 4, 5, 2, 3)


class Softmax3d(torch.nn.Module):

    def __init__(self):
        super(Softmax3d, self).__init__()

    def forward(self, input):
        assert input.dim() == 6  # Expect (B,T,H1,W1,H,W)

        # Get dimensions
        b, t, h, w, _, _ = input.size()

        # Transform input to be (B,H,W,H*W*T)
        input = input.permute(0, 2, 3, 4, 5, 1).reshape(b, h, w, -1)

        # Apply Softmax
        input = F.softmax(input, dim=3)

        # Restore original dimensions
        return input.reshape(b, h, w, h, w, t).permute(0, 5, 1, 2, 3, 4)


class CorrelationVGG(nn.Module):

    def __init__(self, model_vgg, use_softmax=False):
        super(CorrelationVGG, self).__init__()
        self.model_vgg = model_vgg
        self.conv = SeparableConv4d()
        self.softmax = Softmax3d() if use_softmax else None

    def forward(self, x_target, m_target, x_ref, m_ref):
        b, c, ref_n, h, w = x_ref.size()

        # Get the features of the frames from VGG
        with torch.no_grad():
            x_target_feats = self.model_vgg(x_target, normalize_input=False)
            x_ref_feats = self.model_vgg(x_ref.transpose(1, 2).reshape(b * ref_n, c, h, w), normalize_input=False)
        x_target_feats, x_ref_feats = x_target_feats[3], x_ref_feats[3].reshape(b, ref_n, -1, 16, 16).transpose(1, 2)

        # Update the parameters to the VGG features
        b, c, ref_n, h, w = x_ref_feats.size()

        # Interpolate the feature masks
        v_target = F.interpolate(1 - m_target, size=(h, w), mode='nearest')
        v_ref = F.interpolate(
            1 - m_ref.transpose(1, 2).reshape(b * ref_n, 1, m_ref.size(3), m_ref.size(4)), size=(h, w), mode='nearest'
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2)

        # Compute the feature correlation
        corr = utils.correlation.compute_masked_4d_correlation(x_target_feats, v_target, x_ref_feats, v_ref)

        # Fill holes in the correlation matrix using a NN
        corr = self.conv(corr)

        # Compute the Softmax over each pixel (b, t, h, w, h, w)
        return self.softmax(corr) if self.softmax else corr


class AlignmentCorrelationMixer(nn.Module):
    def __init__(self, corr_size=16):
        super(AlignmentCorrelationMixer, self).__init__()
        assert corr_size == 16
        self.mixer = nn.Sequential(
            nn.Conv2d(corr_size ** 2, corr_size ** 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size ** 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size, kernel_size=3, padding=1), nn.ReLU(),  # Out = 16
            nn.Conv2d(corr_size, corr_size, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size // 2, kernel_size=3, padding=1), nn.ReLU(),  # Out = 8
            nn.Conv2d(corr_size // 2, corr_size // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 4, kernel_size=3, padding=1), nn.ReLU(),  # Out = 4
            nn.Conv2d(corr_size // 4, corr_size // 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 8, kernel_size=3, padding=1),  # Out = 2
            nn.Conv2d(corr_size // 8, corr_size // 8, kernel_size=5, padding=2),
            nn.Conv2d(corr_size // 8, corr_size // 8, kernel_size=3, padding=1)
        )

    def forward(self, corr):
        b, f, h, w, *_ = corr.size()
        corr = corr.reshape(b * f, -1, 16, 16)
        return self.mixer(corr).reshape(b, f, 2, h, w).permute(0, 1, 3, 4, 2)


class FlowEstimator(nn.Module):
    def __init__(self, in_c=10):
        super(FlowEstimator, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_c, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=5, padding=2, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, padding=1)
        )

    def forward(self, x_target, m_target, x_ref, m_ref, flow_pre):
        b, c, ref_n, h, w = x_ref.size()

        # Prepare data and propagate it through the model
        nn_input = torch.cat([
            x_ref.transpose(1, 2).reshape(b * ref_n, c, h, w),
            x_target.unsqueeze(1).repeat(1, ref_n, 1, 1, 1).reshape(b * ref_n, c, h, w),
            m_ref.transpose(1, 2).reshape(b * ref_n, 1, h, w),
            m_target.unsqueeze(1).repeat(1, ref_n, 1, 1, 1).reshape(b * ref_n, 1, h, w),
            flow_pre.reshape(b * ref_n, h, w, 2).permute(0, 3, 1, 2)
        ], dim=1)

        # Return flow in the form (B,F,H,W,2)
        return self.nn(nn_input).reshape(b, ref_n, 2, h, w).permute(0, 1, 3, 4, 2)


class ThesisAlignmentModel(nn.Module):

    def __init__(self, model_vgg):
        super(ThesisAlignmentModel, self).__init__()
        self.corr = CorrelationVGG(model_vgg)
        self.corr_mixer = AlignmentCorrelationMixer()
        self.flow_64 = FlowEstimator()
        self.flow_256 = FlowEstimator()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x_target, m_target, x_refs, m_refs):
        b, c, ref_n, h, w = x_refs.size()

        # Normalize the input
        x_target = (x_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        x_refs = (x_refs - self.mean) / self.std

        # Resize the data to be squared 256x256
        x_target_sq, m_target_sq, x_ref_sq, m_ref_sq = self.interpolate_data(x_target, m_target, x_refs, m_refs, 256, 256)

        # Apply the CorrelationVGG module. Corr is (b, t, h, w, h, w)
        corr = self.corr(x_target_sq, m_target_sq, x_ref_sq, m_ref_sq)

        # Mix the corr 4D volume to obtain a 16x16 dense flow estimation of size (b, t, 16, 16, 2)
        flow_16 = self.corr_mixer(corr)

        # Interpolate x, m and flow_16 to be 64xW
        x_target_64, m_target_64, x_ref_64, m_ref_64 = self.interpolate_data(x_target, m_target, x_refs, m_refs, 64, 64)
        flow_64_pre = self.interpolate_flow(flow_16, 64, 64)

        # Estimate 64x64 flow correction of size (b, t, 64, 64, 2)
        flow_64 = self.flow_64(x_target_64, m_target_64, x_ref_64, m_ref_64, flow_64_pre)

        # Interpolate flow_64 to be 256x256
        flow_256_pre = self.interpolate_flow(flow_64, 256, 256)

        # Estimate 256x256 flow correction of size (b, t, 256, 256, 2)
        flow_256 = self.flow_256(x_target_sq, m_target_sq, x_ref_sq, m_ref_sq, flow_256_pre)

        # Return both corr and corr_mixed
        return corr, flow_16, flow_64, self.interpolate_flow(flow_256, h, w)

    def interpolate_data(self, x_target, m_target, x_ref, m_ref, h_new, w_new):
        b, c, ref_n, h, w = x_ref.size()
        if h == h_new and w == w_new:
            return x_target, m_target, x_ref, m_ref
        x_target_res = F.interpolate(x_target, (h_new, w_new), mode='bilinear')
        m_target_res = F.interpolate(m_target, (h_new, w_new), mode='nearest')
        x_ref_res = F.interpolate(x_ref.transpose(1, 2).reshape(b * ref_n, c, h, w), (h_new, w_new), mode='bilinear') \
            .reshape(b, ref_n, c, h_new, w_new).transpose(1, 2)
        m_ref_res = F.interpolate(m_ref.transpose(1, 2).reshape(b * ref_n, 1, h, w), (h_new, w_new), mode='nearest')\
            .reshape(b, ref_n, 1, h_new, w_new).transpose(1, 2)
        return x_target_res, m_target_res, x_ref_res, m_ref_res

    def interpolate_flow(self, flow, h_new, w_new):
        b, ref_n, h, w, _ = flow.size()
        return F.interpolate(
            flow.reshape(b * ref_n, flow.size(2), flow.size(3), 2).permute(0, 3, 1, 2), (h_new, w_new), mode='bilinear'
        ).reshape(b, ref_n, 2, h_new, w_new).permute(0, 1, 3, 4, 2)

