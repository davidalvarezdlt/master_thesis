import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nb=10, nf=64, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, stride=2, padding=1),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        )
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class RRDBNetSeparable(nn.Module):

    def __init__(self, in_nc, out_nc, nb=15, nf=64, gc=32):
        super(RRDBNetSeparable, self).__init__()

        # Encoding
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, stride=2, padding=1),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        )
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb // 3)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # After the encoding
        self.conv_second = nn.Conv2d(2 * nf + 1, nf, 3, padding=1)
        self.RRDB_trunk_second = make_layer(RRDB_block_f, nb // 2)
        self.trunk_conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x_target, x_ref_aligned, v_target, v_ref_aligned, v_map):
        # Encode the frames separately
        x_target_enc = self._encode_frame(x_target)
        x_ref_aligned_enc = self._encode_frame(x_ref_aligned)

        # Mask the encodings using the visibility maps
        x_target_enc = x_target_enc * F.interpolate(v_target, (x_target_enc.size(2), x_target_enc.size(3)))
        x_aligned_enc = x_ref_aligned_enc * F.interpolate(
            v_ref_aligned, (x_target_enc.size(2), x_target_enc.size(3))
        )
        v_map_resized = F.interpolate(v_map, (x_target_enc.size(2), x_target_enc.size(3)))

        # Concatenate the channels and propagate second part
        y_hat = self.conv_second(torch.cat((x_target_enc, x_aligned_enc, v_map_resized), dim=1))
        y_hat = self.trunk_conv_second(y_hat)

        # Upsample and return
        y_hat = self.lrelu(self.upconv1(F.interpolate(y_hat, scale_factor=2, mode='nearest')))
        y_hat = self.lrelu(self.upconv2(F.interpolate(y_hat, scale_factor=2, mode='nearest')))
        y_hat = self.lrelu(self.upconv1(y_hat))
        y_hat = self.lrelu(self.upconv2(y_hat))
        y_hat = self.conv_last(self.lrelu(self.HRconv(y_hat)))
        return y_hat

    def _encode_frame(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        return fea + trunk
