"""
Module containing the ``pytorch_lightning.LightningModule`` implementation of
the of the Copy-and-Hallucinate Network (DFPN).
"""
import functools
import os.path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import master_thesis as mt


class CHN(pl.LightningModule):
    """Implementation of the Copy-and-Hallucinate Network (CHN).

    Attributes:
        nn: Instance of a ``RRDBNet`` layer.
        model_vgg: Instance of a ``VGGFeatures`` network.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """
    LOSSES_NAMES = ['loss_nh', 'loss_vh', 'loss_nvh', 'loss_perceptual',
                    'loss_grad']

    def __init__(self, model_vgg, model_lpips, model_aligner, **kwargs):
        super(CHN, self).__init__()
        self.nn = RRDBNet(in_nc=9, out_nc=3, nb=20)
        self.register_buffer(
            'mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            'std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        )

        self.model_vgg = model_vgg
        self.model_lpips = model_lpips
        self.model_aligner = model_aligner
        self.kwargs = kwargs

    def forward(self, x_target, v_target, x_refs_aligned, v_refs_aligned,
                v_maps):
        """Forward pass through the Copy-and-Hallucinate Network (CHN).

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the frame to
                inpaint.
            v_target: Tensor of size ``(B,1,H,W)`` containing the visibility
                map of ``x_target``.
            x_refs_aligned: Tensor of size ``(B,C,F,H,W)`` containing the
                reference frames.
            v_refs_aligned: Tensor of size ``(B,1,F,H,W)`` containing the
                visibility maps of ``x_refs_aligned``.
            v_maps: Tensor of size ``(B,1,F,H,W)`` containing a map indicating
                which areas of the target frame are visible in the reference
                frames.

        Returns:
            Tuple of two positions containing:
                - Tensor of size ``(B,C,F,H,W)`` containing the output of the
                    network.
                - Tensor of size ``(B,C,F,H,W)`` containing the combination of
                    network outputs and ground-truth backgrounds.
        """
        b, c, f, h, w = x_refs_aligned.size()

        x_target = x_target.unsqueeze(2).repeat(1, 1, f, 1, 1)
        v_target = v_target.unsqueeze(2).repeat(1, 1, f, 1, 1)

        x_target_norm = (x_target - self.mean) / self.std
        x_ref_aligned_norm = (x_refs_aligned - self.mean) / self.std

        nn_input = torch.cat(
            [x_target_norm, x_ref_aligned_norm, v_target, v_refs_aligned,
             v_maps], dim=1
        )
        nn_output = self.nn(nn_input.transpose(1, 2).reshape(b * f, 9, h, w)) \
            .reshape(b, f, c, h, w).transpose(1, 2)

        y_hat = torch.clamp(nn_output * self.std + self.mean, 0, 1)
        y_hat_comp = v_target * x_target + (1 - v_target) * y_hat
        return y_hat, y_hat_comp

    def inpaint_ff(self, x, m, s=1, D=20, e=1):
        """Inpaints a sequence using the Frame-by-Frame Algorithm.

        Args:
            x: Tensor of size ``(C,F,H,W)`` containing the sequence to inpaint.
            m: Tensor of size ``(C,1,H,W)`` containing the mask of the object
                to inpaint.
            s: Number indicating the minimum distance between frames.
            D: Number indicating the maximum distance between frames.
            e: Number indicating the maximum percentage of unfilled hole.

        Returns:
            Tensor of size ``(C,F,H,W)`` containing the inpainted sequence.
        """
        fill_color = torch.as_tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1).to(x.device)

        y_inpainted = torch.zeros_like(x)
        for t in range(x.size(1)):
            x_target, m_target = x[:, t].unsqueeze(0), m[:, t].unsqueeze(0)
            y_hat_comp = None

            inp_per = 0
            t_candidates = CHN.get_indexes_ff(t, x.size(1), s=s, D=D)
            while (len(t_candidates) > 0 and inp_per > e) \
                    or y_hat_comp is None:
                r_list = [t_candidates.pop(0)]
                x_ref_aligned, v_ref_aligned, v_map = self.model_aligner.align(
                    x_target,
                    m_target,
                    x[:, r_list].unsqueeze(0),
                    m[:, r_list].unsqueeze(0)
                )
                y_hat, y_hat_comp = self(
                    x_target,
                    1 - m_target,
                    x_ref_aligned,
                    v_ref_aligned,
                    v_map
                )
                m_target = m_target - v_map[:, :, 0]
                x_target = (1 - m_target) * y_hat_comp[:, :, 0] + \
                    m_target.repeat(1, 3, 1, 1) * fill_color
                inp_per = torch.sum(m_target) * 100 / m_target.numel()

            y_inpainted[:, t] = y_hat_comp[:, :, 0]

        return y_inpainted

    def inpaint_ip(self, x, m, s=1, D=20, e=1):
        """Inpaints a sequence using the Inpaint and Propagate Algorithm.

        Args:
            x: Tensor of size ``(C,F,H,W)`` containing the sequence to inpaint.
            m: Tensor of size ``(C,1,H,W)`` containing the mask of the object
                to inpaint.
            s: Number indicating the minimum distance between frames.
            D: Number indicating the maximum distance between frames.
            e: Number indicating the maximum percentage of unfilled hole.

        Returns:
            Tensor of size ``(C,F,H,W)`` containing the inpainted sequence.
        """
        fill_color = torch.as_tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1).to(x.device)

        y_inpainted, m_inpainted = x.unsqueeze(0), m.unsqueeze(0)
        t_list = sorted(
            list(range(x.size(1))), key=lambda xi: abs(xi - x.size(1) // 2)
        )
        for t in t_list:
            t_candidates = CHN.get_indexes_ip(t, t_list, s, D)
            y_hat_comp, inp_per = None, 0

            while (len(t_candidates) > 0 and inp_per > e) \
                    or y_hat_comp is None:
                r_list = [t_candidates.pop(0)]

                x_ref_aligned, v_ref_aligned, v_map = self.model_aligner.align(
                    y_inpainted[:, :, t],
                    m_inpainted[:, :, t],
                    y_inpainted[:, :, r_list],
                    m_inpainted[:, :, r_list]
                )
                y_hat, y_hat_comp = self(
                    y_inpainted[:, :, t],
                    1 - m_inpainted[:, :, t],
                    x_ref_aligned,
                    v_ref_aligned,
                    v_map
                )

                m_inpainted[:, :, t] = m_inpainted[:, :, t] - v_map[:, :, 0]
                y_inpainted[:, :, t] = \
                    (1 - m_inpainted[:, :, t]) * y_hat_comp[:, :, 0] + \
                    m_inpainted[:, :, t].repeat(1, 3, 1, 1) * fill_color
                inp_per = torch.sum(m_inpainted[:, :, t]) * 100 \
                    / m_inpainted[:, :, t].numel()

            m_inpainted[:, :, t] = 0
            y_inpainted[:, :, t] = y_hat_comp[:, :, 0]

        return y_inpainted[0]

    def inpaint_cp(self, x, m, N=20, s=1, e=1):
        """Inpaints a sequence using the Copy and Propagate Algorithm.

        Args:
            x: Tensor of size ``(C,F,H,W)`` containing the sequence to inpaint.
            m: Tensor of size ``(C,1,H,W)`` containing the mask of the object
                to inpaint.
            N: Number indicating the maximum number of iterations.
            s: Number indicating the spacing between frames.
            e: Number indicating the maximum percentage of unfilled hole.

        Returns:
            Tensor of size ``(C,F,H,W)`` containing the inpainted sequence.
        """
        fill_color = torch.as_tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1).to(x.device)

        y_inpainted, m_inpainted = x.unsqueeze(0), m.unsqueeze(0)
        for i in range(N):
            t_list = [
                t for t in range(y_inpainted.size(2))
                if (t // s) % (s if s > 1 else 2) == i % 2
            ]

            for t in t_list:
                if m_inpainted[:, :, t].sum() == 0:
                    continue

                for delta_t in [-s, s]:
                    if not 0 <= t + delta_t < y_inpainted.size(2):
                        continue
                    r_list = [t + delta_t]

                    x_ref_aligned, v_ref_aligned, v_map = \
                        self.model_aligner.align(
                            y_inpainted[:, :, t],
                            m_inpainted[:, :, t],
                            y_inpainted[:, :, r_list],
                            m_inpainted[:, :, r_list]
                        )
                    y_hat, y_hat_comp = self(
                        y_inpainted[:, :, t],
                        1 - m_inpainted[:, :, t],
                        x_ref_aligned,
                        v_ref_aligned,
                        v_map
                    )

                    m_inpainted[:, :, t] = \
                        m_inpainted[:, :, t] - v_map[:, :, 0]
                    y_inpainted[:, :, t] = \
                        (1 - m_inpainted[:, :, t]) * y_hat_comp[:, :, 0] + \
                        m_inpainted[:, :, t].repeat(1, 3, 1, 1) * fill_color
                    inp_per = torch.sum(m_inpainted[:, :, t]) * 100 / \
                        m_inpainted[:, :, t].numel()

                    if inp_per < e or i >= N - 2:
                        m_inpainted[:, :, t] = 0
                        y_inpainted[:, :, t] = y_hat_comp[:, :, 0]

        return y_inpainted[0]

    def training_step(self, batch, batch_idx):
        """Performs a single pass through the training dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        (x, m), y, info = batch
        t, r_list = CHN.get_indexes(x.size(2))

        x_ref_aligned, v_ref_aligned, v_map = self.model_aligner.align(
            x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        y_hat, y_hat_comp = self(
            x[:, :, t], 1 - m[:, :, t], x_ref_aligned, v_ref_aligned, v_map
        )

        loss, loss_items = self.compute_loss(
            y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )
        self._log_losses(loss, loss_items, 'training')
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single pass through the validation dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        (x, m), y, info = batch
        t, r_list = CHN.get_indexes(x.size(2))

        x_ref_aligned, v_ref_aligned, v_map = self.model_aligner.align(
            x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        y_hat, y_hat_comp = self(
            x[:, :, t], 1 - m[:, :, t], x_ref_aligned, v_ref_aligned, v_map
        )

        loss, loss_items = self.compute_loss(
            y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )
        self._log_losses(loss, loss_items, 'validation')
        self._log_metrics(y_hat_comp[:, :, 0], y[:, :, t], 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single pass through the test dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.
        """
        (x, m), y, info = batch
        y_hat_comp_ff = self.inpaint_ff(x[0], m[0])
        y_hat_comp_ip = self.inpaint_ip(x[0], m[0])
        y_hat_comp_cp = self.inpaint_cp(x[0], m[0])
        self._log_sequence(y_hat_comp_ff, info[0][0] + '_ff')
        self._log_sequence(y_hat_comp_ip, info[0][0] + '_ip')
        self._log_sequence(y_hat_comp_cp, info[0][0] + '_cp')

    def compute_loss(self, y_target, v_target, y_hat, y_hat_comp, v_map):
        """Computes the loss of the Copy-and-Hallucinate Network (CHN).

        Args:
            y_target: Tensor of size ``(B,C,H,W)`` containing the ground-truth
                frame.
            v_target: Tensor of size ``(B,1,H,W)`` containing the visibility
                map of ``y_target``.
            y_hat: Tensor of size ``(B,C,F,H,W)`` containing the output of the
                network.
            y_hat_comp: Tensor of size ``(B,C,F,H,W)`` containing the
                combination of network outputs and ground-truth backgrounds.
            v_map: Tensor of size ``(B,1,F,H,W)`` containing a map indicating
                which areas of the target frame are visible in the reference
                frames.

        Returns:
            Tuple of two positions containing:
                - The sum of the different losses.
                - List containing the different loss items in the same order as
                    ``CHN.LOSSES_NAMES``.
        """
        b, c, h, w = y_target.size()
        target_img = y_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)

        nh_mask = v_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        loss_nh = mt.LossesUtils.masked_l1(
            y_hat, target_img, nh_mask, reduction='sum', weight=0.50
        )

        vh_mask = v_map
        loss_vh = mt.LossesUtils.masked_l1(
            y_hat, target_img, vh_mask, reduction='sum', weight=2
        )

        nvh_mask = (1 - nh_mask) - vh_mask
        loss_nvh = mt.LossesUtils.masked_l1(
            y_hat_comp, target_img, nvh_mask, reduction='sum', weight=1
        )

        loss_perceptual, *_ = mt.LossesUtils.perceptual(
            y_hat.transpose(1, 2).reshape(-1, c, h, w),
            target_img.transpose(1, 2).reshape(-1, c, h, w),
            model_vgg=self.model_vgg,
            weight=0.50,
        )

        loss_grad = mt.LossesUtils.grad(
            y_hat.squeeze(2), target_img.squeeze(2), reduction='mean', weight=1
        )

        loss = loss_nh + loss_vh + loss_nvh + loss_perceptual + loss_grad
        return loss, [loss_nh, loss_vh, loss_nvh, loss_perceptual, loss_grad]

    def configure_optimizers(self):
        """Configures the optimizer and LR scheduler used in the package.

        Returns:
            Dictionary containing a configured ``torch.optim.Adam``
            optimizer and ``torch.optim.lr_scheduler.StepLR`` scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.kwargs['lr_scheduler_step_size'],
            gamma=self.kwargs['lr_scheduler_gamma']
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def _log_losses(self, loss, loss_items, split):
        """Logs the losses in TensorBoard.

        Args:
            loss: Tensor containing the loss between predictions and ground
                truths.
            loss_items: Dictionary containing a tensor for every different
                loss.
            split: Identifier of the data split.
        """
        self.log('loss_{}'.format(split), loss)
        for i, loss_item_id in enumerate(CHN.LOSSES_NAMES):
            loss_name = 'loss_{}_{}'.format(loss_item_id, split)
            self.log(loss_name, loss_items[i])

    def _log_metrics(self, y_hat_comp, y_target, split):
        """Logs the metrics in TensorBoard.

        Args:
            y_hat_comp: Tensor of size ``(B,C,H,W)`` containing the
                combination of network output and ground-truth background.
            y_target: Tensor of size ``(B,C,H,W)`` containing the ground-truth
                frame.
        """
        psnr = mt.MeasuresUtils.psnr(y_hat_comp, y_target)
        ssim = mt.MeasuresUtils.ssim(y_hat_comp, y_target)
        lpips = mt.MeasuresUtils.lpips(y_hat_comp, y_target, self.model_lpips)
        self.log('measures_psnr_{}'.format(split), psnr)
        self.log('measures_ssim_{}'.format(split), ssim)
        self.log('measures_lpips_{}'.format(split), lpips)

    def _log_sequence(self, x, file_name):
        """Saves a set of samples.

        Args:
            x: Tensor of size ``(C,F,H,W)`` containing the sequence to save.
            file_name: Name of the sequences to be stored.
        """
        if not os.path.exists('generated_sequences'):
            os.makedirs('generated_sequences')
        video = cv2.VideoWriter(
            os.path.join('generated_sequences', '{}.avi'.format(file_name)),
            cv2.VideoWriter_fourcc(*'MJPG'), 10, (x.size(3), x.size(2))
        )
        x = (x.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 3, 0)
        for i in range(x.shape[0]):
            video.write(cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR))
        video.release()

    @staticmethod
    def get_indexes(size):
        """Returns the indexes of both the target frame and the reference
        frames given a sequence of size ``size``.

        Args:
            size: Number indicating the size of the input sequence.

        Returns:
            Tuple of two positions containing:
                - The index of the target frame.
                - A list of indexes of the reference frames.
        """
        t, r_list = size // 2, list(range(size))
        r_list.pop(t)
        return t, r_list

    @staticmethod
    def get_indexes_ff(t, max_t, s, D):
        """Returns the indexes of the reference frames for the frame with index
        ``t`` using the Frame-by-Frame Algorithm.

        Args:
            t: Index of the target frame.
            max_t: Length of the sequence.
            s: Number indicating the minimum distance between frames.
            D: Number indicating the maximum distance between frames.

        Returns:
            List containing the indexes of the reference frames.
        """
        ref_candidates = list(range(max_t))
        ref_candidates.pop(t)
        ref_candidates_dist = list(map(lambda x: abs(x - t), ref_candidates))
        ref_candidates_sorted = [
            r[1] for r in sorted(zip(ref_candidates_dist, ref_candidates))
        ]
        return list(filter(
            lambda x: abs(x - t) <= D and abs(x - t) % s == 0,
            ref_candidates_sorted
        ))

    @staticmethod
    def get_indexes_ip(t, t_list, s, D):
        """Returns the indexes of the reference frames for the frame with index
        ``t`` using the Inpaint and Propagate Algorithm.

        Args:
            t: Index of the target frame.
            t_list: List containing the order of the inpainting frames.
            s: Number indicating the minimum distance between frames.
            D: Number indicating the maximum distance between frames.

        Returns:
            List containing the indexes of the reference frames.
        """
        t_list_inpainted = list(reversed(t_list[:t_list.index(t)]))
        t_list_ff = CHN.get_indexes_ff(t, len(t_list), s, D)
        t_list_ff = [
            t_item for t_item in t_list_ff if t_item not in t_list_inpainted
        ]
        return t_list_inpainted + t_list_ff


class RRDBNet(nn.Module):
    """Implementation of the RRDB Network.

    Attributes:
        conv_first: Instance of a ``torch.nn.Sequential`` layer.
        rrdb_trunk: Instance of a ``torch.nn.Sequential`` layer.
        trunk_conv: Instance of a ``torch.nn.Conv2d`` layer.
        upconv1: Instance of a ``torch.nn.Conv2d`` layer.
        upconv2: Instance of a ``torch.nn.Conv2d`` layer.
        hr_conv: Instance of a ``torch.nn.Conv2d`` layer.
        conv_last: Instance of a ``torch.nn.Conv2d`` layer.
        lrelu: Instance of a ``torch.nn.LeakyReLU`` layer.
    """

    def __init__(self, in_nc, out_nc, nb=10, nf=64, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_nc, nf, (3, 3), padding=1),
            nn.Conv2d(nf, nf, (3, 3), padding=1),
            nn.Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(nf, nf, (3, 3), padding=1),
            nn.Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=1)
        )
        rrdb_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.rrdb_trunk = nn.Sequential(*[rrdb_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.hr_conv = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, (3, 3), (1, 1), 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """"Forward pass through the RRDB Network.

        Args:
            x: Tensor of size ``(B,C,H,W)`` containing the input volume.

        Returns:
            Tensor of size ``(B,C,H,W)`` containing the output volume.
        """
        y = self.conv_first(x)
        y = y + self.trunk_conv(self.rrdb_trunk(y))
        y = self.lrelu(
            self.upconv1(F.interpolate(y, scale_factor=2, mode='nearest'))
        )
        y = self.lrelu(
            self.upconv2(F.interpolate(y, scale_factor=2, mode='nearest'))
        )
        return self.conv_last(self.lrelu(self.hr_conv(y)))


class RRDB(nn.Module):
    """Implementation of the RRDB layer.

    Attributes:
        rdb1: Instance of a ``ResidualDenseBlock5C`` layer.
        rdb2: Instance of a ``ResidualDenseBlock5C`` layer.
        rdb3: Instance of a ``ResidualDenseBlock5C`` layer.
    """

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock5C(nf, gc)
        self.rdb2 = ResidualDenseBlock5C(nf, gc)
        self.rdb3 = ResidualDenseBlock5C(nf, gc)

    def forward(self, x):
        """"Forward pass through the RRDB layer.

        Args:
            x: Tensor of size ``(B,C,H,W)`` containing the input volume.

        Returns:
            Tensor of size ``(B,C,H,W)`` containing the input volume.
        """
        y = self.rdb1(x)
        y = self.rdb2(y)
        y = self.rdb3(y)
        return 0.2 * y + x


class ResidualDenseBlock5C(nn.Module):
    """Implementation of the Residual Dense Block 5C layer.

    Attributes:
        conv1: Instance of a ``torch.nn.Conv2d`` layer.
        conv2: Instance of a ``torch.nn.Conv2d`` layer.
        conv3: Instance of a ``torch.nn.Conv2d`` layer.
        conv4: Instance of a ``torch.nn.Conv2d`` layer.
        conv5: Instance of a ``torch.nn.Conv2d`` layer.
        lrelu: Instance of a ``torch.nn.LeakyReLU`` layer.
    """

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, (3, 3), (1, 1), 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, (3, 3), (1, 1), 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, (3, 3), (1, 1), 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, (3, 3), (1, 1), 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, (3, 3), (1, 1), 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """"Forward pass through the Residual Dense Block 5C layer.

        Args:
            x: Tensor of size ``(B,C,H,W)`` containing the input volume.

        Returns:
            Tensor of size ``(B,C,H,W)`` containing the output volume.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return 0.2 * x5 + x
