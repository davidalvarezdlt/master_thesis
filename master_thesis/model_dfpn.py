"""
Module containing the ``pytorch_lightning.LightningModule`` implementation of
the of the Dense Flow Prediction Network (DFPN).
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import master_thesis as mt


class DFPN(pl.LightningModule):
    """Implementation of the Dense Flow Prediction Network (DFPN).

    Attributes:
        corr: Instance of a ``CorrelationVGG`` layer.
        corr_mixer: Instance of a ``AlignmentCorrelationMixer`` layer.
        flow_64: Instance of a ``FlowEstimator`` layer.
        flow_256: Instance of a ``FlowEstimator`` layer.
        model_vgg: Instance of a ``VGGFeatures`` network.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """
    LOSSES_NAMES = [
        'corr_loss', 'flow_16', 'flow_64', 'flow_256',
        'alignment_recons_64', 'alignment_recons_256'
    ]

    def __init__(self, model_vgg, **kwargs):
        super(DFPN, self).__init__()
        self.corr = CorrelationVGG(model_vgg)
        self.corr_mixer = AlignmentCorrelationMixer()
        self.flow_64 = FlowEstimator()
        self.flow_256 = FlowEstimator()
        self.register_buffer(
            'mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            'std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        )

        self.model_vgg = model_vgg
        self.kwargs = kwargs

    def forward(self, x_target, m_target, x_refs, m_refs):
        """Forward pass through the Dense Flow Prediction Network (DFPN).

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the masked frame
                of which to align other frames.
            m_target: Tensor of size ``(B,1,H,W)`` containing the mask of
                ``x_target``.
            x_refs: Tensor of size ``(B,C,F,H,W)`` containing the frames to
                align with respect to ``x_target``.
            m_refs: Tensor of size ``(B,1,F,H,W)`` containing the masks of
                ``x_refs``.

        Returns:
            Tuple of four positions containing:
                - Tensor of size ``(B,F,16,16,16,16)`` containing the filled
                    correlation volume between target and reference frames.
                - Tensor of size ``(B,F,16,16,2)`` containing the predicted
                    flow in the lowest resolution.
                - Tensor of size ``(B,F,64,64,2)`` containing the predicted
                    flow in the middle resolution.
                - Tensor of size ``(B,F,H,W,2)`` containing the predicted flow
                    in the highest resolution.
        """
        b, c, ref_n, h, w = x_refs.size()
        x_target = (x_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        x_refs = (x_refs - self.mean) / self.std

        x_target_sq, m_target_sq, x_ref_sq, m_ref_sq = \
            mt.TransformsUtils.resize_set_bis(
                x_target, m_target, x_refs, m_refs, (256, 256)
            )
        x_target_64, m_target_64, x_ref_64, m_ref_64 = \
            mt.TransformsUtils.resize_set_bis(
                x_target, m_target, x_refs, m_refs, (64, 64)
            )

        corr = self.corr(x_target_sq, m_target_sq, x_ref_sq, m_ref_sq)
        flow_16 = self.corr_mixer(corr)

        flow_64_pre = mt.FlowsUtils.resize_flow(
            flow_16, (64, 64), mode='bilinear'
        )
        flow_64 = self.flow_64(
            x_target_64, m_target_64, x_ref_64, m_ref_64, flow_64_pre
        )

        flow_256_pre = mt.FlowsUtils.resize_flow(
            flow_64, (256, 256), mode='bilinear'
        )
        flow_256 = self.flow_256(
            x_target_sq, m_target_sq, x_ref_sq, m_ref_sq, flow_256_pre
        )

        return corr, flow_16, flow_64, \
            mt.FlowsUtils.resize_flow(flow_256, (h, w), mode='bilinear')

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
        with torch.no_grad():
            *_, flow_256 = self(x_target, m_target, x_refs, m_refs)

        x_ref_aligned, v_ref_aligned = mt.FlowsUtils.align_set(
            x_refs, (1 - m_refs), flow_256
        )
        v_map = (v_ref_aligned - (1 - m_target).unsqueeze(2)).clamp(0, 1)

        return x_ref_aligned, v_ref_aligned, v_map

    def training_step(self, batch, batch_idx):
        """Performs a single pass through the training dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        (x, m), y, info = batch
        flows_use, flow_gt = info[2], info[4]
        t, r_list = DFPN.get_indexes(x.size(2))

        corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use = \
            DFPN._train_val_wrapper(
                self, x, m, y, flow_gt, flows_use, t, r_list
            )

        loss, loss_items = self.compute_loss(
            corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list
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
        flows_use, flow_gt = info[2], info[4]
        t, r_list = DFPN.get_indexes(x.size(2))

        corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use = \
            DFPN._train_val_wrapper(
                self, x, m, y, flow_gt, flows_use, t, r_list
            )

        loss, loss_items = self.compute_loss(
            corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list
        )
        self._log_losses(loss, loss_items, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single pass through the test dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        (x, m), y, info = batch
        flows_use, flow_gt = info[2], info[5]
        t, r_list = DFPN.get_indexes(x.size(2))

        corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use = \
            DFPN._train_val_wrapper(
                self, x, m, y, flow_gt, flows_use, t, r_list
            )

        loss, loss_items = self.compute_loss(
            corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list
        )
        self._log_frames(x, m, y, t, r_list)
        return loss

    def compute_loss(self, corr, xs, vs, ys, xs_aligned, flows, flows_gt,
                     flows_use, t, r_list):
        """Computes the loss of the Dense Flow Prediction Network (DFPN).

        Args:
            corr: Tensor of size ``(B,F,16,16,16,16)`` containing the filled
                correlation volume between target and reference frames.
            xs: Tuple of three positions containing the masked background
                frames in different resolutions.
            vs: Tuple of three positions containing the visibility maps in
                different resolutions.
            ys: Tuple of three positions containing the original background
                frames in different resolutions.
            xs_aligned: Tuple of three positions containing the reference
                masked background frames aligned with respect to the target.
            flows: Tuple of three positions containing the predicted flows in
                different resolutions.
            flows_gt: Tuple of three positions containing the ground-truth
                flows in different resolutions.
            flows_use: Tensor of size ``(--batch_size)`` indicating if the data
                item has been obtained using fake transormations.
            t: Index of the target frame
            r_list: List of indexes of the reference frames

        Returns:
            Tuple of two positions containing:
                - The sum of the different losses.
                - List containing the different loss items in the same order as
                    ``DFPN.LOSSES_NAMES``.
        """
        b, c, f, h, w = ys[2].size()

        with torch.no_grad():
            if h == 256 and w == 256:
                y_vgg_input = ys[2].transpose(1, 2).reshape(b * f, c, h, w)
            else:
                y_vgg_input = F.interpolate(
                    ys[2].transpose(1, 2).reshape(b * f, c, h, w),
                    (256, 256),
                    mode='bilinear',
                )
            y_vgg_feats = self.model_vgg(y_vgg_input)
        y_vgg_feats = y_vgg_feats[3].reshape(b, f, -1, 16, 16).transpose(1, 2)

        corr_y = CorrelationVGG.correlation_masked_4d(
            y_vgg_feats[:, :, t], None, y_vgg_feats[:, :, r_list], None
        )
        corr_loss = F.l1_loss(corr, corr_y)

        flow_loss_16 = mt.LossesUtils.masked_l1(
            flows[0], flows_gt[0], torch.ones_like(flows[0]), flows_use
        )
        flow_loss_64 = mt.LossesUtils.masked_l1(
            flows[1], flows_gt[1], torch.ones_like(flows[1]), flows_use
        )
        flow_loss_256 = mt.LossesUtils.masked_l1(
            flows[2], flows_gt[2], torch.ones_like(flows[2]), flows_use
        )

        mask_out_64 = ((flows[1] < -1).float() + (flows[1] > 1).float()) \
            .sum(4).clamp(0, 1).unsqueeze(1)
        mask_out_256 = ((flows[2] < -1).float() + (flows[2] > 1).float()) \
            .sum(4).clamp(0, 1).unsqueeze(1)

        alignment_recons_64 = mt.LossesUtils.masked_l1(
            xs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1),
            xs_aligned[1],
            vs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
            * (1 - mask_out_64),
            reduction='sum',
        )
        alignment_recons_256 = mt.LossesUtils.masked_l1(
            xs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1),
            xs_aligned[2],
            vs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
            * (1 - mask_out_256),
            reduction='sum',
        )

        total_loss = corr_loss + flow_loss_16 + flow_loss_64 + flow_loss_256
        total_loss += alignment_recons_64 + alignment_recons_256
        return total_loss, [corr_loss, flow_loss_16, flow_loss_64,
                            flow_loss_256, alignment_recons_64,
                            alignment_recons_256]

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

    def _train_val_wrapper(self, x, m, y, flow_gt, flows_use, t, r_list):
        """Auxiliary method used to propagate through training and validation
        iterations.

        Args:
            x: Tensor of size ``(B,C,F,H,W)`` containing the masked background
                frames in the highest resolution.
            m: Tensor of size ``(B,1,F,H,W)`` containing the mask frames in the
                highest resolution.
            y: Tensor of size ``(B,C,F,H,W)`` containing the original
                background frames in the highest resolution.
            flow_gt:
            flows_use: Tensor of size ``(--batch_size)`` indicating if the data
                item has been obtained using fake transormations.
            t: Index of the target frame
            r_list: List of indexes of the reference frames

        Returns:
            Tuple of eight positions containing:
                - Tensor of size ``(B,F,16,16,16,16)`` containing the filled
                    correlation volume between target and reference frames.
                - Tuple of three positions containing the masked background
                    frames in different resolutions.
                - Tuple of three positions containing the visibility maps in
                    different resolutions.
                - Tuple of three positions containing the original background
                    frames in different resolutions.
                - Tuple of three positions containing the reference masked
                    background frames aligned with respect to the target.
                - Tuple of three positions containing the predicted flows in
                    different resolutions.
                - Tuple of three positions containing the ground-truth flows in
                    different resolutions.
                - Tensor of size ``(--batch_size)`` indicating if the data item
                    has been obtained using fake transormations.
        """
        corr, flow_16, flow_64, flow_256 = self(
            x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

        x_16, v_16, y_16 = mt.TransformsUtils.resize_set(x, 1 - m, y, 16)
        x_64, v_64, y_64 = mt.TransformsUtils.resize_set(x, 1 - m, y, 64)
        x_256, v_256, y_256 = x, 1 - m, y

        flow_16_gt = mt.FlowsUtils.resize_flow(flow_gt[:, r_list], (16, 16))
        flow_64_gt = mt.FlowsUtils.resize_flow(flow_gt[:, r_list], (64, 64))
        flow_256_gt = flow_gt[:, r_list]

        x_64_aligned_gt, v_64_aligned_gt = mt.FlowsUtils.align_set(
            x_64[:, :, r_list], v_64[:, :, r_list], flow_64_gt
        )
        x_256_aligned_gt, v_256_aligned_gt = mt.FlowsUtils.align_set(
            x_256[:, :, r_list], v_256[:, :, r_list], flow_256_gt
        )

        v_map_64_gt = torch.zeros_like(v_64_aligned_gt)
        v_map_64_gt[flows_use] = (
                v_64_aligned_gt[flows_use] - v_64[flows_use, :, t]
                .unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)

        v_map_256_gt = torch.zeros_like(v_256_aligned_gt)
        v_map_256_gt[flows_use] = (
                v_256_aligned_gt[flows_use] - v_256[flows_use, :, t]
                .unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)

        x_16_aligned, v_16_aligned = mt.FlowsUtils.align_set(
            x_16[:, :, r_list], v_16[:, :, r_list], flow_16
        )
        x_64_aligned, v_64_aligned = mt.FlowsUtils.align_set(
            x_64[:, :, r_list], v_64[:, :, r_list], flow_64
        )
        x_256_aligned, v_256_aligned = mt.FlowsUtils.align_set(
            x_256[:, :, r_list], v_256[:, :, r_list], flow_256
        )

        xs = (x_16, x_64, x_256)
        vs = (v_16, v_64, v_256)
        ys = (y_16, y_64, y_256)
        xs_aligned = (x_16_aligned, x_64_aligned, x_256_aligned)
        flows = (flow_16, flow_64, flow_256)
        flows_gt = (flow_16_gt, flow_64_gt, flow_256_gt)

        return corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use

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
        for i, loss_item_id in enumerate(DFPN.LOSSES_NAMES):
            loss_name = 'loss_{}_{}'.format(loss_item_id, split)
            self.log(loss_name, loss_items[i])

    def _log_frames(self, x, m, y, t, r_list):
        """Logs aligned frames in TensorBoard.

        Args:
            x: Tensor of size ``(B,C,F,H,W)`` containing masked frames.
            m: Tensor of size ``(B,1,F,H,W)`` containing the masks of ``x``.
            y: Tensor of size ``(B,C,F,H,W)`` containing the frames without a
                mask.
            t: Index of the target frame.
            r_list: List of indexes of the reference frames.
        """
        b, c, frames_n, h, w = x.size()

        x_ref_aligned, v_ref_aligned, v_map = self.align(
            x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        y_hat_trivial = x[:, :, t].unsqueeze(2).repeat(
            1, 1, frames_n - 1, 1, 1
        ) * (1 - v_map) + x_ref_aligned * v_map

        x = x.cpu().numpy()
        m = m.cpu().numpy()
        y = y.cpu().numpy()
        x_ref_aligned = x_ref_aligned.cpu().numpy()
        v_ref_aligned = v_ref_aligned.cpu().numpy()
        y_hat_trivial = y_hat_trivial.cpu().numpy()

        for b in range(x_ref_aligned.shape[0]):
            x_aligned_sample = np.insert(
                arr=x_ref_aligned[b], obj=t, values=x[b, :, t], axis=1
            )
            v_map_sample = np.insert(
                arr=v_ref_aligned[b].repeat(3, axis=0), obj=t,
                values=m[b, :, t].repeat(3, axis=0), axis=1
            )
            y_hat_trivial_sample = np.insert(
                arr=y_hat_trivial[b], obj=t, values=y[b, :, t], axis=1
            )
            sample = np.concatenate((
                x[b], x_aligned_sample, v_map_sample, y_hat_trivial_sample
            ), axis=2)
            self.logger.experiment.add_images(
                'frames/{}'.format(b + 1),
                sample.transpose((1, 0, 2, 3)),
                global_step=self.current_epoch
            )

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


class CorrelationVGG(nn.Module):
    """Implementation of the CorrelationVGG layer.

    Attributes:
        conv: Instance of a ``SeparableConv4d`` layer.
        model_vgg: Instance of a ``VGGFeatures`` network.
        use_softmax: Whether or not to apply a softmax at the output.
    """

    def __init__(self, model_vgg, use_softmax=False):
        super(CorrelationVGG, self).__init__()
        self.conv = SeparableConv4d()
        self.model_vgg = model_vgg
        self.use_softmax = use_softmax

    def forward(self, x_target, m_target, x_refs, m_refs):
        """Forward pass through the 4D Separable Convolution layer.

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the masked frame
                of which to align other frames.
            m_target: Tensor of size ``(B,1,H,W)`` containing the mask of
                ``x_target``.
            x_refs: Tensor of size ``(B,C,F,H,W)`` containing the frames to
                align with respect to ``x_target``.
            m_refs: Tensor of size ``(B,1,F,H,W)`` containing the masks of
                ``x_refs``.

        Returns:
            Tensor of size ``(B,F,16,16,16,16)`` containing the correlation
            volume between target and reference frames.
        """
        b, c, ref_n, h, w = x_refs.size()

        with torch.no_grad():
            x_target_feats = self.model_vgg(x_target, normalize_input=False)
            x_ref_feats = self.model_vgg(
                x_refs.transpose(1, 2).reshape(b * ref_n, c, h, w),
                normalize_input=False
            )
        x_target_feats = x_target_feats[3]
        x_ref_feats = x_ref_feats[3].reshape(b, ref_n, -1, 16, 16) \
            .transpose(1, 2)

        b, c, ref_n, h, w = x_ref_feats.size()
        v_target = F.interpolate(1 - m_target, size=(h, w), mode='nearest')
        v_ref = F.interpolate(
            1 - m_refs.transpose(1, 2).reshape(
                b * ref_n, 1, m_refs.size(3), m_refs.size(4)
            ), size=(h, w), mode='nearest'
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2)

        corr = CorrelationVGG.correlation_masked_4d(
            x_target_feats, v_target, x_ref_feats, v_ref
        )
        corr = self.conv(corr)
        return CorrelationVGG.softmax_3d(corr) if self.use_softmax else corr

    @staticmethod
    def correlation_masked_4d(x_target_feats, v_target, x_ref_feats, v_ref):
        """Computes the normalized correlation between the feature maps of the
        target and reference frames.

        Args:
            x_target_feats: Tensor of size (B,C,H,W) containing the feature map
                of the target frame.
            v_target: Tensor of size (B,1,H,W) containing the visibility map of
                the target frame.
            x_ref_feats: Tensor of size (B,C,F,H,W) containing the feature maps
                of the reference frames.
            v_ref: Tensor of size (B,1,H,W) containing the visibility maps of
                the reference frames.

        Returns:
            4D correlation volume of size (B,F,H,W,H,W).
        """
        b, c, ref_n, h, w = x_ref_feats.size()

        x_target_feats = x_target_feats * v_target if v_target is not None \
            else x_target_feats
        x_ref_feats = x_ref_feats * v_ref if v_ref is not None else x_ref_feats

        corr_1 = x_target_feats.reshape(b, c, -1).transpose(-1, -2) \
            .unsqueeze(1)
        corr_1_norm = torch.norm(corr_1, dim=3).unsqueeze(3) + 1e-9
        corr_2 = x_ref_feats.reshape(b, c, ref_n, -1).permute(0, 2, 1, 3)
        corr_2_norm = torch.norm(corr_2, dim=2).unsqueeze(2) + 1e-9

        return torch.matmul(corr_1 / corr_1_norm, corr_2 / corr_2_norm) \
            .reshape(b, ref_n, h, w, h, w)

    @staticmethod
    def softmax_3d(x):
        """Computes a 3D softmax function over the 4D correlation volume.

        Args:
            x: Tensor of size ``(B,F,16,16,16,16)`` containing the correlation
            volume between target and reference frames.

        Returns:
            Tensor of size ``(B,F,16,16,16,16)`` containing the correlation
            volume between target and reference frames after applying the
            3D softmax function.
        """
        b, t, h, w, _, _ = x.size()
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(b, h, w, -1)
        x = F.softmax(x, dim=3)
        return x.reshape(b, h, w, h, w, t).permute(0, 5, 1, 2, 3, 4)


class SeparableConv4d(nn.Module):
    """Implementation of the 4D Separable Convolution layer.

    Attributes:
        conv_1: Instance of a ``torch.nn.Sequential`` layer.
        conv_2: Instance of a ``torch.nn.Sequential`` layer.
    """

    def __init__(self):
        super(SeparableConv4d, self).__init__()
        self.conv_1 = nn.Sequential(
            torch.nn.Conv2d(1, 128, (3, 3), padding=1), nn.ReLU(),
            torch.nn.Conv2d(128, 256, (3, 3), padding=1), nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
        )
        self.conv_2 = nn.Sequential(
            torch.nn.Conv2d(256, 256, (3, 3), padding=1), nn.ReLU(),
            torch.nn.Conv2d(256, 128, (3, 3), padding=1), nn.ReLU(),
            torch.nn.Conv2d(128, 1, (3, 3), padding=1),
        )

    def forward(self, corr):
        """Forward pass through the 4D Separable Convolution layer.

        Args:
            corr: Tensor of size ``(B,F,16,16,16,16)`` containing the
                correlation volume between target and reference frames.

        Returns:
            Tensor of size ``(B,F,16,16,16,16)`` containing the filled
            correlation volume between target and reference frames.
        """
        corr = corr.unsqueeze(4)
        b, t, h, w, c, *_ = corr.size()
        x2_bis = self.conv_1(corr.reshape(-1, c, h, w))
        x2_bis = x2_bis.reshape(b, t, h * w, x2_bis.size(1), h * w).permute(
            0, 1, 4, 3, 2
        )
        x3_bis = self.conv_2(x2_bis.reshape(-1, x2_bis.size(3), h, w))
        x3_bis = x3_bis.reshape(b, t, h, w, x3_bis.size(1), h, w).squeeze(4)
        return x3_bis.permute(0, 1, 4, 5, 2, 3)


class AlignmentCorrelationMixer(nn.Module):
    """Implementation of the Alignment Correlation Mixer layer.

    Attributes:
        mixer: Instance of a ``torch.nn.Sequential`` layer.
    """

    def __init__(self, corr_size=16):
        super(AlignmentCorrelationMixer, self).__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(corr_size ** 2, corr_size ** 2, (5, 5), padding=2),
            nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size ** 2, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size, (5, 5), padding=2), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size // 2, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 2, (5, 5), padding=2),
            nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 2, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 4, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 4, (5, 5), padding=2),
            nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 4, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 8, (3, 3), padding=1),
            nn.Conv2d(corr_size // 8, corr_size // 8, (5, 5), padding=2),
            nn.Conv2d(corr_size // 8, corr_size // 8, (3, 3), padding=1)
        )

    def forward(self, corr):
        """Forward pass through the Alignment Correlation Mixer layer.

        Args:
            corr: Tensor of size ``(B,F,16,16,16,16)`` containing the filled
                correlation volume between target and reference frames.

        Returns:
            Tensor of size ``(B,F,16,16,2)`` containing the predicted flow.
        """
        b, f, h, w, *_ = corr.size()
        corr = corr.reshape(b * f, -1, 16, 16)
        return self.mixer(corr).reshape(b, f, 2, h, w).permute(0, 1, 3, 4, 2)


class FlowEstimator(nn.Module):
    """Implementation of the Flow Estimator layer.

    Attributes:
        nn: Instance of a ``torch.nn.Sequential`` layer.
    """

    def __init__(self, in_c=10):
        super(FlowEstimator, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_c, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (2, 2), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (5, 5), (2, 2), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), (1, 1), 2), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 2, (3, 3), (1, 1), 1)
        )

    def forward(self, x_target, m_target, x_refs, m_refs, flow_pre):
        """Forward pass through the Flow Estimator layer.

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the masked frame
                of which to align other frames.
            m_target: Tensor of size ``(B,1,H,W)`` containing the mask of
                ``x_target``.
            x_refs: Tensor of size ``(B,C,F,H,W)`` containing the frames to
                align with respect to ``x_target``.
            m_refs: Tensor of size ``(B,1,F,H,W)`` containing the masks of
                ``x_refs``.
            flow_pre: Tensor of size ``(B,F,H,W,2)`` containing the flow in
                lower resolution.

        Returns:
            Tensor of size ``(B,F,H',W',2)`` containing the flow in the
                upscaled resolution.
        """
        b, c, ref_n, h, w = x_refs.size()
        nn_input = torch.cat([
            x_refs.transpose(1, 2).reshape(b * ref_n, c, h, w),
            x_target.unsqueeze(1).repeat(1, ref_n, 1, 1, 1)
            .reshape(b * ref_n, c, h, w),
            m_refs.transpose(1, 2).reshape(b * ref_n, 1, h, w),
            m_target.unsqueeze(1).repeat(1, ref_n, 1, 1, 1)
            .reshape(b * ref_n, 1, h, w),
            flow_pre.reshape(b * ref_n, h, w, 2).permute(0, 3, 1, 2),
        ], dim=1)
        return self.nn(nn_input).reshape(b, ref_n, 2, h, w) \
            .permute(0, 1, 3, 4, 2)
