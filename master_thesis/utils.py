import math
import random

import cv2
import numpy as np
import skimage.metrics
import skimage.transform
import torch
import torch.nn.functional as F


class FlowsUtils:
    """Utilities class containing flow-related methods."""

    @staticmethod
    def flow_abs_to_relative(flow):
        """Given a normalized flow between [-1, 1], returns the relative
        flow between [-2, 2].

        Args:
            flow: Tensor of size ``(B,F,H,W,2)`` containing absolute flows.

        Returns:
            Tensor of size ``(B,F,H,W,2)`` containing relative flows.
        """
        b, f, h, w, _ = flow.size()
        flow_pos_identity = F.affine_grid(
            torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).unsqueeze(0),
            [1, 1, h, w],
            align_corners=True,
        ).view(1, 1, h, w, 2)
        return flow - flow_pos_identity.repeat(b, f, 1, 1, 1)

    @staticmethod
    def flow_relative_to_abs(flow_rel):
        """Given a relative flow between [-2, 2], returns the absolute flow
        between [-1, 1]

        Args:
            flow_rel: Tensor of size ``(B,F,H,W,2)`` containing relative flows.

        Returns:
            Tensor of size ``(B,F,H,W,2)`` containing absolute flows.
        """
        b, f, h, w, _ = flow_rel.size()
        flow_pos_identity = F.affine_grid(
            torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).unsqueeze(0),
            [1, 1, h, w],
            align_corners=True,
        ).view(1, 1, h, w, 2)
        return flow_rel + flow_pos_identity.repeat(b, f, 1, 1, 1)

    @staticmethod
    def crop_flow(flow, crop_size, crop_position):
        """Cuts an absolute flow between at the position ``crop_position``.

        Args:
            flow: Tensor of size ``(B,F,H,W,2)`` containing absolute flows.
            crop_size: Tuple containing the size of the cropped flow.
            crop_position: Tuple containing the position of the cropped flow.

        Returns:
            Tensor of size ``(B,F,H',W',2)`` containing the cropped flow.
        """
        b, f, h, w, _ = flow.size()
        flow_rel = FlowsUtils.flow_abs_to_relative(flow)
        flow_h_from, flow_h_to = crop_position[0], \
            crop_position[0] + crop_size[0]
        flow_w_from, flow_w_to = crop_position[1], \
            crop_position[1] + crop_size[1]
        flow_rel_cut = \
            flow_rel[:, :, flow_h_from: flow_h_to, flow_w_from: flow_w_to]
        flow_rel_cut[:, :, :, :, 0] *= w / crop_size[1]
        flow_rel_cut[:, :, :, :, 1] *= h / crop_size[0]
        return FlowsUtils.flow_relative_to_abs(flow_rel_cut)

    @staticmethod
    def align_set(x, v, flow):
        """Aligns the images ``x`` and ``v`` using the flow given in ``flow``.

        Args:
            x: Tensor of size ``(B,C,F,H,W)`` containing masked background
                frames.
            v: Tensor of size ``(B,1,F,H,W)`` containing visibility maps.
            flow: Tensor of size ``(B,F,H,W,2)`` containing the flows.

        Returns:
            Tuple of two positions containing:
                - The masked background frames ``x`` aligned using ``flow``.
                - The visibility maps ``v`` aligned using ``flow``.
        """
        b, c, f, h, w = x.size()
        x_aligned = F.grid_sample(
            x.transpose(1, 2).reshape(-1, c, h, w),
            flow.reshape(-1, h, w, 2),
            align_corners=True,
        ).reshape(b, -1, 3, h, w).transpose(1, 2)
        v_aligned = F.grid_sample(
            v.transpose(1, 2).reshape(-1, 1, h, w),
            flow.reshape(-1, h, w, 2),
            align_corners=True,
            mode='nearest',
        ).reshape(b, -1, 1, h, w).transpose(1, 2)
        return x_aligned, v_aligned

    @staticmethod
    def resize_flow(flow, size, mode='nearest'):
        """Resizes a flow to a new resolution given by ``size``.

        Args:
            flow: Tensor of size ``(B,F,H,W,2)`` containing the flow in the
                original resolution.
            size: Tuple containing the new height and width of the flow.
            mode: Mode used to resize the flow. Same format as in
                ``torch.nn.functional.interpolate()``.

        Returns:
            Tensor of size ``(B,F,h_new,w_new,2)`` containing the flow in the
                new resolution.
        """
        b, f, h, w, _ = flow.size()
        flow_resized = F.interpolate(
            flow.reshape(b * f, h, w, 2).permute(0, 3, 1, 2), size, mode=mode
        )
        return flow_resized.reshape(b, f, 2, size[0], size[1]) \
            .permute(0, 1, 3, 4, 2)


class LossesUtils:
    """Utilities class containing loss-related methods."""
    _GRAD_H = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).repeat((3, 1, 1, 1))
    _GRAD_V = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).repeat((3, 1, 1, 1))

    @staticmethod
    def masked_l1(y_hat, y, mask, batch_mask=None, reduction='mean', weight=1):
        """Computes the L1 loss of the image ``y_hat`` with respect to the
        ground truth ``y`` on those positions which are not masked by ``mask``.

        Args:
            y_hat: Tensor of size ``(B,C,H,W,*)`` containing the estimated
                image.
            y: Tensor of size ``(B,C,H,W,*)`` containing the ground-truth
                image.
            mask: Tensor of size ``(B,1,H,W,*)`` containing the mask.
            batch_mask: Tensor of the same size as ``--batch_size`` indicating
                if that element should be taken into account to compute the
                loss.
            reduction: Reduction mode applied to the loss.
            weight: Scaling factor applied to the loss.

        Returns:
            Masked L1 loss between ``y_hat`` and ``y``.
        """
        if batch_mask is not None and not any(batch_mask):
            return torch.zeros(1).to(y_hat.device)
        elif batch_mask is not None:
            y_hat, y, mask = (
                y_hat[batch_mask],
                y[batch_mask],
                mask[batch_mask],
            )
        masked_l1_loss = F.l1_loss(y_hat * mask, y * mask, reduction=reduction)
        return weight * masked_l1_loss / (
            torch.sum(mask) + 1e-9 if reduction == 'sum' else 1
        )

    @staticmethod
    def perceptual(y_hat, y, model_vgg, weight=1):
        """Computes the perceptual loss of the image ``y_hat`` with respect
        to the ground truth ``y``.

        Args:
            y_hat: Tensor of size ``(B,C,H,W)`` containing the estimated image.
            y: Tensor of size ``(B,C,H,W)`` containing the ground-truth image.
            weight: Scaling factor applied to the loss.

        Returns:
            Perceptual loss between ``y_hat`` and ``y``.
        """
        input_vgg = model_vgg(y_hat.contiguous())
        target_vgg = model_vgg(y.contiguous())

        loss_perceptual = 0
        for p in range(len(input_vgg)):
            loss_perceptual += F.l1_loss(input_vgg[p], target_vgg[p])

        return loss_perceptual * weight / len(input_vgg), input_vgg, target_vgg

    @staticmethod
    def grad(y_hat, y, reduction, weight=1):
        """Computes the gradient loss of the image ``y_hat`` with respect
        to the ground truth ``y``.

        Args:
            y_hat: Tensor of size ``(B,C,H,W)`` containing the estimated image.
            y: Tensor of size ``(B,C,H,W)`` containing the ground-truth image.
            reduction: Reduction mode applied to the loss.
            weight: Scaling factor applied to the loss.

        Returns:
            Gradient loss between ``y_hat`` and ``y``.
        """
        if y_hat.device != LossesUtils._GRAD_H.device:
            LossesUtils._GRAD_H = LossesUtils._GRAD_H.to(y_hat.device)
            LossesUtils._GRAD_V = LossesUtils._GRAD_V.to(y_hat.device)

        input_grads = torch.cat((
            F.conv2d(y_hat, padding=1, weight=LossesUtils._GRAD_H, groups=3),
            F.conv2d(y_hat, padding=1, weight=LossesUtils._GRAD_V, groups=3)
        ), dim=1)
        target_grads = torch.cat((
            F.conv2d(y, padding=1, weight=LossesUtils._GRAD_H, groups=3),
            F.conv2d(y, padding=1, weight=LossesUtils._GRAD_V, groups=3)
        ), dim=1)
        mask = torch.ones_like(input_grads).to(y_hat.device)

        return LossesUtils.masked_l1(
            input_grads, target_grads, mask, batch_mask=None,
            reduction=reduction, weight=weight,
        )


class MovementsUtils:
    """Utilities class containing movement-related methods.

    Args:
        max_displacement: Number indicating the maximum displacement applied to
            an image.
        max_scaling: Number indicating the maximum scaling applied to an image.
        max_rotation: Number indicating the maximum rotation applied to an
            image.
    """

    def __init__(self, max_displacement, max_scaling, max_rotation):
        self.max_displacement = max_displacement
        self.max_scaling = max_scaling
        self.max_rotation = max_rotation

    def random_affine(self):
        """Returns a random affine transformation matrix.

        Returns:
            Tensor of size ``(3,3)`` containing a randomly-generated affine
            transformation matrix.
        """
        tx, ty = np.random.randint(
            low=-self.max_displacement, high=self.max_displacement, size=2
        ) if self.max_displacement > 0 else (0, 0)
        sx, sy = np.random.uniform(
            low=1 - self.max_scaling, high=1 + self.max_scaling, size=2
        )
        rot = np.random.uniform(low=-self.max_rotation, high=self.max_rotation)
        affine_matrix = skimage.transform.AffineTransform(
            translation=(tx, ty), scale=(sx, sy), rotation=rot
        ).params
        return torch.from_numpy(affine_matrix).float()

    def simulate_movement(self, x, n, affine_matrices=None):
        """Simulates a moving sequence of ``n` frames using ``frame`` as
        starting point.

        Args:
            x: Tensor of size ``(C,H,W)`` containing the first frame.
            n: Number of frames of the sequence.
            affine_matrices: Tensor of size ``(n,3,3)`` containing the
            transformations to apply, or ``None``.

        Returns:
            Tensor of size ``(C,F,H,W)`` containing the moving sequence.
        """
        c, h, w = x.size()

        if affine_matrices is None:
            affine_matrices = [self.random_affine() for _ in range(n - 1)]
            affine_matrices = (
                    affine_matrices[: n // 2]
                    + [MovementsUtils.identity_affine()]
                    + affine_matrices[n // 2:]
            )
        affine_matrices_inv = [
            MovementsUtils.affine_inverse(affine_mat) for affine_mat in
            affine_matrices
        ]

        affine_matrices_s, affine_matrices_inv = torch.stack(
            affine_matrices
        ), torch.stack(affine_matrices_inv)

        affine_matrices_s = MovementsUtils.stack_transformations(
            affine_matrices_s, t=n // 2
        )
        affine_matrices_inv = MovementsUtils.stack_transformations(
            affine_matrices_inv, t=n // 2
        )
        affine_matrices_theta = torch.stack(
            [MovementsUtils.affine2theta(ra, h, w) for ra in affine_matrices_s]
        )
        affine_matrices_inv_theta = torch.stack(
            [MovementsUtils.affine2theta(ra, h, w) for ra in
             affine_matrices_inv]
        )

        flow = F.affine_grid(
            affine_matrices_theta, [n, c, h, w], align_corners=True
        )
        flow_inv = F.affine_grid(
            affine_matrices_inv_theta, [n, c, h, w], align_corners=True
        )

        y = F.grid_sample(
            x.unsqueeze(0).repeat(n, 1, 1, 1), flow, align_corners=True
        )

        return y.permute(1, 0, 2, 3), flow_inv, affine_matrices

    @staticmethod
    def identity_affine():
        """Returns the identity transformation matrix.

        Returns:
            Tensor of size ``(3,3)`` containing the identity transformation
            matrix.
        """
        affine_matrix = np.linalg.inv(skimage.transform.AffineTransform(
            translation=(0, 0), scale=(1, 1), rotation=0
        ).params)
        return torch.from_numpy(affine_matrix).float()

    @staticmethod
    def affine_inverse(affine):
        """Returns the inverse of a transformation matrix.

        Args:
            affine: Tensor of size ``(3,3)`` containing a transformation
                matrix.

        Returns:
            Tensor of size ``(3,3)`` containing a the inverse transformation
            matrix.
        """
        return torch.from_numpy(np.linalg.inv(affine))

    @staticmethod
    def stack_transformations(affine_matrices, t):
        """Stacks a set of single transformations.

        Given a set of ``F`` independent affine matrices and the target frame
        at position ``t``, it computes the transformation required to move from
        position ``t`` to ``[..., t-1, t-2, t+1, t+2, ...]``.

        Args:
            affine_matrices: Tensor of size ``(F,3,3)`` containing the
                transformations to stack.
            t: Index of the target frame.

        Returns:
            Tensor of size ``(F,3,3)`` containing the stacked transformation
            matrices with respect to ``t``.
        """
        affine_matrices_stacked = torch.zeros(
            affine_matrices.size(), dtype=torch.float32
        )
        affine_matrices_stacked[t] = affine_matrices[t]
        for i in reversed(range(t)):
            affine_matrices_stacked[i] = torch.matmul(
                torch.inverse(affine_matrices[i]),
                affine_matrices_stacked[i + 1]
            )
        for i in range(t + 1, len(affine_matrices)):
            affine_matrices_stacked[i] = torch.matmul(
                affine_matrices[i], affine_matrices_stacked[i - 1]
            )
        return affine_matrices_stacked

    @staticmethod
    def affine2theta(param, h, w):
        """Converts an affine transformation matrix to the format used by
        ``torch.nn.functional.affine_grid``.

        Args:
            param: Tensor of size ``(3,3)`` containing an affine transformation
                matrix.
            h: Height of the image.
            w: Width of the image.

        Returns:
            Transformation matrix in the format used by
            ``torch.nn.functional.affine_grid``.
        """
        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * h / w
        theta[0, 2] = param[0, 2] * 2 / w + param[0, 0] + param[0, 1] - 1
        theta[1, 0] = param[1, 0] * w / h
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / h + param[1, 0] + param[1, 1] - 1
        return torch.from_numpy(theta).float()


class MeasuresUtils:
    """Utilities class containing measurements-related methods."""

    @staticmethod
    def psnr(y_hat, y):
        """Computes the PSNR between two images.

        Args:
            y_hat: Tensor of size ``(B,C,H,W)`` containing the estimated image.
            y: Tensor of size ``(B,C,H,W)`` containing the ground-truth image.
        """
        items_psnr = []
        for f in range(y.size(0)):
            items_psnr.append(
                skimage.metrics.peak_signal_noise_ratio(
                    y[f].cpu().numpy(), y_hat[f].cpu().numpy()
                )
            )
        items_psnr = [
            100 if math.isnan(i_psnr) else i_psnr
            for i_psnr in items_psnr
        ]
        return np.mean([
            item_psnr for item_psnr in items_psnr
            if not np.isinf(item_psnr) and not np.isnan(item_psnr)
        ])

    @staticmethod
    def ssim(y_hat, y):
        """Computes the SSIM between two images.

        Args:
            y_hat: Tensor of size ``(B,C,H,W)`` containing the estimated image.
            y: Tensor of size ``(B,C,H,W)`` containing the ground-truth image.
        """
        items_ssim = []
        for f in range(y.size(0)):
            items_ssim.append(
                skimage.metrics.structural_similarity(
                    y[f].permute(1, 2, 0).cpu().numpy(),
                    y_hat[f].permute(1, 2, 0).cpu().numpy(),
                    multichannel=True
                )
            )
        return np.mean(items_ssim)

    @staticmethod
    def lpips(y_hat, y, model=None):
        """Computes the LPIPS between two images.

        Args:
            y_hat: Tensor of size ``(B,C,H,W)`` containing the estimated image.
            y: Tensor of size ``(B,C,H,W)`` containing the ground-truth image.
            model: Instance of a ``lpips.LPIPS`` network.
        """
        with torch.no_grad():
            return np.mean(
                model.forward(2 * y_hat - 1, 2 * y - 1).flatten().cpu()
                .tolist()
            )


class TransformsUtils:
    """Utilities class containing transformation-related methods."""

    @staticmethod
    def resize(image, size, mode='bilinear', keep_ratio=True):
        """Resize an image using the the algorithm given in ``mode``.

        Args:
            image: Tensor of size ``(C,F,H,W)`` containing original images.
            size: Tuple of two positions containing the desired new size of the
                image.
            mode: Mode used to resize the image. Same format as in
                ``torch.nn.functional.interpolate()``.
            keep_ratio: Whether or not to keep the original image ratio. If so,
                one of the elements of ``size`` should be ``-1``.

        Returns:
            Tensor of size ``(C,F,size[0],size[1])`` containing resized images.
        """
        if keep_ratio and size[1] == -1:
            new_height = size[0]
            new_width = round(image.size(3) * size[0] / image.size(2))
            new_size = (new_height, new_width)
            return F.interpolate(
                image.transpose(0, 1), new_size, mode=mode
            ).transpose(0, 1)[:, :, : size[0], : size[1]]
        elif keep_ratio:
            new_height = (
                size[0]
                if image.size(2) < image.size(3)
                else round(image.size(2) * size[1] / image.size(3))
            )
            new_width = (
                size[1]
                if image.size(3) <= image.size(2)
                else round(image.size(3) * size[0] / image.size(2))
            )
            new_size = (new_height, new_width)
            return F.interpolate(
                image.transpose(0, 1), new_size, mode=mode
            ).transpose(0, 1)[:, :, : size[0], : size[1]]
        else:
            return F.interpolate(image.transpose(0, 1), size, mode=mode) \
                .transpose(0, 1)

    @staticmethod
    def resize_set(x, v, y, size):
        """Resizes the entire set of masked backgrounds, visibility maps and
        original background frames.

        Args:
            x: Tensor of size ``(B,C,F,H,W)`` containing the original masked
                backgrounds.
            v: Tensor of size ``(B,1,F,H,W)`` containing the visibility maps.
            y: Tensor of size ``(B,C,F,H,W)`` containing the original
                backgrounds.
            size: Tuple of two positions containing the desired new size of the
                images.

        Returns:
            Tuple of three positions containing:
                - Tensor of size ``(B,C,F,size[0],size[1])`` containing resized
                    masked backgrounds.
                - Tensor of size ``(B,1,F,size[0],size[1])`` containing resized
                    visibility maps.
                - Tensor of size ``(B,C,F,size[0],size[1])`` containing resized
                    original backgrounds.
        """
        b, c, f, h, w = x.size()
        x_new = (
            F.interpolate(
                x.transpose(1, 2).reshape(-1, c, h, w), (size, size),
                mode='bilinear'
            ).reshape(b, f, c, size, size).transpose(1, 2)
        )
        v_new = F.interpolate(
            v.transpose(1, 2).reshape(-1, 1, h, w), (size, size)
        ).reshape(b, f, 1, size, size).transpose(1, 2)
        y_new = F.interpolate(
            y.transpose(1, 2).reshape(-1, c, h, w), (size, size),
            mode='bilinear'
        ).reshape(b, f, c, size, size).transpose(1, 2)

        return x_new, v_new, y_new

    @staticmethod
    def resize_set_bis(x_target, m_target, x_ref, m_ref, size):
        """Resizes the entire set of target and reference masked frames and
        masks.

        Args:
            x_target: Tensor of size ``(B,C,H,W)`` containing the masked
                background of the target frame.
            m_target: Tensor of size ``(B,1,H,W)`` containing the mask of the
                target frame.
            x_ref: Tensor of size ``(B,C,F,H,W)`` containing the masked
                backgrounds of the reference frames.
            m_ref: Tensor of size ``(B,1,F,H,W)`` containing the masks of the
                reference frames.
            size: Tuple of two positions containing the desired new size of the
                images.

        Returns:
            Tuple of four positions containing:
                - Tensor of size ``(B,C,size[0],size[1])`` containing the
                    resized masked background of the target frame.
                - Tensor of size ``(B,1,size[0],size[1])`` containing the
                    resized masks of the target frame.
                - Tensor of size ``(B,C,F,size[0],size[1])`` containing the
                    resized masked backgrounds of the reference frames.
                - Tensor of size ``(B,1,F,size[0],size[1])`` containing the
                    resized masks of the reference frames.
        """
        b, c, ref_n, h, w = x_ref.size()
        if h == size[0] and w == size[1]:
            return x_target, m_target, x_ref, m_ref

        x_target_new = F.interpolate(x_target, size, mode='bilinear')
        m_target_new = F.interpolate(m_target, size, mode='nearest')

        x_ref_new = F.interpolate(
            x_ref.transpose(1, 2).reshape(b * ref_n, c, h, w), size,
            mode='bilinear',
        ).reshape(b, ref_n, c, size[0], size[1]).transpose(1, 2)
        m_ref_new = F.interpolate(
            m_ref.transpose(1, 2).reshape(b * ref_n, 1, h, w), size,
            mode='nearest',
        ).reshape(b, ref_n, 1, size[0], size[1]).transpose(1, 2)

        return x_target_new, m_target_new, x_ref_new, m_ref_new

    @staticmethod
    def crop(image, size):
        """Crops a patch of an image.

        Args:
            image: Tensor of size ``(C,F,H,W)`` containing original images
            size: Tuple of two positions containing the desired new size of the
                images.

        Returns:
            Patch of the image of size ``size``.
        """
        crop_position = (
            random.randint(0, image.size(2) - size[0]),
            random.randint(0, image.size(3) - size[1]),
        )
        crop_h_from, crop_h_to = crop_position[0], crop_position[0] + size[0]
        crop_w_from, crop_w_to = crop_position[1], crop_position[1] + size[1]
        return image[:, :, crop_h_from: crop_h_to, crop_w_from: crop_w_to], \
            crop_position

    @staticmethod
    def dilate(images, filter_size, iterations):
        """Applies a dilation filter of size `filter_size`` to the masks.

        Args:
            images: Tensor of size ``(1,F,H,W)`` containing the original masks.
            filter_size: Tuple of two positions containing the size of the
                filter.
            iterations: Number of times the filter is applied.

        Returns:
            Tensor of size ``(1,F,H,W)`` containing the masks after the filter.
        """
        images_dilated = torch.zeros(images.size())
        for f in range(images.size(1)):
            images_dilated[:, f] = torch.from_numpy(
                cv2.dilate(
                    images[:, f].permute(1, 2, 0).numpy(),
                    cv2.getStructuringElement(cv2.MORPH_CROSS, filter_size),
                    iterations=iterations
                )
            ).unsqueeze(0).float()
        return images_dilated
