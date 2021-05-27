import torch


def compute_masked_correlation(feats_1, feats_2, mask_1, mask_2):
    """Computes the normalized masked correlation pixel-by-pixel between two feature maps.

    Args:
        feats_1 (torch.FloatTensor): tensor of size (B,C,H,W) containing the features of the first image.
        feats_2 (torch.FloatTensor): tensor of size (B,C,H,W) containing the features of the second image.

    Returns:
        torch.FloatTensor: normalized pixel-by-pixel correlation between the feature maps of size (B,1,H,W).

    """
    b, c, h, w = feats_1.size()

    # Mask the features
    feats_1, feats_2 = feats_1 * mask_1, feats_2 * mask_2

    # Compute correlation multiplication and return
    feats_1 = feats_1.permute(0, 2, 3, 1).reshape(b * h * w, -1)
    feats_1_norm = torch.norm(feats_1, dim=1).unsqueeze(1) + 1e-9
    feats_2 = feats_2.permute(0, 2, 3, 1).reshape(b * h * w, -1)
    feats_2_norm = torch.norm(feats_2, dim=1).unsqueeze(1) + 1e-9
    return ((feats_1 / feats_1_norm) * (feats_2 / feats_2_norm)).sum(1).reshape(b, h, w).unsqueeze(1)


def compute_masked_4d_correlation(x_target_feats, v_target, x_ref_feats, v_ref):
    """Computes the normalized correlation between the feature maps of t and r_list.

    Args:
        x_target_feats (torch.FloatTensor): tensor of size (B,C,H,W) containing the feature map of the target frame.
        v_target (torch.FloatTensor): tensor of size (B,1,H,W) containing visibility map of the target frame.
        x_ref_feats (torch.FloatTensor): tensor of size (B,C,F,H,W) containing the feature maps of the reference frames.
        v_ref (torch.FloatTensor): tensor of size (B,1,H,W) containing visibility maps of the reference frames.

    Returns:
        torch.FloatTensor: 4D correlation volume of size (B,F,H,W,H,W).
    """
    b, c, ref_n, h, w = x_ref_feats.size()

    # Mask the features
    x_target_feats = x_target_feats * v_target if v_target is not None else x_target_feats
    x_ref_feats = x_ref_feats * v_ref if v_ref is not None else x_ref_feats

    # Compute the correlation with target frame.
    corr_1 = x_target_feats.reshape(b, c, -1).transpose(-1, -2).unsqueeze(1)
    corr_1_norm = torch.norm(corr_1, dim=3).unsqueeze(3) + 1e-9
    corr_2 = x_ref_feats.reshape(b, c, ref_n, -1).permute(0, 2, 1, 3)
    corr_2_norm = torch.norm(corr_2, dim=2).unsqueeze(2) + 1e-9
    corr = torch.matmul(corr_1 / corr_1_norm, corr_2 / corr_2_norm).reshape(b, ref_n, h, w, h, w)

    # Return 4D volume corr
    return corr
