from skimage.transform import AffineTransform, warp
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MovementSimulator:

    def __init__(self, max_displacement, max_scaling, max_rotation):
        self.max_displacement = max_displacement
        self.max_scaling = max_scaling
        self.max_rotation = max_rotation

    def random_affine(self):
        tx, ty = np.random.randint(low=-self.max_displacement, high=self.max_displacement,
                                   size=2) if self.max_displacement > 0 else (0, 0)
        sx, sy = np.random.uniform(low=1 - self.max_scaling, high=1 + self.max_scaling, size=2)
        rot = np.random.uniform(low=-self.max_rotation, high=self.max_rotation)
        affine_matrix = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=rot).params
        return torch.from_numpy(affine_matrix).float()

    def simulate_movement(self, x, n, affine_matrices=None):
        """Simulates a moving sequence of ``n` frames using ``frame`` as starting point.

        Args:
            x (torch.FloatTensor): tensor of size (C,H,W) containing the first frame.
            n (int): number of frames of the sequence.
            affine_matrices (torch.FloatTensor): tensor of size (n,3,3) containing the transformations to apply.

        Returns:
            torch.FloatTensor: tensor of size (C,F,H,W) containing the moving sequence.
        """
        c, h, w = x.size()

        # Create a Tensor with n affine transformations. random_affine \in (n_frames, 3, 3)
        if affine_matrices is None:
            affine_matrices = [self.random_affine() for _ in range(n - 1)]
            affine_matrices = affine_matrices[:n // 2] + [MovementSimulator.identity_affine()] + \
                              affine_matrices[n // 2:]
        affine_matrices_inv = [MovementSimulator.affine_inverse(affine_mat) for affine_mat in affine_matrices]

        # Stack matrices
        affine_matrices_s, affine_matrices_inv = torch.stack(affine_matrices), torch.stack(affine_matrices_inv)

        # Stack affine transformations with respect to the central frame
        affine_matrices_s = MovementSimulator.stack_transformations(affine_matrices_s, t=n // 2)
        affine_matrices_inv = MovementSimulator.stack_transformations(affine_matrices_inv, t=n // 2)
        affine_matrices_theta = torch.stack([MovementSimulator.affine2theta(ra, h, w) for ra in affine_matrices_s])
        affine_matrices_inv_theta = torch.stack([
            MovementSimulator.affine2theta(ra, h, w) for ra in affine_matrices_inv])

        # Create the grid
        flow = F.affine_grid(affine_matrices_theta, [n, c, h, w], align_corners=True)
        flow_inv = F.affine_grid(affine_matrices_inv_theta, [n, c, h, w], align_corners=True)

        # Apply the flow to the target frame
        y = F.grid_sample(x.unsqueeze(0).repeat(n, 1, 1, 1), flow, align_corners=True)

        # Return both data_out and random_thetas_stacked
        return y.permute(1, 0, 2, 3), flow_inv, affine_matrices

    @staticmethod
    def identity_affine():
        affine_matrix = np.linalg.inv(AffineTransform(translation=(0, 0), scale=(1, 1), rotation=0).params)
        return torch.from_numpy(affine_matrix).float()

    @staticmethod
    def identity_affine_theta(h, w):
        return MovementSimulator.affine2theta(MovementSimulator.identity_affine(), h, w)

    @staticmethod
    def affine_inverse(affine):
        return torch.from_numpy(np.linalg.inv(affine))

    @staticmethod
    def transform_single(image, flow):
        c, h, w = image.size()
        transformation_theta = MovementSimulator.affine2theta(flow, h, w).unsqueeze(0)
        affine_grid = F.affine_grid(transformation_theta, [1, c, h, w])
        return F.grid_sample(image.unsqueeze(0), affine_grid).squeeze(0)

    @staticmethod
    def stack_transformations(affine_matrices, t):
        """Stacks a set of single transformations to apply `affine_grid` easily.

        Given a set of n independent `affine_matrices` and the reference frame at position t, it computes the
        transformation required to move from position t to [..., t-1, t-2, t+1, t+2, ...].
        """
        affine_matrices_stacked = torch.zeros(affine_matrices.size(), dtype=torch.float32)
        affine_matrices_stacked[t] = affine_matrices[t]
        for i in reversed(range(t)):
            affine_matrices_stacked[i] = torch.matmul(torch.inverse(affine_matrices[i]), affine_matrices_stacked[i + 1])
        for i in range(t + 1, len(affine_matrices)):
            affine_matrices_stacked[i] = torch.matmul(affine_matrices[i], affine_matrices_stacked[i - 1])
        return affine_matrices_stacked

    @staticmethod
    def affine2theta(param, h, w):
        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * h / w
        theta[0, 2] = param[0, 2] * 2 / w + param[0, 0] + param[0, 1] - 1
        theta[1, 0] = param[1, 0] * w / h
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / h + param[1, 0] + param[1, 1] - 1
        return torch.from_numpy(theta).float()
