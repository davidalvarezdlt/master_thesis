"""
Module containing the ``torch.utils.data.Dataset`` implementation of the
package.
"""
import os.path
import random

import cv2
import jpeg4py as jpeg
import numpy as np
import torch.utils.data

import master_thesis


class MasterThesisDataset(torch.utils.data.Dataset):
    """Implementation of the custom dataset class used in the package.

    Both the background and the mask will be either extracted from a real
    sequence or simulated using single real frames after applying affine
    transformations. The CLI arguments ``--p_simulator_bgs`` and
    ``-p_simulator_masks`` define the probability that a certain item
    data item has been obtained using the simulation mode.

    If ``self.kwargs['frames_n']`` is set to ``-1``, the entire sequence
    associated with the desired item is returned.

    Attributes:
        bgs_dataset: Instance of ``MasterThesisContentProvider`` returning
            background images.
        masks_dataset: Instance of ``MasterThesisContentProvider`` returning
            binary masks.
        split: Split of the dataset.
        kwargs: Dictionary containing the CLI arguments used in the execution.
    """
    FILL_COLOR = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)

    def __init__(self, bgs_dataset_meta, masks_dataset_meta, bgs_simulator,
                 masks_simulator, split, **kwargs):
        bgs_movement_min_height = kwargs['train_size'][0] * 2 \
            if split in ['train', 'validation'] else -1
        self.bgs_dataset = MasterThesisContentProvider(
            bgs_dataset_meta, bgs_movement_min_height, bgs_simulator, **kwargs
        )
        self.masks_dataset = MasterThesisContentProvider(
            masks_dataset_meta, -1, masks_simulator, **kwargs
        ) if masks_dataset_meta is not None else None
        self.image_size = kwargs['train_size'] \
            if split in ['train', 'validation'] else kwargs['test_size']
        self.split = split
        self.kwargs = kwargs

    def __getitem__(self, item):
        """Returns an item from the dataset, where an item is composed by a
        background image, its masked version, the binary mask and some
        auxiliary information.

        Args:
            item: Data index between 0 and ``self.__len__()``.

        Returns:
            Tuple of the form ``((x, m), y, (bg_name, bg_indexes,
            use_simulator_bgs, use_simulator_masks, 0, gt_movement,
            m_movement))``, where:
                - x: Masked sequence.
                - m: Binary shape used to mask ``y``.
                - y: Original background sequence before being masked.
                - bg_name: Name of the sequence used in the original dataset.
                - bg_indexes: Relative positions of the frames using a
                    comma-separated string.
                - use_simulator_bgs: Boolean indicating if the background
                  sequence has been obtained by applying random
                  transformations to a frame.
                - use_simulator_masks: Boolean indicating if the masks
                    sequence has been obtained by applying random
                    transformations to a mask.
                - gt_movement: Tensor containing the movement applied to the
                    background central frame, if the simulated movement mode
                    has been used.
                - m_movement: Tensor containing the movement applied to the
                    mask central frame, if the simulated movement mode has been
                    used.
        """
        item = self._sample_index(item)

        # The parameters p_simulator_bgs and p_simulator_masks define the
        # probability (between 0 and 1) that the a fake affine transformation
        # is applied to both the background and the mask, respectively.
        use_simulator_bgs = np.random.choice([False, True], p=[
            1 - self.kwargs['p_simulator_bgs'], self.kwargs['p_simulator_bgs']
        ])
        use_simulator_masks = np.random.choice([False, True], p=[
            1 - self.kwargs['p_simulator_masks'],
            self.kwargs['p_simulator_masks']
        ])

        # Check that it exists a movement simulator, if not replace whatever
        # value was generated by false
        if self.bgs_dataset is None or \
                self.bgs_dataset.movement_simulator is None:
            use_simulator_bgs = False

        if self.masks_dataset is None or \
                self.masks_dataset.movement_simulator is None:
            use_simulator_masks = False

        # When the argument frames_n is set to -1, we want to return the
        # entire sequence. This use case is required when inpainting an
        # entire video for test purposes.
        if self.kwargs['frames_n'] == -1:
            y, m, bg_name, bg_indexes, gt_movement, m_movement = \
                self.bgs_dataset.get_sequence(item)

        # During training we do not train using entire video sequences,
        # but small subsequences of frames_n frames.
        else:
            y, m, bg_name, bg_indexes, gt_movement, m_movement = \
                self.bgs_dataset.get_patch(
                    item, self.kwargs['frames_n'], use_simulator_bgs
                )

            # During training the datasets returning backgrounds and masks
            # will be different. In such a case, we need to get the fake
            # masks by making use of self.masks_dataset.
            if self.masks_dataset is not None:
                masks_n = self.kwargs['frames_n'] \
                    if self.kwargs['frames_n'] != -1 else y.size(1)

                _, m, m_name, m_indexes, _, m_movement = self.masks_dataset. \
                    get_patch_random(masks_n, use_simulator_masks)

        # We apply some transformations to both background data and the
        # masks. We also increase the size of the mask by applying a dilatation
        # operator.
        if self.kwargs['image_resize']:
            y = master_thesis.TransformsUtils.resize(
                y, self.image_size, keep_ratio=False
            )
            gt_movement = master_thesis.FlowsUtils.resize_flow(
                gt_movement.unsqueeze(0), self.image_size
            ).squeeze(0)
        else:
            y, crop_position = master_thesis.TransformsUtils.crop(
                y, self.image_size
            )
            gt_movement = master_thesis.FlowsUtils.crop_flow(
                gt_movement.unsqueeze(0), self.image_size, crop_position
            ).squeeze(0)

        if self.image_size != [m.size(2), m.size(3)]:
            m = master_thesis.TransformsUtils.resize(
                m, self.image_size, mode='nearest', keep_ratio=False
            )
            m_movement = master_thesis.FlowsUtils.resize_flow(
                m_movement.unsqueeze(0), self.image_size
            ).squeeze(0)

        m = master_thesis.TransformsUtils.dilate(
            m,
            self.kwargs['dilatation_filter_size'],
            self.kwargs['dilatation_iterations'],
        )

        # We compute the value of x by removing the content of the original
        # image y in those positions where the mask is 1. Instead of filling
        # the void using white, we fill it with a gray defined in
        # self.fill_color.
        x = (1 - m) * y + \
            (m.permute(3, 2, 1, 0) * self.FILL_COLOR).permute(3, 2, 1, 0)

        # The variable bg_indexes contains the relative spacing between the
        # central frame (relative spacing of 0) and its neighbors. We
        # transform this list into a comma-separated string.
        bg_indexes = ','.join(
            [str(gti - bg_indexes[len(bg_indexes) // 2]) for gti in bg_indexes]
            if bg_indexes
            else ['-' for _ in range(self.kwargs['frames_n'])]
        )

        return (x, m), y, (bg_name, bg_indexes, use_simulator_bgs,
                           use_simulator_masks, gt_movement, m_movement)

    def __len__(self):
        """Returns the length of the dataset. If the argument frames_n is
        set to -1, the length is the number of background videos available. If
        set to another value, the length is the sum of all the sequences
        frames.

        Returns:
            Number of dataset items using the given configuration.
        """
        if self.kwargs['frames_n'] == -1:
            return self.bgs_dataset.len_sequences()
        else:
            if self.split == 'train':
                max_iterations = self.kwargs['train_max_iterations']
            elif self.split == 'validation':
                max_iterations = self.kwargs['validation_max_iterations']
            else:
                max_iterations = 1
            return self.kwargs['batch_size'] * max_iterations

    def _sample_index(self, item):
        """Samples a new value for ``item`` in ``self.__getitem__``.

        If ``self.kwargs['frames_n']`` is set to ``-1``, the value is returned
        as it is. Otherwise, we completely replace the value by a random index
        between ``[0, len(self.bgs_dataset)``.

        Args:
            item: Original value of ``ìtem`` in ``self.__getitem__``.

        Returns:
            New value of ``item``.
        """
        if self.kwargs['frames_n'] == -1:
            return item
        else:
            return random.randint(0, len(self.bgs_dataset) - 1)


class MasterThesisContentProvider(torch.utils.data.Dataset):
    """Implementation of the custom dataset class used to extract either
    sequences of backgrounds or masks.

    Attributes:
        dataset_meta: Dictionary containing a mapping between the ids of
            the different data samples and their paths in disk.
        bgs_movement_min_height: Minimum height of the frame at which a random
            movement is applied.
        movement_simulator: Instance of ``master_thesis.utils.MovementsUtils``.
        items_names: List containing the names of the sequences.
        items_limits: List containing the cumulative lengths of the sequences.
        kwargs: Dictionary containing the CLI arguments used in the execution.
    """

    def __init__(self, dataset_meta, bgs_movement_min_height,
                 movement_simulator, **kwargs):
        self.dataset_meta = dataset_meta
        self.bgs_movement_min_height = bgs_movement_min_height
        self.movement_simulator = movement_simulator
        self.items_names = list(self.dataset_meta.keys())
        self.items_limits = np.cumsum([
            len(self.dataset_meta[item_name][0])
            if self.dataset_meta[item_name][0] is not None
            else len(self.dataset_meta[item_name][1])
            for item_name in self.items_names
        ])
        self.kwargs = kwargs

    def __getitem__(self, item):
        """Returns both the background and the mask associated with index
        ``frame_item``.

        Args:
            item: Frame index between 0 and ``self.__len__()``.

        Returns:
            Tuple of three positions containing:
                - Tensor of size ``(C,H,W)`` containing a background image
                    quantized between [0,1] or ``None``.
                - Tensor of size ``(1,H,W)`` containing a mask image quantized
                    between [0, 1] or ``None``.
                - Name of the sequence.
        """
        sequence_index = next(
            x[0] for x in enumerate(self.items_limits) if x[1] > item
        )
        frame_index_bis = item - (
            self.items_limits[sequence_index - 1] if sequence_index > 0 else 0
        )

        y = self._get_item_background(sequence_index, frame_index_bis)
        m = self._get_item_mask(sequence_index, frame_index_bis)

        return y, m, self.items_names[sequence_index]

    def _get_item_background(self, sequence_index, frame_index_bis):
        """Returns the background associated with the frame ``frame_index_bis``
        of ``sequence_index``. It will return ``None`` if the path at that
        position is also ``None``.

        Args:
            sequence_index: Index of the sequence in ``self.items_names``.
            frame_index_bis: Index of the frame in
                ``self.dataset_meta[sequence_name][0]``.

        Returns:
            Tensor of size ``(C,H,W)`` containing a background frame quantized
            between [0,1] or ``None``.
        """
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][0] is None:
            return None

        item_path = os.path.join(
            self.kwargs['data_path'],
            self.dataset_meta[item_name][0][frame_index_bis]
        )
        return torch.from_numpy(jpeg.JPEG(item_path).decode() / 255) \
            .permute(2, 0, 1).float()

    def _get_item_mask(self, sequence_index, frame_index_bis):
        """Returns the mask associated to frame ``frame_index_bis`` of
        ``sequence_index``. It will return None if the path at that position
        is ``None``.

        Args:
            sequence_index: Index of the sequence in ``self.items_names``.
            frame_index_bis: Index of the frame in
                ``self.dataset_meta[sequence_name][1]``.

        Returns:
            Tensor of size ``(C,H,W)`` containing a mask image quantized
            between [0,1] or ``None``.
        """
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][1] is None:
            return None

        item_path = os.path.join(
            self.kwargs['data_path'],
            self.dataset_meta[item_name][1][frame_index_bis]
        )
        return torch.from_numpy(
            cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) / 255 > 0
        ).float()

    def get_items(self, frames_indexes):
        """Returns the backgrounds and the masks at indexes ``frames_indexes``.

        Args:
            frames_indexes: List of indexes of the frames that should be
                returned.

        Returns:
            Tuple of two positions containing:
                - Tensor of size ``(C,F,H,W)`` containing the backgrounds.
                - Tensor of size ``(1,F,H,W)`` containing the masks.
        """
        y, m = None, None
        y0, m0, _ = self.__getitem__(frames_indexes[0])
        if y0 is not None:
            y = torch.zeros(
                (3, len(frames_indexes), y0.size(1), y0.size(2)),
                dtype=torch.float32
            )
            y[:, 0] = y0
        if m0 is not None:
            m = torch.zeros(
                (1, len(frames_indexes), m0.size(0), m0.size(1)),
                dtype=torch.float32
            )
            m[:, 0] = m0.unsqueeze(0)
        for i in range(1, len(frames_indexes)):
            yi, mi, _ = self.__getitem__(frames_indexes[i])
            if y is not None:
                y[:, i] = yi
            if m is not None:
                m[:, i] = mi
        return y, m

    def __len__(self):
        """Returns the sum of the frames of all sequences.

        Returns:
            Length of the dataset, given by the sum of the frames of all
            sequences.
        """
        return self.items_limits[-1]

    def get_sequence(self, sequence_index):
        """Returns the sequence with index ``sequence_index``.

        Args:
            sequence_index: Sequence index between 0 and
                ``self.len_sequences()``.

        Returns:
            Tuple of six position containing:
                - Tensor of size ``(C,frames_n,H,W)`` containing the background
                    frames, or ``None``.
                - Tensor of size ``(1,frames_n,H,W)`` containing the masks
                    frames, or ``None``.
                - Name of the sequence.
                - List containing the indexes of the frames, or ``None``.
                - Tensor containing the random movement applied to the central
                    background frame, or ``None``.
                - Tensor containing the random movement applied to the central
                    mask frame, or ``None``.
        """
        sequence_first_frame_index = (
            self.items_limits[sequence_index - 1] if sequence_index > 0 else 0
        )
        sequence_last_frame_index = self.items_limits[sequence_index] - 1
        frames_indexes = list(
            range(sequence_first_frame_index, sequence_last_frame_index + 1)
        )

        y, m = self.get_items(frames_indexes)
        gt_movement = torch.zeros(
            (len(frames_indexes), y.size(2), y.size(3), 2)
        )
        m_movement = torch.zeros(
            (len(frames_indexes), m.size(2), m.size(3), 2)
        )

        return y, m, self.items_names[sequence_index], frames_indexes, \
            gt_movement, m_movement

    def len_sequences(self):
        """Return the number of different sequences."""
        return len(self.items_names)

    def get_patch(self, frame_index, frames_n, use_movement_simulator):
        """Returns a patch of ``frames_n`` frames centered around the frame with
        index ``frame_index``.

        Args:
            frame_index: Index of the central frame.
            frames_n: Number of frames to return, including the central frame.
            use_movement_simulator: Whether or not to obtain reference frames
                using a movement simulator.

        Returns:
            Tuple of six position containing:
                - Tensor of size ``(C,frames_n,H,W)`` containing the background
                    frames, or ``None``.
                - Tensor of size ``(1,frames_n,H,W)`` containing the masks
                    frames, or ``None``.
                - Name of the sequence.
                - List containing the indexes of the frames, or ``None``.
                - Tensor containing the random movement applied to the central
                    background frame, or ``None``.
                - Tensor containing the random movement applied to the central
                    mask frame, or ``None``.
        """
        if use_movement_simulator and self.movement_simulator is not None:
            return self._get_patch_simulated(frame_index, frames_n)
        else:
            return self._get_patch_contiguous(
                frame_index,
                frames_n,
                self.kwargs['frames_spacing'],
                self.kwargs['frames_randomize'],
            )

    def get_patch_random(self, frames_n, use_movement_simulator):
        """Returns a patch of ``frames_n`` frames centered around a random
        frame.

        Args:
            frames_n: Number of frames to return, including the central frame.
            use_movement_simulator: Whether or not to obtain reference frames
                using a movement simulator.

        Returns:
            Tuple of six position containing:
                - Tensor of size ``(C,frames_n,H,W)`` containing the background
                    frames, or ``None``.
                - Tensor of size ``(1,frames_n,H,W)`` containing the masks
                    frames, or ``None``.
                - Name of the sequence.
                - List containing the indexes of the frames, or ``None``.
                - Tensor containing the random movement applied to the central
                    background frame, or ``None``.
                - Tensor containing the random movement applied to the central
                    mask frame, or ``None``.
        """
        return self.get_patch(random.randint(0, self.__len__() - 1), frames_n,
                              use_movement_simulator)

    def _get_patch_contiguous(self, frame_index, frames_n, frames_spacing,
                              randomize_frames):
        """Returns a patch of ``frames_n`` contiguous frames centered around the
        frame with index ``frame_index``.

        Args:
            frame_index: Index of the central frame.
            frames_n: Number of frames to return, including the central frame.
            frames_spacing: Minimum separation between returned frames.
            randomize_frames: Whether or not to randomize the order of returned
                samples, meaning that frames with smaller index might appear in
                the future with respect to ``frame_index``.

        Returns:
            Tuple of six position containing:
                - Tensor of size ``(C,frames_n,H,W)`` containing the background
                    frames, or ``None``.
                - Tensor of size ``(1,frames_n,H,W)`` containing the masks
                    frames, or ``None``.
                - Name of the sequence.
                - List containing the indexes of the frames, or ``None``.
                - Tensor containing the random movement applied to the central
                    background frame, or ``None``.
                - Tensor containing the random movement applied to the central
                    mask frame, or ``None``.
        """
        if not (frames_n % 2 == 1 or frames_n == 2):
            raise ValueError(
                'The value of --frames_n must be either 2 or an even number.'
            )

        sequence_item = next(
            x[0] for x in enumerate(self.items_limits) if x[1] > frame_index
        )

        sequence_first_frame_index = (
            self.items_limits[sequence_item - 1] if sequence_item > 0 else 0
        )
        sequence_last_frame_index = self.items_limits[sequence_item] - 1

        frame_indexes_candidates_pre = list(
            range(frame_index - (frames_n // 2) * frames_spacing, frame_index)
        )
        frame_indexes_candidates_post = list(range(
            frame_index + 1,
            frame_index + (frames_n // 2) * frames_spacing + 1
        ))

        frame_indexes_candidates_pre = [
            max(i, sequence_first_frame_index)
            for i in frame_indexes_candidates_pre
        ]
        frame_indexes_candidates_post = [
            min(i, sequence_last_frame_index)
            for i in frame_indexes_candidates_post
        ]

        if randomize_frames:
            # In randomize mode there is no symmetry between past and future
            # frames. Indexes are not sorted, and it may be possible that
            # indexes on the left of the target (middle element) are bigger.
            frame_indexes_candidates = set(frame_indexes_candidates_pre).union(
                set(frame_indexes_candidates_post)
            )
            if frame_index in frame_indexes_candidates:
                frame_indexes_candidates.remove(frame_index)
            frames_indexes = sorted(
                random.sample(frame_indexes_candidates, frames_n - 1)
            )
            frames_indexes.insert(frames_n // 2, frame_index)

        else:
            frames_indexes_before = \
                frame_indexes_candidates_pre[::frames_spacing]
            frames_indexes_after = \
                frame_indexes_candidates_post[::frames_spacing] \
                if frames_n > 2 else []

            frames_indexes = frames_indexes_before + [frame_index] + \
                frames_indexes_after

        y, m = self.get_items(frames_indexes)
        gt_movement = None if y is None \
            else torch.zeros((len(frames_indexes), y.size(2), y.size(3), 2))
        m_movement = None if m is None \
            else torch.zeros((len(frames_indexes), m.size(2), m.size(3), 2))

        return y, m, self.items_names[sequence_item], frames_indexes, \
            gt_movement, m_movement

    def _get_patch_simulated(self, frame_index, frames_n):
        """Returns a patch of ``frames_n`` simulated frames centered around the
        frame with index ``frame_index``.

        Args:
            frame_index: Index of the central frame.
            frames_n: Number of frames to return, including the central frame.

        Returns:
            Tuple of six position containing:
                - Tensor of size ``(C,frames_n,H,W)`` containing the background
                    frames, or ``None``.
                - Tensor of size ``(1,frames_n,H,W)`` containing the masks
                    frames, or ``None``.
                - Name of the sequence.
                - List containing the indexes of the frames, or ``None``.
                - Tensor containing the random movement applied to the central
                    background frame, or ``None``.
                - Tensor containing the random movement applied to the central
                    mask frame, or ``None``.
        """
        y, m, item_name = self.__getitem__(frame_index)
        m = m.unsqueeze(0) if m is not None and len(m.size()) == 2 else m
        gt_movement, m_movement, affine_transform = None, None, None

        if y is not None:
            if self.bgs_movement_min_height != -1 \
                    and y.size(1) < self.bgs_movement_min_height:
                y = master_thesis.TransformsUtils.resize(
                    y.unsqueeze(1), (self.bgs_movement_min_height, -1)
                )
                y = y.squeeze(1)
            y, gt_movement, affine_transform = \
                self.movement_simulator.simulate_movement(
                    y, frames_n, gt_movement
                )

        if m is not None:
            m, m_movement, _ = self.movement_simulator.simulate_movement(
                m, frames_n, affine_transform
            )

        return y, m, item_name, None, gt_movement, m_movement