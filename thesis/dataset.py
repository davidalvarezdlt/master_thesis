import torch.utils.data
import numpy as np
import random
import jpeg4py as jpeg
import cv2
import os.path
import utils.transforms
import matplotlib.pyplot as plt
import utils.flow
import torch.nn.functional as F
import skimage.draw


class ContentProvider(torch.utils.data.Dataset):
    data_folder = None
    dataset_meta = None
    gts_movement_min_height = None
    logger = None
    items_names = None
    items_limits = None
    _ram_data = None

    def __init__(self, data_folder, dataset_meta, logger, gts_movement_min_height=-1, load_in_ram=False):
        self.data_folder = data_folder
        self.dataset_meta = dataset_meta
        self.logger = logger
        self.gts_movement_min_height = gts_movement_min_height
        self.items_names = list(self.dataset_meta.keys())
        self.items_limits = np.cumsum([
            len(self.dataset_meta[item_name][0]) if self.dataset_meta[item_name][0] is not None
            else len(self.dataset_meta[item_name][1]) for item_name in self.items_names
        ])
        if load_in_ram:
            self._load_data_in_ram()

    def __getitem__(self, frame_index):
        """Returns both the GT and the mask associated with index ``frame_item``.

        Args:
            frame_index (int): frame index between 0 and ``self.__len__()``.

        Returns:
            torch.FloatTensor: tensor of size (C,H,W) containing a GT image quantized between [0,1] or None.
            torch.FloatTensor: tensor of size (C,H,W) containing a mask image quantized between [0,1] or None.
            str: name of the sequence.
        """
        sequence_index = next(x[0] for x in enumerate(self.items_limits) if x[1] > frame_index)
        frame_index_bis = frame_index - (self.items_limits[sequence_index - 1] if sequence_index > 0 else 0)
        y = self._get_item_background(sequence_index, frame_index_bis)
        m = self._get_item_mask(sequence_index, frame_index_bis, y.shape if y is not None else None)
        return y, m, self.items_names[sequence_index]

    def _get_item_background(self, sequence_index, frame_index_bis):
        """Returns the GT associated to frame `frame_index_bis` of `sequence_index`. It will return None if the path
        at that position is None.

        Returns:
            torch.FloatTensor: tensor of size (C,H,W) containing a GT image quantized between [0,1] or None.
        """
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][0] is None:
            return None
        if self._ram_data is not None and self._ram_data[item_name][0][frame_index_bis] is not None:
            return self._ram_data[item_name][0][frame_index_bis]
        item_path = os.path.join(self.data_folder, self.dataset_meta[item_name][0][frame_index_bis])
        return torch.from_numpy(jpeg.JPEG(item_path).decode() / 255).permute(2, 0, 1).float()

    def _get_item_mask(self, sequence_index, frame_index_bis, frame_size):
        """Returns the mask associated to frame `frame_index_bis` of `sequence_index`. It will return None if the path
        at that position is None.

        Returns:
            torch.FloatTensor: tensor of size (C,H,W) containing a mask image quantized between [0,1].
        """
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][1] is None:
            return None
        elif type(self.dataset_meta[item_name][1][frame_index_bis]) is list:
            return self._get_item_mask_got(self.dataset_meta[item_name][1][frame_index_bis], frame_size)
        elif self._ram_data is not None and self._ram_data[item_name][1][frame_index_bis] is not None:
            return self._ram_data[item_name][1][frame_index_bis]
        item_path = os.path.join(self.data_folder, self.dataset_meta[item_name][1][frame_index_bis])
        return torch.from_numpy(cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) / 255 > 0).float()

    def _get_item_mask_got(self, mask_coords, frame_size):
        m, m_coords = np.zeros((frame_size[1], frame_size[2])), list(map(int, map(float, mask_coords)))
        rr, cc = skimage.draw.rectangle(
            (m_coords[1], m_coords[0]), extent=(m_coords[3], m_coords[2]), shape=(frame_size[1], frame_size[2])
        )
        m[rr, cc] = 1
        return torch.from_numpy(m)

    def get_items(self, frames_indexes):
        y, m = None, None
        y0, m0, _ = self.__getitem__(frames_indexes[0])
        if y0 is not None:
            y = torch.zeros((3, len(frames_indexes), y0.size(1), y0.size(2)), dtype=torch.float32)
            y[:, 0] = y0
        if m0 is not None:
            m = torch.zeros((1, len(frames_indexes), m0.size(0), m0.size(1)), dtype=torch.float32)
            m[:, 0] = m0.unsqueeze(0)
        for i in range(1, len(frames_indexes)):
            yi, mi, _ = self.__getitem__(frames_indexes[i])
            if y is not None:
                y[:, i] = yi
            if m is not None:
                m[:, i] = mi
        return y, m

    def __len__(self):
        """Returns the sum of the frames of all sequences."""
        return self.items_limits[-1]

    def get_sequence(self, sequence_index):
        """Returns the sequence with index ``sequence_item``.

        Args:
            sequence_index (int): sequence index between 0 and ``self.len_sequences()``.

        Returns:
            torch.FloatTensor: sequence quantized between [0,1] with shape (C,F,H,W).
            str: name of the sequence.
        """
        sequence_first_frame_index = self.items_limits[sequence_index - 1] if sequence_index > 0 else 0
        sequence_last_frame_index = self.items_limits[sequence_index] - 1
        frames_indexes = list(range(sequence_first_frame_index, sequence_last_frame_index + 1))
        y, m = self.get_items(frames_indexes)
        gt_movement = torch.zeros((len(frames_indexes), y.size(2), y.size(3), 2))
        m_movement = torch.zeros((len(frames_indexes), m.size(2), m.size(3), 2))
        return y, m, self.items_names[sequence_index], frames_indexes, gt_movement, m_movement

    def len_sequences(self):
        """Return the number of different sequences."""
        return len(self.items_names)

    def get_patch(self, frame_index, frames_n, frames_spacing, randomize_frames, movement_simulator):
        if movement_simulator is not None:
            return self._get_patch_simulated(frame_index, frames_n, movement_simulator)
        else:
            return self._get_patch_contiguous(frame_index, frames_n, frames_spacing, randomize_frames)

    def get_patch_random(self, frames_n, frames_spacing, randomize_frames, movement_simulator):
        return self.get_patch(
            random.randint(0, self.__len__() - 1), frames_n, frames_spacing, randomize_frames, movement_simulator
        )

    def _get_patch_contiguous(self, frame_index, frames_n, frames_spacing, randomize_frames):
        assert frames_n % 2 == 1 or frames_n == 2
        sequence_item = next(x[0] for x in enumerate(self.items_limits) if x[1] > frame_index)
        sequence_first_frame_index = self.items_limits[sequence_item - 1] if sequence_item > 0 else 0
        sequence_last_frame_index = self.items_limits[sequence_item] - 1
        frame_indexes_candidates_pre = list(range(frame_index - (frames_n // 2) * frames_spacing, frame_index))
        frame_indexes_candidates_post = list(range(frame_index + 1, frame_index + (frames_n // 2) * frames_spacing + 1))
        frame_indexes_candidates_pre = [max(i, sequence_first_frame_index) for i in frame_indexes_candidates_pre]
        frame_indexes_candidates_post = [min(i, sequence_last_frame_index) for i in frame_indexes_candidates_post]
        if randomize_frames:
            # In randomize mode there is no symmetry between past and future frames. Indexes are not sorted, and it may
            # be possible that indexes on the left of the target (middle element) are bigger.
            frame_indexes_candidates = set(frame_indexes_candidates_pre).union(set(frame_indexes_candidates_post))
            if frame_index in frame_indexes_candidates:
                frame_indexes_candidates.remove(frame_index)
            frames_indexes = sorted(random.sample(frame_indexes_candidates, frames_n - 1))
            frames_indexes.insert(frames_n // 2, frame_index)
        else:
            frames_indexes_before = frame_indexes_candidates_pre[::frames_spacing]
            frames_indexes_after = frame_indexes_candidates_post[::frames_spacing] if frames_n > 2 else []
            frames_indexes = frames_indexes_before + [frame_index] + frames_indexes_after
        y, m = self.get_items(frames_indexes)
        gt_movement = None if y is None else torch.zeros((len(frames_indexes), y.size(2), y.size(3), 2))
        m_movement = None if m is None else torch.zeros((len(frames_indexes), m.size(2), m.size(3), 2))
        return y, m, self.items_names[sequence_item], frames_indexes, gt_movement, m_movement

    def _get_patch_simulated(self, frame_index, frames_n, movement_simulator):
        y, m, item_name = self.__getitem__(frame_index)
        m = m.unsqueeze(0) if m is not None and len(m.size()) == 2 else m
        gt_movement, m_movement = None, None
        if y is not None:
            if self.gts_movement_min_height != -1 and y.size(1) < self.gts_movement_min_height:
                y = utils.transforms.ImageTransforms.resize(y.unsqueeze(1), (self.gts_movement_min_height, -1))
                y = y.squeeze(1)
            y, gt_movement, affine_transform = movement_simulator.simulate_movement(y, frames_n, gt_movement)
        if m is not None:
            m, m_movement, _ = movement_simulator.simulate_movement(m, frames_n, affine_transform)
        return y, m, item_name, None, gt_movement, m_movement

    def _load_data_in_ram(self):
        self._ram_data = {}
        for dataset_item_key in self.dataset_meta.keys():
            self._ram_data[dataset_item_key] = [None, None]
            if self.dataset_meta[dataset_item_key][0] is not None:
                self._ram_data[dataset_item_key][0] = [None] * len(self.dataset_meta[dataset_item_key][0])
                self._load_data_in_ram_background(dataset_item_key)
            if self.dataset_meta[dataset_item_key][1] is not None:
                self._ram_data[dataset_item_key][1] = [None] * len(self.dataset_meta[dataset_item_key][1])
                self._load_data_in_ram_masks(dataset_item_key)

    def _load_data_in_ram_background(self, dataset_item_key):
        for i, item_path in enumerate(self.dataset_meta[dataset_item_key][1]):
            self._ram_data[dataset_item_key][0][i] = self._get_item_background(
                self.items_names.index(dataset_item_key), i
            )

    def _load_data_in_ram_masks(self, dataset_item_key):
        for i, item_path in enumerate(self.dataset_meta[dataset_item_key][1]):
            self._ram_data[dataset_item_key][1][i] = self._get_item_mask(self.items_names.index(dataset_item_key), i)


class MaskedSequenceDataset(torch.utils.data.Dataset):
    gts_dataset = None
    masks_dataset = None
    gts_simulator = None
    masks_simulator = None
    image_size = None
    frames_n = None
    frames_spacing = None
    frames_randomize = None
    dilatation_filter_size = None
    dilatation_iterations = None
    force_resize = None
    keep_ratio = None
    fill_color = None
    p_simulator_gts = None
    p_simulator_masks = None
    p_repeat = None

    def __init__(self, gts_dataset, masks_dataset, gts_simulator, masks_simulator, image_size, frames_n, frames_spacing,
                 frames_randomize, dilatation_filter_size, dilatation_iterations, force_resize, keep_ratio,
                 p_simulator_gts=0, p_simulator_masks=0, p_repeat=0):
        self.gts_dataset = gts_dataset
        self.masks_dataset = masks_dataset
        self.gts_simulator = gts_simulator
        self.masks_simulator = masks_simulator
        self.image_size = image_size
        self.frames_n = frames_n
        self.frames_spacing = frames_spacing
        self.frames_randomize = frames_randomize
        self.force_resize = force_resize
        self.keep_ratio = keep_ratio
        self.dilatation_filter_size = dilatation_filter_size
        self.dilatation_iterations = dilatation_iterations
        self.p_simulator_gts = p_simulator_gts
        self.p_simulator_masks = p_simulator_masks
        self.p_repeat = p_repeat
        self.fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        assert 0 <= self.p_simulator_gts <= 1
        assert 0 <= self.p_simulator_masks <= 1
        assert not (self.p_simulator_gts > 0 and self.gts_simulator is None)
        assert not (self.p_simulator_masks > 0 and self.masks_simulator is None)

    def __getitem__(self, item):
        # Define if the current item is going to be a real video or a simulated one
        use_simulator_gts = np.random.choice([False, True], p=[1 - self.p_simulator_gts, self.p_simulator_gts])
        use_simulator_masks = np.random.choice([False, True], p=[1 - self.p_simulator_masks, self.p_simulator_masks])
        gts_simulator_item = self.gts_simulator if use_simulator_gts == 1 else None
        masks_simulator_item = self.masks_simulator if use_simulator_masks == 1 else None

        # Define if the current item should be re-used in further iterations
        repeat_item = np.random.choice([False, True], p=[1 - self.p_repeat, self.p_repeat])

        # Return entire sequence. Used only in the test.
        if self.frames_n == -1:
            y, m, gt_name, gt_indexes, gt_movement, m_movement = self.gts_dataset.get_sequence(item)

        # Return patches of training data
        else:
            y, m, gt_name, gt_indexes, gt_movement, m_movement = self.gts_dataset.get_patch(
                item, self.frames_n, self.frames_spacing, self.frames_randomize, gts_simulator_item
            )

            # If self.gts_dataset and self.masks_dataset are not the same, obtain new mask
            # Added if m is None
            if self.masks_dataset is not None:
                masks_n = self.frames_n if self.frames_n != -1 else y.size(1)
                _, m, m_name, m_indexes, _, m_movement = self.masks_dataset.get_patch_random(
                    masks_n, self.frames_spacing, self.frames_randomize, masks_simulator_item
                )

        # Apply GT transformations
        if self.force_resize:
            y = utils.transforms.ImageTransforms.resize(y, self.image_size, keep_ratio=False)
            gt_movement = utils.flow.resize_flow(gt_movement.unsqueeze(0), self.image_size).squeeze(0)
        else:
            y, crop_position = utils.transforms.ImageTransforms.crop(y, self.image_size)
            gt_movement = utils.flow.crop_flow(gt_movement.unsqueeze(0), self.image_size, crop_position).squeeze(0)

        # Apply Mask transformations
        if self.image_size != (m.size(2), m.size(3)):  # keep_ratio = self.keep_ratio
            m = utils.transforms.ImageTransforms.resize(m, self.image_size, mode='nearest', keep_ratio=False)
            m_movement = utils.flow.resize_flow(m_movement.unsqueeze(0), self.image_size).squeeze(0)

        # Apply a dilatation to the mask
        m = utils.transforms.ImageTransforms.dilatate(m, self.dilatation_filter_size, self.dilatation_iterations)

        # Compute x
        x = (1 - m) * y + (m.permute(3, 2, 1, 0) * self.fill_color).permute(3, 2, 1, 0)

        # Prepare gt_indexes, which represents relative spacing between the frames
        if gt_indexes:
            gt_indexes = ','.join([str(gti - gt_indexes[len(gt_indexes) // 2]) for gti in gt_indexes])
        else:
            gt_indexes = ','.join(['-' for _ in range(self.frames_n)])

        # Return data
        return (x, m), y, (gt_name, gt_indexes, use_simulator_gts, use_simulator_masks, repeat_item, gt_movement,
                           m_movement)

    def __len__(self):
        return self.gts_dataset.len_sequences() if self.frames_n == -1 else len(self.gts_dataset)
