"""
Module containing the ``pytorch_lightning.LightningDataModule`` implementation
of the package.
"""
import glob
import os.path
import pickle
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch.utils.data

import master_thesis


class MasterThesisData(pl.LightningDataModule):
    """Class used to handle data pre-processing steps and persist data splits.

    Attributes:
        train_bgs_meta: Dictionary containing a mapping between the ids of
            the different data samples of the backgrounds training dataset and
            their paths in disk.
        train_masks_meta: Dictionary containing a mapping between the ids of
            the different data samples of the masks training dataset and
            their paths in disk.
        validation_bgs_meta: Dictionary containing a mapping between the ids of
            the different data samples of the backgrounds validation dataset
            and their paths in disk.
        validation_masks_meta: Dictionary containing a mapping between the ids
            of the different data samples of the backgrounds validation
            dataset and their paths in disk.
        test_meta: Dictionary containing a mapping between the ids of
            the test dataset samples and their paths in disk.
        kwargs: dictionary containing the CLI arguments required to
            create an instance of ``MasterThesisData``.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.train_bgs_meta = None
        self.train_masks_meta = None
        self.validation_bgs_meta = None
        self.validation_masks_meta = None
        self.test_meta = None
        self.kwargs = kwargs

    def prepare_data(self):
        """Prepares the data used to train the model.

        1. Fills the meta class attributes with the mapping between data
           samples and their paths, for the different datasets and splits.
        2. Cleans those masks samples which are either too small or too big,
           defined by the ``--min_mask_size``  and ``--max_mask_size`` CLI
           arguments.
        3. Samples a small portion of the data so that validation and test
           pipelines are consistent between epochs.

        The results of this method will be saved in ``--data_ckpt_path`` so
        that subsequent calls do not have to wait for this pre-processing
        step to finish running.
        """
        if os.path.exists(self.kwargs['data_ckpt_path']):
            with open(self.kwargs['data_ckpt_path'], 'rb') as data_ckpt:
                (
                    self.train_bgs_meta,
                    self.train_masks_meta,
                    self.validation_bgs_meta,
                    self.validation_masks_meta,
                    self.test_meta
                ) = pickle.load(data_ckpt)
        else:
            self._prepare_data_meta()
            self._prepare_data_clean()
            with open(self.kwargs['data_ckpt_path'], 'wb') as data_ckpt:
                pickle.dump((
                    self.train_bgs_meta,
                    self.train_masks_meta,
                    self.validation_bgs_meta,
                    self.validation_masks_meta,
                    self.test_meta
                ), data_ckpt)

    def _prepare_data_meta(self):
        self.train_bgs_meta = MasterThesisData.get_meta_got10k(
            self.kwargs['data_path'], 'train'
        )
        self.train_masks_meta = MasterThesisData.get_meta_youtube_vos(
            self.kwargs['data_path'], 'train'
        )
        self.validation_bgs_meta = MasterThesisData.get_meta_got10k(
            self.kwargs['data_path'], 'validation'
        )
        self.validation_masks_meta = MasterThesisData.get_meta_youtube_vos(
            self.kwargs['data_path'], 'validation'
        )
        self.test_meta = MasterThesisData.get_meta_davis(
            self.kwargs['data_path']
        )

    def _prepare_data_clean(self):
        if self.kwargs['max_mask_size'] is None:
            return

        for masks_meta in [self.train_masks_meta, self.validation_masks_meta]:
            for train_masks_item in list(masks_meta.keys()):
                mask_item_values = []
                samples_paths = random.sample(
                    masks_meta[train_masks_item][1],
                    min(10, len(masks_meta[train_masks_item][1])),
                )

                for train_masks_item_path in samples_paths:
                    data_path = self.kwargs['data_path'].__str__()
                    item_path = os.path.join(data_path, train_masks_item_path)
                    image = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) > 0
                    mask_item_values.append(
                        np.count_nonzero(image) /
                        (image.shape[0] * image.shape[1])
                    )

                mask_size = np.mean(mask_item_values)
                if mask_size <= self.kwargs['min_mask_size'] \
                        or mask_size >= self.kwargs['max_mask_size']:
                    masks_meta.pop(train_masks_item)

    def train_dataloader(self):
        """Returns the data loader containing the training data.

        Returns:
            Data loader containing the training data.
        """
        train_dataset = master_thesis.MasterThesisDataset(
            bgs_dataset_meta=self.train_bgs_meta,
            masks_dataset_meta=self.train_masks_meta,
            bgs_simulator=master_thesis.MovementsUtils(
                *self.kwargs['bgs_movement_params']
            ),
            masks_simulator=master_thesis.MovementsUtils(
                *self.kwargs['masks_movement_params']
            ),
            split='train',
            **self.kwargs
        )
        return torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.kwargs['batch_size'],
            num_workers=self.kwargs['num_workers'],
            worker_init_fn=MasterThesisData.load_loaders_fn,
        )

    def val_dataloader(self):
        """Returns the data loader containing the validation data.

        Returns:
            Data loader containing the validation data.
        """
        val_dataset = master_thesis.MasterThesisDataset(
            bgs_dataset_meta=self.validation_bgs_meta,
            masks_dataset_meta=self.validation_masks_meta,
            bgs_simulator=None,
            masks_simulator=None,
            split='validation',
            **self.kwargs
        )
        return torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.kwargs['batch_size'],
            num_workers=self.kwargs['num_workers'],
            worker_init_fn=MasterThesisData.load_loaders_fn,
        )

    def test_dataloader(self):
        """Returns the data loader containing the test data.

        Returns:
            Data loader containing the test data.
        """
        test_dataset = master_thesis.MasterThesisDataset(
            bgs_dataset_meta=self.test_meta,
            masks_dataset_meta=None,
            bgs_simulator=None,
            masks_simulator=None,
            split='test',
            **self.kwargs
        )
        return torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.kwargs['batch_size'],
            num_workers=self.kwargs['num_workers'],
            worker_init_fn=MasterThesisData.load_loaders_fn
        )

    @staticmethod
    def load_loaders_fn(_):
        """Initializes the different random seeds to a random value between
        0 and 1 billion."""
        random_seed = random.randint(0, 1000000000)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    @staticmethod
    def get_meta_got10k(data_folder, split):
        """Returns the metadata of the GOT-10k dataset.

        Args:
            data_folder: folder where the dataset is stored.
            split: split from which to extract the data. Can be either
            ``train`` or ``validation``.

        Returns:
            Dictionary associating sequence ids with the paths in disk of its
            individual frame backgrounds.
        """
        dataset_folder = os.path.join(data_folder, 'GOT10k')
        split_folder = 'train' if split == 'train' else 'val'
        items_file = open(
            os.path.join(dataset_folder, split_folder, 'list.txt')
        )

        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            if os.path.exists(os.path.join(
                    dataset_folder, split_folder, item_name
            )):
                item_bgs_paths = sorted(glob.glob(os.path.join(
                    dataset_folder, split_folder, item_name, '*.jpg'
                )))
                item_bgs_paths = [
                    os.path.relpath(path, data_folder)
                    for path in item_bgs_paths
                ]

                if len(item_bgs_paths) > 0:
                    items_meta[item_name] = (item_bgs_paths, None)

        return items_meta

    @staticmethod
    def get_meta_youtube_vos(data_folder, split):
        """Returns the metadata of the YouTubeVOS dataset.

        The original YouTubeVOS dataset only contains a mask for the entire
        sequence in its training split. Therefore, to simulate a validation
        split we take the original training split and divide it in 90% training
        and 10% validation.

        Args:
            data_folder: folder where the dataset is stored.
            split: split from which to extract the data. Can be either
            ``train`` or ``validation``.

        Returns:
            Dictionary associating sequence ids with the paths in disk of its
            individual frame masks.
        """
        samples_paths = sorted(os.listdir(os.path.join(
            data_folder, 'YouTubeVOS', 'train', 'Annotations'
        )))
        random.Random(0).shuffle(samples_paths)  # Same order every time
        split_paths = (
            samples_paths[: int(0.9 * len(samples_paths))]
            if split == 'train'
            else samples_paths[int(0.9 * len(samples_paths)):]
        )

        items_meta = {}
        for item_name in split_paths:
            item_masks_paths = sorted(glob.glob(os.path.join(
                data_folder, 'YouTubeVOS', 'train', 'Annotations',
                item_name, '*.png'
            )))
            item_masks_paths = [
                os.path.relpath(path, data_folder) for path in item_masks_paths
            ]

            items_meta[item_name] = (None, item_masks_paths)

        return items_meta

    @staticmethod
    def get_meta_davis(data_folder):
        """Returns the metadata of the DAVIS dataset.

        Args:
            data_folder: folder where the dataset is stored.

        Returns:
            Dictionary associating sequence ids with the paths in disk of its
            individual frames, both backgrounds and masks.
        """
        dataset_folder = os.path.join(data_folder, 'DAVIS-2017')
        items_file = open(
            os.path.join(dataset_folder, 'ImageSets', 'custom.txt')
        )
        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            item_bgs_paths = sorted(glob.glob(os.path.join(
                dataset_folder, 'JPEGImages', '480p', item_name, '*.jpg'
            )))
            item_bgs_paths = [
                os.path.relpath(path, data_folder) for path in item_bgs_paths
            ]

            item_masks_path = sorted(glob.glob(os.path.join(
                dataset_folder, 'Annotations_Dense', '480p', item_name, '*.png'
            )))
            item_masks_path = [
                os.path.relpath(path, data_folder) for path in item_masks_path
            ]

            items_meta[item_name] = (item_bgs_paths, item_masks_path)

        return items_meta

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-specific arguments so that they can be modified
        directly from the CLI.

        Args:
            parent_parser: parser object just before adding data-specific
            arguments.

        Returns:
            Parser object after adding data-specific arguments.
        """
        parser = parent_parser.add_argument_group('Data Arguments')
        parser.add_argument('--data_path', default='./data')
        parser.add_argument(
            '--data_ckpt_path', default='./lightning_logs/data.ckpt'
        )
        parser.add_argument('--image_resize', type=bool, default=True)
        parser.add_argument('--min_mask_size', type=float, default=0.05)
        parser.add_argument('--max_mask_size', type=float, default=0.15)
        parser.add_argument(
            '--train_size', type=int, nargs='+', default=[256, 256]
        )
        parser.add_argument(
            '--test_size', type=int, nargs='+', default=[240, 480]
        )
        parser.add_argument('--frames_n', type=int, default=2)
        parser.add_argument('--frames_spacing', type=int, default=10)
        parser.add_argument('--frames_randomize', type=bool, default=True)
        parser.add_argument(
            '--dilatation_filter_size', type=int, nargs='+', default=[3, 3]
        )
        parser.add_argument('--dilatation_iterations', type=int, default=4)
        parser.add_argument('--p_simulator_bgs', type=float, default=0.5)
        parser.add_argument('--p_simulator_masks', type=float, default=0.0)
        parser.add_argument(
            '--bgs_movement_params', type=float, nargs='+',
            default=[50, 0.10, 0.20]
        )
        parser.add_argument(
            '--masks_movement_params', type=float, nargs='+',
            default=[50, 0.10, 0.20]
        )
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--train_max_iterations', type=int, default=2000)
        parser.add_argument('--validation_max_iterations', type=int,
                            default=200)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr_scheduler_step_size', type=int, default=50)
        parser.add_argument('--lr_scheduler_gamma', type=int, default=0.5)
        return parent_parser
