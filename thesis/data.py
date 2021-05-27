import skeltorch
from .dataset import ContentProvider
from .dataset import MaskedSequenceDataset
import random
import torch.utils.data
import utils.movement
import utils.paths
import cv2
import os.path
import numpy as np


class ThesisData(skeltorch.Data):
    train_gts_meta = None
    train_masks_meta = None
    validation_gts_meta = None
    validation_masks_meta = None
    test_meta = None
    validation_frames_indexes = None
    validation_objective_measures_indexes = None
    test_frames_indexes = None
    test_sequences_indexes = None

    def create(self, data_path):
        self.train_gts_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'train_gts_dataset'),
            data_folder=data_path,
            split='train',
            return_masks=False
        )
        self.train_masks_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'train_masks_dataset'),
            data_folder=data_path,
            split='train',
            return_gts=False
        )
        self.validation_gts_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'validation_gts_dataset'),
            data_folder=data_path,
            split='validation',
            return_masks=False
        )
        self.validation_masks_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'validation_masks_dataset'),
            data_folder=data_path,
            split='train',
            return_gts=False
        )
        self.test_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'test_dataset'),
            data_folder=data_path,
            split='test'
        )

        # Clean masks that are too big
        if self.experiment.configuration.get('data', 'max_mask_size') is not None:
            self._clean_masks(data_path, self.train_masks_meta)
            self._clean_masks(data_path, self.validation_masks_meta)

        # Initialize test indexes
        validation_samples = sum([len(v[0]) for k, v in self.validation_gts_meta.items()])
        test_samples = sum([len(v[0]) for k, v in self.test_meta.items()])
        self.validation_frames_indexes = random.sample(list(range(validation_samples)), 20)
        self.validation_objective_measures_indexes = random.sample(list(range(test_samples)), 1000)
        self.test_frames_indexes = random.sample(list(range(test_samples)), 20)
        self.test_sequences_indexes = [12, 15, 22, 50, 53]

    def _clean_masks(self, data_path, masks_meta):
        train_masks_items = list(masks_meta.keys())
        for train_masks_item in train_masks_items:
            mask_item_values = []
            samples_paths = random.sample(
                masks_meta[train_masks_item][1], min(10, len(masks_meta[train_masks_item][1]))
            )
            for train_masks_item_path in samples_paths:
                item_path = os.path.join(data_path, train_masks_item_path)
                image = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) > 0
                mask_item_values.append(np.count_nonzero(image) / (image.shape[0] * image.shape[1]))
            if not (self.experiment.configuration.get('data', 'min_mask_size') <= np.mean(mask_item_values) <=
                    self.experiment.configuration.get('data', 'max_mask_size')):
                masks_meta.pop(train_masks_item)

    def load_datasets(self, data_path):
        gts_datasets = self._load_datasets_gts(data_path)
        masks_datasets = self._load_datasets_masks(data_path)
        self.datasets['train'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[0],
            masks_dataset=masks_datasets[0],
            gts_simulator=utils.movement.MovementSimulator(
                *self.experiment.configuration.get('data', 'gts_movement_params')
            ),
            masks_simulator=utils.movement.MovementSimulator(
                *self.experiment.configuration.get('data', 'masks_movement_params')
            ),
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            frames_randomize=self.experiment.configuration.get('data', 'frames_randomize'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=self.experiment.configuration.get('data', 'train_resize'),
            keep_ratio=True,
            p_simulator_gts=self.experiment.configuration.get('data', 'p_simulator_gts'),
            p_simulator_masks=self.experiment.configuration.get('data', 'p_simulator_masks'),
            p_repeat=self.experiment.configuration.get('data', 'p_repeat')
        )
        self.datasets['validation'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[1],
            masks_dataset=masks_datasets[1],
            gts_simulator=None,
            masks_simulator=None,
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            frames_randomize=self.experiment.configuration.get('data', 'frames_randomize'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=self.experiment.configuration.get('data', 'train_resize'),
            keep_ratio=True
        )
        self.datasets['test'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[2],
            masks_dataset=masks_datasets[1],
            gts_simulator=None,
            masks_simulator=None,
            image_size=tuple(self.experiment.configuration.get('data', 'test_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            frames_randomize=self.experiment.configuration.get('data', 'frames_randomize'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=self.experiment.configuration.get('data', 'train_resize'),
            keep_ratio=True
        )
        self.datasets['test_sequences'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[2],
            masks_dataset=masks_datasets[2],
            gts_simulator=None,
            masks_simulator=None,
            image_size=tuple(self.experiment.configuration.get('data', 'test_size')),
            frames_n=-1,
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            frames_randomize=self.experiment.configuration.get('data', 'frames_randomize'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=True,
            keep_ratio=False
        )

    def _load_datasets_gts(self, data_path):
        train_gts_movement_min_height = self.experiment.configuration.get('data', 'train_size')[0] * 2
        train_gts_dataset = ContentProvider(data_path, self.train_gts_meta, self.logger, train_gts_movement_min_height)
        validation_gts_dataset = ContentProvider(
            data_path, self.validation_gts_meta, self.logger, train_gts_movement_min_height
        )
        test_gts_dataset = ContentProvider(data_path, self.test_meta, self.logger)
        return train_gts_dataset, validation_gts_dataset, test_gts_dataset

    def _load_datasets_masks(self, data_path):
        train_masks_dataset = ContentProvider(data_path, self.train_masks_meta, self.logger)
        validation_masks_dataset = ContentProvider(data_path, self.validation_masks_meta, self.logger)
        return train_masks_dataset, validation_masks_dataset, None

    def load_loaders(self, data_path, num_workers):
        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=[]),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers,
            worker_init_fn=self.load_loaders_fn
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=[]),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers,
            worker_init_fn=self.load_loaders_fn
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=[]),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers,
            worker_init_fn=self.load_loaders_fn
        )

    def load_loaders_fn(self, worker_id):
        random_seed = random.randint(0, 1E9)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    def regenerate_loader_indexes(self, epoch):
        self.logger.info('Regenerating indexes for the data loaders...')
        batch_size = self.experiment.configuration.get('training', 'batch_size')

        # Generate training samples of the epoch
        train_max_items = batch_size * self.experiment.configuration.get('training', 'train_max_iterations')
        if len(self.datasets['train']) > train_max_items:
            train_gts_indexes = random.sample(list(range(len(self.datasets['train']))), train_max_items)
        else:
            train_gts_indexes = list(range(len(self.datasets['train'])))

        # Generate validation samples of the epoch
        validation_max_items = batch_size * self.experiment.configuration.get('training', 'validation_max_iterations')
        if len(self.datasets['validation']) > validation_max_items:
            val_gts_indexes = random.sample(list(range(len(self.datasets['validation']))), validation_max_items)
        else:
            val_gts_indexes = list(range(len(self.datasets['validation'])))

        # Generate test samples of the epoch
        test_max_items = batch_size * self.experiment.configuration.get('training', 'test_max_iterations')
        if len(self.datasets['test']) > test_max_items:
            test_gts_indexes = random.sample(list(range(len(self.datasets['test']))), test_max_items)
        else:
            test_gts_indexes = list(range(len(self.datasets['test'])))

        # Assign the indexes directly
        self.loaders['train'].sampler.indices = train_gts_indexes
        self.loaders['validation'].sampler.indices = val_gts_indexes
        self.loaders['test'].sampler.indices = test_gts_indexes
