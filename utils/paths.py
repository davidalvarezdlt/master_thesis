import os.path
import glob
import random


class DatasetPaths:

    @staticmethod
    def get_items(dataset_name, data_folder, split, return_gts=True, return_masks=True):
        assert return_gts or return_masks
        if dataset_name == 'davis-2017':
            return DatasetPaths.get_davis(data_folder, return_gts, return_masks)
        elif dataset_name == 'got-10k':
            return DatasetPaths.get_got10k(data_folder, split, return_gts, return_masks)
        elif dataset_name == 'youtube-vos':
            return DatasetPaths.get_youtube_vos(data_folder, split, return_gts, return_masks)

    @staticmethod
    def get_davis(data_folder, return_gts, return_masks):
        dataset_folder = os.path.join(data_folder, 'DAVIS-2017')
        split_filename = 'custom.txt'
        items_file = open(os.path.join(dataset_folder, 'ImageSets', '2017', split_filename))
        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            item_gts_paths = sorted(
                glob.glob(os.path.join(dataset_folder, 'JPEGImages', '480p', item_name, '*.jpg'))
            )
            item_masks_path = sorted(
                glob.glob(os.path.join(dataset_folder, 'Annotations_Dense', '480p', item_name, '*.png'))
            )
            item_gts_paths = [os.path.relpath(path, data_folder) for path in item_gts_paths] if return_gts else None
            item_masks_path = [os.path.relpath(path, data_folder) for path in item_masks_path] if return_masks else None
            items_meta[item_name] = (item_gts_paths, item_masks_path)
        return items_meta

    @staticmethod
    def get_got10k(data_folder, split, return_gts, return_masks):
        dataset_folder = os.path.join(data_folder, 'GOT10k')
        split_folder = 'train' if split == 'train' else 'val' if split == 'validation' else 'test'
        split_folder = 'val_with_mask' if split == 'validation' and return_masks else split_folder
        items_file = open(os.path.join(dataset_folder, split_folder, 'list.txt'))
        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            if os.path.exists(os.path.join(dataset_folder, split_folder, item_name)):
                item_gts_paths = sorted(glob.glob(os.path.join(dataset_folder, split_folder, item_name, '*.jpg')))
                item_gts_paths = [os.path.relpath(path, data_folder) for path in item_gts_paths]
                if return_masks:
                    gt_path = os.path.join(dataset_folder, split_folder, item_name, 'groundtruth.txt')
                    item_masks_path = [line.rstrip().split(',') for line in open(gt_path, 'r').readlines()]
                else:
                    item_masks_path = None
                if len(item_gts_paths) > 0:
                    items_meta[item_name] = (item_gts_paths, item_masks_path)
        return items_meta

    @staticmethod
    def get_youtube_vos(data_folder, split, return_gts, return_masks):
        assert split in ['train', 'validation']
        dataset_folder = os.path.join(data_folder, 'YouTubeVOS')
        type_folder = 'JPEGImages' if return_gts else 'Annotations'
        samples_paths = sorted(os.listdir(os.path.join(dataset_folder, 'train', type_folder)))
        random.Random(0).shuffle(samples_paths)
        split_paths = samples_paths[:int(0.9 * len(samples_paths))] if split == 'train' else \
            samples_paths[int(0.9 * len(samples_paths)):]
        items_meta = {}
        for item_name in split_paths:
            item_gts_paths = None
            item_masks_paths = None
            if return_gts:
                item_gts_paths = sorted(
                    glob.glob(os.path.join(dataset_folder, 'train', 'JPEGImages', item_name, '*.jpg'))
                )
                item_gts_paths = [os.path.relpath(path, data_folder) for path in item_gts_paths]
            if return_masks:
                item_masks_paths = sorted(
                    glob.glob(os.path.join(dataset_folder, 'train', 'Annotations', item_name, '*.png'))
                )
                item_masks_paths = [os.path.relpath(path, data_folder) for path in item_masks_paths]
            items_meta[item_name] = (item_gts_paths, item_masks_paths)
        return items_meta
