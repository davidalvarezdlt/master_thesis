import argparse
import utils.paths
import os.path
import numpy as np
import cv2
import jpeg4py as jpeg
import random

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
args = parser.parse_args()

# Define list of items
datasets_list = ['got-10k']
splits_list = ['train', 'val', 'test']


# Create a function to compute the average mask size of a sequence
def compute_avg_mask_size(item, n_frames=5):
    background_size, mask_size = jpeg.JPEG(os.path.join(args.data_path, item[0][0])).decode().shape, []
    for i in random.sample(list(range(len(item[0]))), min(n_frames, len(item[0]))):
        mask = cv2.imread(os.path.join(args.data_path, item[1][i]), cv2.IMREAD_GRAYSCALE) / 255
        mask_size.append(np.sum(mask) / (background_size[0] * background_size[1]))
    return np.mean(mask_size) * 100


# Prepare the dataset
for dataset_name in datasets_list:
    n_sequences = 0
    avg_samples_per_video = []
    avg_mask_size = []
    for split in splits_list:
        if dataset_name == 'davis-2017' and split not in ['train']:
            continue
        data_meta = utils.paths.DatasetPaths.get_items(dataset_name, args.data_path, split, return_masks=False)
        n_sequences += len(data_meta)
        avg_samples_per_video += [len(data_item[1][0]) for data_item in data_meta.items()]
        if dataset_name in ['davis-2017', 'youtube-vos'] and split in ['train']:
            avg_mask_size += [compute_avg_mask_size(data_item[1]) for data_item in data_meta.items()]
    samples_per_video = np.mean(avg_samples_per_video)
    avg_mask_size = np.mean(avg_mask_size)
    print(
        'Number of sequences: {} | Avg. Frames per Video: {} | Avg. Mask Size: {}'.format(
            n_sequences, samples_per_video, avg_mask_size
        )
    )
