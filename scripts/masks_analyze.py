import argparse
import progressbar
import os
import cv2
import random
import numpy as np

parser = argparse.ArgumentParser(description='Analyzes the percentage of masks inside an image')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--num-images', type=int, default=3, help='Number of images to analyze per folder')
parser.add_argument('--formats', nargs='+', default=['png'], help='Image formats to search in the path')

args = parser.parse_args()


def handle_folder(folder_path, args, bar, i):
    folder_content = os.listdir(folder_path)
    random.shuffle(folder_content)
    folder_percentages = []
    for folder_item in folder_content:
        file_path = os.path.join(folder_path, folder_item)
        if os.path.isfile(file_path) and os.path.splitext(folder_item)[-1].replace('.', '') in args.formats:
            image = (cv2.imread(os.path.join(folder_path, folder_item), cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
            folder_percentages.append(np.sum(image == 1) / (image.shape[0] * image.shape[1]))
        if len(folder_percentages) == args.num_images:
            break
    return np.mean(folder_percentages) if len(folder_percentages) > 0 else None


# Generate a list of sequences
folder_paths = sorted([root for root, _, _ in os.walk(args.data_path)])

# Create progress bar
bar = progressbar.ProgressBar(max_value=len(folder_paths))

# Walk through the folders of args.data_path
data_percentages = []
for i, folder_path in enumerate(folder_paths):
    folder_percentage = handle_folder(folder_path, args, bar, i)
    if folder_percentage is not None:
        data_percentages.append(folder_percentage)

# Print result
print('Mean mask occupation: {:.2f}%'.format(np.mean(data_percentages) * 100))
