import argparse
import cv2
import os.path
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualize samples from the dataset')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
args = parser.parse_args()

# Iterate over the .avi files inside data_path
for file_name in glob.glob(os.path.join(args.data_path, '*.avi')):

    # Check that the file is part of a hole sequence
    name_normalized = os.path.basename(file_name).replace('davis_', '')
    name_normalized = os.path.basename(file_name)
    # if '_hole' not in name_normalized:
    #     continue

    # Prepare the folder where the frames will be stored
    name_normalized = name_normalized.replace('_hole.avi', '')
    frames_folder = os.path.join(args.data_path, 'to_frames', name_normalized)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Read the video and save the frames inside the folder
    video, count = cv2.VideoCapture(file_name), 0
    success, image = video.read()
    while success:
        cv2.imwrite(os.path.join(frames_folder, "{}.png".format(str(count).zfill(5))), image)
        success, image = video.read()
        count += 1