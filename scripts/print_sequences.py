import os.path
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define folder where the sequences are
sequences_folders = {
    'sota': '/Users/DavidAlvarezDLT/Desktop/CPN',
    'cpn': '/Users/DavidAlvarezDLT/Desktop/cpa_chn/epoch-88',
    'dfpn': '/Users/DavidAlvarezDLT/Desktop/dfpn_chn/epoch-88'
}

# Define the folder where the results are stored
results_folder = '/Users/DavidAlvarezDLT/Desktop/seq_results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get all the videos of the first sequences_folder
videos_list = glob.glob(os.path.join(sequences_folders['sota'], '*.avi'))


# Define a function to get middle frames of a video
def get_middle_frames(video_path):
    cap, success = cv2.VideoCapture(video_path), True
    frames = []
    while success:
        success, image = cap.read()
        if success:
            frames.append(image)
    return frames[len(frames) // 2 - 2:len(frames) // 2 + 3]


# Iterate over the set of videos_list
for video_item in videos_list:
    video_name = video_item.replace(sequences_folders['sota'], '')[1:-4]
    video_rows, save_video = [], True
    for _, sequence_item in sequences_folders.items():
        if not os.path.exists(os.path.join(sequence_item, video_name + '.avi')):
            save_video = False
            continue
        middle_frames = get_middle_frames(os.path.join(sequence_item, video_name + '.avi'))
        video_rows.append(np.concatenate(middle_frames, axis=1))
    if save_video:
        video = np.concatenate(video_rows, axis=0)
        cv2.imwrite(os.path.join(results_folder, 'results_seq_' + video_name + '.png'), video)