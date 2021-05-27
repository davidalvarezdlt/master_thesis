import os.path
import time
import cv2
import numpy as np
import re
from PIL import Image


class OverlapFrames:

    def __init__(self, reference_frame_index, alpha, frame_step=1):
        self.reference_index = reference_frame_index
        self.alpha = alpha
        self.frame_step = frame_step
        self.sequence = None

    def add_sequence(self, sequence):
        self.sequence = sequence

    def add_sequence_from_path(self, sequence_path):
        frames = []
        for frame_path in sorted(os.listdir(sequence_path), key=lambda x: int(re.search(r'\d+', x).group())):
            frame_bgr = cv2.imread(os.path.join(sequence_path, frame_path))
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.sequence = np.stack(frames, axis=0)

    def _get_image_path(self, dest_folder, filename):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        if filename:
            return os.path.join(dest_folder, filename + '.png')
        else:
            while True:
                video_name = os.path.join(dest_folder, time.strftime("%H%M%S") + '.png')
                if not os.path.exists(video_name):
                    break
            return video_name

    def _validate_parameters(self):
        assert self.sequence is not None
        assert self.reference_index < self.sequence.shape[0]

    def save(self, dest_folder, filename=None):

        # Validate that the parameters are valid
        self._validate_parameters()

        # Compose the image
        overlapped_image = Image.fromarray(self.sequence[self.reference_index], mode='RGB').convert('RGBA')
        for i in range(0, self.sequence.shape[0], self.frame_step):
            if i == self.reference_index:
                continue
            aux_frame = Image.fromarray(self.sequence[i]).convert('RGBA')
            aux_frame.putalpha(50)
            overlapped_image = Image.alpha_composite(overlapped_image, aux_frame)

        # Save the image
        overlapped_image.save(self._get_image_path(dest_folder, filename))


class FramesToVideo:

    def __init__(self, margin, rate, resize):
        self.margin = margin
        self.rate = rate
        self.resize = resize
        self.sequences = []

    def add_sequence(self, sequence):
        """Add a sequence of frames to the final video.

        Args:
            frames (np.ndarray): array of uint8 values [0-255] of size (F,H,W,C) containing the frames.
        """
        assert len(self.sequences) < 4
        self.sequences.append(sequence)

    def add_sequence_from_path(self, sequence_path):
        frames = []
        for frame_path in sorted(os.listdir(sequence_path), key=lambda x: int(re.search(r'\d+', x).group())):
            frame_bgr = cv2.imread(os.path.join(sequence_path, frame_path))
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.sequences.append(np.stack(frames, axis=0))

    def _get_sequence_size(self):
        frame_size = (self.sequences[0].shape[1], self.sequences[0].shape[2])
        for sequence in self.sequences:
            if frame_size != (sequence.shape[1], sequence.shape[2]):
                if self.resize is None:
                    exit('All the videos must be of the same size')
                elif self.resize == 'upscale' and sequence.shape[1] > frame_size[0] or self.resize == 'downscale' and \
                        sequence.shape[1] < frame_size[0]:
                    frame_size = (sequence.shape[1], sequence.shape[2])
        return frame_size

    def _get_video_path(self, dest_folder, filename):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        if filename:
            return os.path.join(dest_folder, filename + '.avi')
        else:
            while True:
                video_name = os.path.join(dest_folder, time.strftime("%H%M%S") + '.avi')
                if not os.path.exists(video_name):
                    break
            return video_name

    def save(self, dest_folder, filename=None):

        # Get the size of each sequence in the video. Add a margin to each sequence.
        sequence_size = self._get_sequence_size()
        sequence_padded_size = (sequence_size[0] + 2 * self.margin, sequence_size[1] + 2 * self.margin)

        # Get the size of the complete video
        video_size = (
            sequence_padded_size[0] * (2 if len(self.sequences) > 2 else 1),
            sequence_padded_size[1] * (2 if len(self.sequences) > 1 else 1),
            3
        )

        # Get the number of frames of the video, which is the smallest number of frames in one of the sequences.
        num_frames = min([sequence.shape[0] for sequence in self.sequences])

        # Get the path where the video has to be stored
        video_path = self._get_video_path(dest_folder, filename)

        # Create the video
        final_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), self.rate, (video_size[1],
                                                                                               video_size[0]))
        # Iterate over all the frames and sequences
        for i in range(num_frames):
            it_frame = np.zeros(video_size, dtype=np.uint8)
            for j in range(len(self.sequences)):
                video_frame = self.sequences[j][i]
                if (video_frame.shape[0], video_frame.shape[1]) != sequence_size:
                    video_frame = cv2.resize(
                        video_frame, dsize=(sequence_size[1], sequence_size[0]), interpolation=cv2.INTER_LINEAR
                    )
                video_frame = np.pad(video_frame, ((self.margin, self.margin), (self.margin, self.margin), (0, 0)))
                it_frame[(j // 2) * sequence_padded_size[0]:(j // 2 + 1) * sequence_padded_size[0],
                (j % 2) * sequence_padded_size[1]:(j % 2 + 1) * sequence_padded_size[1], :] = video_frame
            final_video.write(cv2.cvtColor(it_frame, cv2.COLOR_RGB2BGR))
        final_video.release()
