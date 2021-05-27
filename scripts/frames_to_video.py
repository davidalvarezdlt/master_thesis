import argparse
import utils

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--sequences', required=True, nargs='+', help='Path to one or more folders containing the frames')
parser.add_argument('--margin', type=int, default=10, help='Margin added to each of the items')
parser.add_argument('--rate', type=int, default=25, help='Frame rate of the resulting video')
parser.add_argument('--resize', default=None, choices=['upscale', 'downscale'], help='Resizing strategy')
parser.add_argument('--dest_folder', type=str, default='.', help='Path where the resulting video should be saved')
parser.add_argument('--filename', type=str, help='Force a name for the output file')
args = parser.parse_args()

# Create a FramesToVideo object
frames_to_video = utils.FramesToVideo(args.margin, args.rate, args.resize)

# Accept a maximum grid of 2x2 videos
if len(args.frames) > 4:
    exit('The maximum number of frame folders that you can place in one video is 4.')

# Add all the sequences
for sequence_path in args.sequences:
    frames_to_video.add_sequence_from_path(sequence_path)

# Save the video
frames_to_video.save(args.dest_folder, args.filename)