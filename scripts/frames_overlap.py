import argparse
import utils

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--sequence', required=True, help='Path to the folder containing the frames')
parser.add_argument('--reference-frame', type=int, default=0, help='Index of the reference frame')
parser.add_argument('--alpha', type=int, default=50, help='Transparency level')
parser.add_argument('--frame-step', type=int, default=1, help='Space between consecutive frame')
parser.add_argument('--dest_folder', type=str, default='.', help='Path where the resulting video should be saved')
parser.add_argument('--filename', type=str, help='Force a name for the output file')
args = parser.parse_args()

# Create a FramesToVideo object
overlap_frames = utils.OverlapFrames(args.reference_frame, args.alpha, args.frame_step)

# Add the sequence to the object
overlap_frames.add_sequence_from_path(args.sequence)

# Save the video
overlap_frames.save(args.dest_folder, args.filename)