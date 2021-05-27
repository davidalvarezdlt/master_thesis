import argparse
import utils.paths
import thesis.dataset
import torch.utils.data
import matplotlib.pyplot as plt
import thesis_inpainting.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import os.path

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--experiments-path', required=True, help='Path where the experiments are stored')
parser.add_argument('--results-path', required=True, help='Path where the results should be stored')
parser.add_argument('--device', default='cpu', help='Device to use')
args = parser.parse_args()

# Prepare the dataset
data_meta = utils.paths.DatasetPaths.get_items('got-10k', args.data_path, 'validation')
dataset = thesis.dataset.MaskedSequenceDataset(
    gts_dataset=thesis.dataset.ContentProvider(args.data_path, data_meta, None), masks_dataset=None,
    gts_simulator=None, masks_simulator=None,
    image_size=(240, 480), frames_n=-1, frames_spacing=1, frames_randomize=False,
    dilatation_filter_size=(3, 3), dilatation_iterations=0,
    force_resize=True, keep_ratio=False
)

# Load the models
model_vgg = models.vgg_16.get_pretrained_model(args.device)
model_alignment = models.thesis_alignment.ThesisAlignmentModel(model_vgg).to(args.device)
model = models.thesis_inpainting.ThesisInpaintingVisible().to(args.device)

# Load aligner checkpoint
experiment_path = os.path.join(args.experiments_path, 'align_double')
checkpoint_path = os.path.join(experiment_path, 'checkpoints', '{}.checkpoint.pkl'.format(64))
with open(checkpoint_path, 'rb') as checkpoint_file:
    model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=args.device)['model'])

# Load inpainting checkpoint
experiment_path = os.path.join(args.experiments_path, 'inpaint_double')
checkpoint_path = os.path.join(experiment_path, 'checkpoints', '{}.checkpoint.pkl'.format(136))
with open(checkpoint_path, 'rb') as checkpoint_file:
    model.load_state_dict(torch.load(checkpoint_file, map_location=args.device)['model'])

# Iterate over the data
for it_data in dataset:
    (x, m), y, info = it_data
    x, m, y = x.to(args.device), m.to(args.device), y.to(args.device)
    y_inpainted = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ff(
        x, m, y, model_alignment, model
    )
    frames_to_video = utils.FramesToVideo(0, 10, None)
    frames_to_video.add_sequence(y_inpainted.cpu().numpy().transpose(1, 2, 3, 0) * 255)
    frames_to_video.save(args.results_path, info[0])
    print('Video saved')