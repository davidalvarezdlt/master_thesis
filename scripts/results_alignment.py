import argparse
import utils.paths
from thesis.dataset import ContentProvider, MaskedSequenceDataset
import models.thesis_alignment
import models.vgg_16
import random
import matplotlib.pyplot as plt
import utils.losses
import seaborn as sns
import pandas as pd
import numpy as np
import thesis_alignment.runner
import progressbar
import models.cpn_original
import thesis_cpn.runner
import utils.baselines

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--experiments-path', required=True, help='Path where the experiments are stored')
parser.add_argument('--n-samples', default=500, type=int, help='Number of samples')
parser.add_argument('--device', default='cpu', help='Device to use')
args = parser.parse_args()

# Load the models
vgg_model = models.vgg_16.get_pretrained_model(args.device)
cpn_model = thesis_cpn.runner.ThesisCPNRunner.init_model_with_state(
    models.cpn_original.CPNOriginal().to(args.device), args.device
)
ours_model = thesis_alignment.runner.ThesisAlignmentRunner.init_model_with_state(
    models.thesis_alignment.ThesisAlignmentModel(vgg_model).to(args.device),
    args.experiments_path, 'alignment_final', 125, args.device
)

# Create utils
sns.set()
loss_utils = utils.losses.LossesUtils(None, args.device)
losses = {}

# Iterate over the data sets and the displacements
for dataset_name in ['got-10k', 'davis-2017']:
    losses[dataset_name] = {}
    for s in range(1, 11):
        losses[dataset_name][s] = {'baseline': [], 'cpn': [], 'ours': []}

        # Create the dataset object
        gts_meta = utils.paths.DatasetPaths.get_items(dataset_name, args.data_path, 'validation', return_masks=False)
        masks_meta = utils.paths.DatasetPaths.get_items('youtube-vos', args.data_path, 'train', return_gts=False)
        gts_data = ContentProvider(args.data_path, gts_meta, None, 512)
        masks_data = ContentProvider(args.data_path, masks_meta, None)
        dataset = MaskedSequenceDataset(
            gts_dataset=gts_data,
            masks_dataset=masks_data,
            gts_simulator=None,
            masks_simulator=None,
            image_size=(256, 256),
            frames_n=2,
            frames_spacing=s,
            frames_randomize=False,
            dilatation_filter_size=(3, 3),
            dilatation_iterations=0,
            force_resize=True,
            keep_ratio=True
        )

        # Iterate over the samples
        bar = progressbar.ProgressBar(max_value=args.n_samples)
        for i, frame_index in enumerate(random.sample(range(len(dataset)), args.n_samples)):
            (x, m), y, info = it_data = dataset[frame_index]
            x, m, y = x.to(args.device), m.to(args.device), y.to(args.device)
            t, r_list = 1, [0]

            # Obtain alignments of the different networks
            try:
                x_aligned_baseline = utils.baselines.alignment(
                    x[:, t].cpu(), x[:, r_list].squeeze(1).cpu()
                ).to(args.device)

                x_aligned_cpn, _, _ = thesis_cpn.runner.ThesisCPNRunner.infer_alignment_step_propagate(
                    cpn_model, x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), x[:, r_list].unsqueeze(0),
                    m[:, r_list].unsqueeze(0)
                )

                x_aligned_ours, _, _ = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                    ours_model, x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), x[:, r_list].unsqueeze(0),
                    m[:, r_list].unsqueeze(0),
                )

                # Compute L1 losses outside the hole
                baseline_loss = loss_utils.masked_l1(x_aligned_baseline, x[:, t], mask=1 - m[:, t])
                cpn_loss = loss_utils.masked_l1(x_aligned_cpn[0, :, 0], x[:, t], mask=1 - m[:, t])
                ours_loss = loss_utils.masked_l1(x_aligned_ours[0, :, 0], x[:, t], mask=1 - m[:, t])

                # Append the losses
                losses[dataset_name][s]['baseline'].append(baseline_loss.cpu().item())
                losses[dataset_name][s]['cpn'].append(cpn_loss.cpu().item())
                losses[dataset_name][s]['ours'].append(ours_loss.cpu().item())
            except Exception:
                continue

            # Update bar
            bar.update(bar.value + 1)

    # Create the plot
    plt.figure()
    plt.title('DAVIS Extended Annotations Set' if dataset_name == 'davis-2017' else 'GOT-10k Validation Set')
    data = pd.DataFrame({
        'Frame distance (s)': list(range(1, 11)),
        'Baseline': [np.mean(losses[dataset_name][s]['baseline']) for s in range(1, 11)],
        'DFPN': [np.mean(losses[dataset_name][s]['ours']) for s in range(1, 11)],
        'State of the Art (CPA)': [np.mean(losses[dataset_name][s]['cpn']) for s in range(1, 11)],
    })
    ax = sns.lineplot(
        x='Frame distance (s)',
        y='Reconstruction Loss',
        hue='Model',
        data=pd.melt(data, ['Frame distance (s)'], var_name='Model', value_name='Reconstruction Loss')
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.savefig('results_alignment_{}.png'.format(dataset_name))
