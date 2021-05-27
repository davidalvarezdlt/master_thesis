import argparse
import utils.paths
from thesis.dataset import ContentProvider, MaskedSequenceDataset
import models.thesis_alignment
import models.vgg_16
import random
import utils.losses
import numpy as np
import thesis_dfpn.runner
import thesis_inpainting.runner
import progressbar
import models.thesis_inpainting
import models.cpn_original
import thesis_cpn.runner
import utils.measures
import thesis.runner
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--experiments-path', required=True, help='Path where the experiments are stored')
parser.add_argument('--n-frames', default=11, type=int, help='Number of frames')
parser.add_argument('--n-samples', default=50, type=int, help='Number of samples')
parser.add_argument('--device', default='cpu', help='Device to use')
args = parser.parse_args()

# Load the models
vgg_model = models.vgg_16.get_pretrained_model(args.device)
cpn_model = thesis_cpn.runner.ThesisCPNRunner.init_model_with_state(
    models.cpn_original.CPNOriginal().to(args.device), args.device
)
dfpn_model = thesis_dfpn.runner.ThesisAlignmentRunner.init_model_with_state(
    models.thesis_alignment.ThesisAlignmentModel(vgg_model), args.experiments_path, 'alignment_final', 125, args.device
)
cpn_chn_model = thesis_inpainting.runner.ThesisInpaintingRunner.init_model_with_state(
    models.thesis_inpainting.ThesisInpaintingVisible(), args.experiments_path, 'inpainting_final_cpn', 88, args.device
)
dfpn_chn_model = thesis_inpainting.runner.ThesisInpaintingRunner.init_model_with_state(
    models.thesis_inpainting.ThesisInpaintingVisible(), args.experiments_path, 'inpainting_final_dfpn', 88, args.device
)

# Iterate over the data sets and the displacements
loss_utils = utils.losses.LossesUtils(None, args.device)
measures_utils = utils.measures.UtilsMeasures()
measures_utils.init_lpips(args.device)
for dataset_name in ['davis-2017']:

    # Create lists to store samples results
    l1_baseline_cpn_ff, l1_baseline_cpn_ip, l1_baseline_cpn_cp = [], [], []
    l1_baseline_dfpn_ff, l1_baseline_dfpn_ip, l1_baseline_dfpn_cp = [], [], []
    l1_cpn_chn_ff, l1_cpn_chn_ip, l1_cpn_chn_cp = [], [], []
    l1_dfpn_chn_ff, l1_dfpn_chn_ip, l1_dfpn_chn_cp = [], [], []
    l1_cpn = []

    psnr_baseline_cpn_ff, psnr_baseline_cpn_ip, psnr_baseline_cpn_cp = [], [], []
    psnr_baseline_dfpn_ff, psnr_baseline_dfpn_ip, psnr_baseline_dfpn_cp = [], [], []
    psnr_cpn_chn_ff, psnr_cpn_chn_ip, psnr_cpn_chn_cp = [], [], []
    psnr_dfpn_chn_ff, psnr_dfpn_chn_ip, psnr_dfpn_chn_cp = [], [], []
    psnr_cpn = []

    ssim_baseline_cpn_ff, ssim_baseline_cpn_ip, ssim_baseline_cpn_cp = [], [], []
    ssim_baseline_dfpn_ff, ssim_baseline_dfpn_ip, ssim_baseline_dfpn_cp = [], [], []
    ssim_cpn_chn_ff, ssim_cpn_chn_ip, ssim_cpn_chn_cp = [], [], []
    ssim_dfpn_chn_ff, ssim_dfpn_chn_ip, ssim_dfpn_chn_cp = [], [], []
    ssim_cpn = []

    lpips_baseline_cpn_ff, lpips_baseline_cpn_ip, lpips_baseline_cpn_cp = [], [], []
    lpips_baseline_dfpn_ff, lpips_baseline_dfpn_ip, lpips_baseline_dfpn_cp = [], [], []
    lpips_cpn_chn_ff, lpips_cpn_chn_ip, lpips_cpn_chn_cp = [], [], []
    lpips_dfpn_chn_ff, lpips_dfpn_chn_ip, lpips_dfpn_chn_cp = [], [], []
    lpips_cpn = []

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
        frames_n=args.n_frames,
        frames_spacing=1,
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

        # Check that the mask is not empty
        if m.sum() == 0:
            print('Invalid mask sample')
            continue

        # Algorithm 1: FF
        y_inpainted_baseline_cpn_ff = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ff(
            x.clone(), m.clone(), cpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_baseline_dfpn_ff = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ff(
            x.clone(), m.clone(), dfpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_cpn_chn_ff = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ff(
            x.clone(), m.clone(), cpn_model, cpn_chn_model
        )
        y_inpainted_dfpn_chn_ff = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ff(
            x.clone(), m.clone(), cpn_model, dfpn_chn_model
        )

        # Algorithm 2: IP
        y_inpainted_baseline_cpn_ip = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ip(
            x.clone(), m.clone(), cpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_baseline_dfpn_ip = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ip(
            x.clone(), m.clone(), dfpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_cpn_chn_ip = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ip(
            x.clone(), m.clone(), cpn_model, cpn_chn_model
        )
        y_inpainted_dfpn_chn_ip = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_ip(
            x.clone(), m.clone(), cpn_model, dfpn_chn_model
        )

        # Algorithm 3: CP
        y_inpainted_baseline_cpn_cp = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_cp(
            x.clone(), m.clone(), cpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_baseline_dfpn_cp = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_cp(
            x.clone(), m.clone(), dfpn_model, thesis.runner.ThesisRunner.inpainting_hard_copy
        )
        y_inpainted_cpn_chn_cp = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_cp(
            x.clone(), m.clone(), cpn_model, cpn_chn_model
        )
        y_inpainted_dfpn_chn_cp = thesis_inpainting.runner.ThesisInpaintingRunner.inpainting_algorithm_cp(
            x.clone(), m.clone(), cpn_model, dfpn_chn_model
        )

        # CPN
        y_inpainted_cpn = thesis_cpn.runner.ThesisCPNRunner.inpainting_algorithm(x.clone(), m.clone(), None, cpn_model)

        # Quality Measure: Hole L1
        l1_baseline_cpn_ff.append(loss_utils.masked_l1(y_inpainted_baseline_cpn_ff, y, mask=m, reduction='sum').item())
        l1_baseline_dfpn_ff.append(loss_utils.masked_l1(y_inpainted_baseline_dfpn_ff, y, mask=m, reduction='sum').item())
        l1_cpn_chn_ff.append(loss_utils.masked_l1(y_inpainted_cpn_chn_ff, y, mask=m, reduction='sum').item())
        l1_dfpn_chn_ff.append(loss_utils.masked_l1(y_inpainted_dfpn_chn_ff, y, mask=m, reduction='sum').item())
        l1_baseline_cpn_ip.append(loss_utils.masked_l1(y_inpainted_baseline_cpn_ip, y, mask=m, reduction='sum').item())
        l1_baseline_dfpn_ip.append(loss_utils.masked_l1(y_inpainted_baseline_dfpn_ip, y, mask=m, reduction='sum').item())
        l1_cpn_chn_ip.append(loss_utils.masked_l1(y_inpainted_cpn_chn_ip, y, mask=m, reduction='sum').item())
        l1_dfpn_chn_ip.append(loss_utils.masked_l1(y_inpainted_dfpn_chn_ip, y, mask=m, reduction='sum').item())
        l1_baseline_cpn_cp.append(loss_utils.masked_l1(y_inpainted_baseline_cpn_cp, y, mask=m, reduction='sum').item())
        l1_baseline_dfpn_cp.append(loss_utils.masked_l1(y_inpainted_baseline_dfpn_cp, y, mask=m, reduction='sum').item())
        l1_cpn_chn_cp.append(loss_utils.masked_l1(y_inpainted_cpn_chn_cp, y, mask=m, reduction='sum').item())
        l1_dfpn_chn_cp.append(loss_utils.masked_l1(y_inpainted_dfpn_chn_cp, y, mask=m, reduction='sum').item())
        l1_cpn.append(loss_utils.masked_l1(y_inpainted_cpn, y, mask=m, reduction='sum').item())

        # Quality Measure: PSNR
        psnr_baseline_cpn_ff.append(measures_utils.psnr(y_inpainted_baseline_cpn_ff.cpu(), y.cpu()))
        psnr_baseline_dfpn_ff.append(measures_utils.psnr(y_inpainted_baseline_dfpn_ff.cpu(), y.cpu()))
        psnr_cpn_chn_ff.append(measures_utils.psnr(y_inpainted_cpn_chn_ff.cpu(), y.cpu()))
        psnr_dfpn_chn_ff.append(measures_utils.psnr(y_inpainted_dfpn_chn_ff.cpu(), y.cpu()))
        psnr_baseline_cpn_ip.append(measures_utils.psnr(y_inpainted_baseline_cpn_ip.cpu(), y.cpu()))
        psnr_baseline_dfpn_ip.append(measures_utils.psnr(y_inpainted_baseline_dfpn_ip.cpu(), y.cpu()))
        psnr_cpn_chn_ip.append(measures_utils.psnr(y_inpainted_cpn_chn_ip.cpu(), y.cpu()))
        psnr_dfpn_chn_ip.append(measures_utils.psnr(y_inpainted_dfpn_chn_ip.cpu(), y.cpu()))
        psnr_baseline_cpn_cp.append(measures_utils.psnr(y_inpainted_baseline_cpn_cp.cpu(), y.cpu()))
        psnr_baseline_dfpn_cp.append(measures_utils.psnr(y_inpainted_baseline_dfpn_cp.cpu(), y.cpu()))
        psnr_cpn_chn_cp.append(measures_utils.psnr(y_inpainted_cpn_chn_cp.cpu(), y.cpu()))
        psnr_dfpn_chn_cp.append(measures_utils.psnr(y_inpainted_dfpn_chn_cp.cpu(), y.cpu()))
        psnr_cpn.append(measures_utils.psnr(y_inpainted_cpn.cpu(), y.cpu()))

        # Quality Measure: SSIM
        ssim_baseline_cpn_ff.append(measures_utils.ssim(y_inpainted_baseline_cpn_ff.cpu(), y.cpu()))
        ssim_baseline_dfpn_ff.append(measures_utils.ssim(y_inpainted_baseline_dfpn_ff.cpu(), y.cpu()))
        ssim_cpn_chn_ff.append(measures_utils.ssim(y_inpainted_cpn_chn_ff.cpu(), y.cpu()))
        ssim_dfpn_chn_ff.append(measures_utils.ssim(y_inpainted_dfpn_chn_ff.cpu(), y.cpu()))
        ssim_baseline_cpn_ip.append(measures_utils.ssim(y_inpainted_baseline_cpn_ip.cpu(), y.cpu()))
        ssim_baseline_dfpn_ip.append(measures_utils.ssim(y_inpainted_baseline_dfpn_ip.cpu(), y.cpu()))
        ssim_cpn_chn_ip.append(measures_utils.ssim(y_inpainted_cpn_chn_ip.cpu(), y.cpu()))
        ssim_dfpn_chn_ip.append(measures_utils.ssim(y_inpainted_dfpn_chn_ip.cpu(), y.cpu()))
        ssim_baseline_cpn_cp.append(measures_utils.ssim(y_inpainted_baseline_cpn_cp.cpu(), y.cpu()))
        ssim_baseline_dfpn_cp.append(measures_utils.ssim(y_inpainted_baseline_dfpn_cp.cpu(), y.cpu()))
        ssim_cpn_chn_cp.append(measures_utils.ssim(y_inpainted_cpn_chn_cp.cpu(), y.cpu()))
        ssim_dfpn_chn_cp.append(measures_utils.ssim(y_inpainted_dfpn_chn_cp.cpu(), y.cpu()))
        ssim_cpn.append(measures_utils.ssim(y_inpainted_cpn.cpu(), y.cpu()))

        # Quality Measure: LPIPS
        lpips_baseline_cpn_ff.append(measures_utils.lpips(y_inpainted_baseline_cpn_ff, y))
        lpips_baseline_dfpn_ff.append(measures_utils.lpips(y_inpainted_baseline_dfpn_ff, y))
        lpips_cpn_chn_ff.append(measures_utils.lpips(y_inpainted_cpn_chn_ff, y))
        lpips_dfpn_chn_ff.append(measures_utils.lpips(y_inpainted_dfpn_chn_ff, y))
        lpips_baseline_cpn_ip.append(measures_utils.lpips(y_inpainted_baseline_cpn_ip, y))
        lpips_baseline_dfpn_ip.append(measures_utils.lpips(y_inpainted_baseline_dfpn_ip, y))
        lpips_cpn_chn_ip.append(measures_utils.lpips(y_inpainted_cpn_chn_ip, y))
        lpips_dfpn_chn_ip.append(measures_utils.lpips(y_inpainted_dfpn_chn_ip, y))
        lpips_baseline_cpn_cp.append(measures_utils.lpips(y_inpainted_baseline_cpn_cp, y))
        lpips_baseline_dfpn_cp.append(measures_utils.lpips(y_inpainted_baseline_dfpn_cp, y))
        lpips_cpn_chn_cp.append(measures_utils.lpips(y_inpainted_cpn_chn_cp, y))
        lpips_dfpn_chn_cp.append(measures_utils.lpips(y_inpainted_dfpn_chn_cp, y))
        lpips_cpn.append(measures_utils.lpips(y_inpainted_cpn, y))

        # Update the bar
        bar.update(bar.value + 1)

    # Plot the measures in a table
    t = PrettyTable(['Model', 'L1 Hole', 'PSNR', 'SSIM', 'LPIPS'])

    # Algorithm 1: FF
    t.add_row([
        'CPN + Baseline + FF', np.mean(l1_baseline_cpn_ff), np.mean(psnr_baseline_cpn_ff), np.mean(ssim_baseline_cpn_ff),
        np.mean(lpips_baseline_cpn_ff)
    ])
    t.add_row([
        'DFPN + Baseline + FF', np.mean(l1_baseline_dfpn_ff), np.mean(psnr_baseline_dfpn_ff), np.mean(ssim_baseline_dfpn_ff),
        np.mean(lpips_baseline_dfpn_ff)
    ])
    t.add_row([
        'Ours (CPN + CHN + FF)', np.mean(l1_cpn_chn_ff), np.mean(psnr_cpn_chn_ff), np.mean(ssim_cpn_chn_ff),
        np.mean(lpips_cpn_chn_ff)
    ])
    t.add_row([
        'Ours (DFPN + CHN + FF)', np.mean(l1_dfpn_chn_ff), np.mean(psnr_dfpn_chn_ff), np.mean(ssim_dfpn_chn_ff),
        np.mean(lpips_dfpn_chn_ff)
    ])

    # Algorithm 2: IP
    t.add_row([
        'CPN + Baseline + IP', np.mean(l1_baseline_cpn_ip), np.mean(psnr_baseline_cpn_ip),
        np.mean(ssim_baseline_cpn_ip), np.mean(lpips_baseline_cpn_ip)
    ])
    t.add_row([
        'DFPN + Baseline + IP', np.mean(l1_baseline_dfpn_ip), np.mean(psnr_baseline_dfpn_ip),
        np.mean(ssim_baseline_dfpn_ip), np.mean(lpips_baseline_dfpn_ip)
    ])
    t.add_row([
        'Ours (CPN + CHN + IP)', np.mean(l1_cpn_chn_ip), np.mean(psnr_cpn_chn_ip), np.mean(ssim_cpn_chn_ip),
        np.mean(lpips_cpn_chn_ip)
    ])
    t.add_row([
        'Ours (DFPN + CHN + IP)', np.mean(l1_dfpn_chn_ip), np.mean(psnr_dfpn_chn_ip), np.mean(ssim_dfpn_chn_ip),
        np.mean(lpips_dfpn_chn_ip)
    ])

    # Algorithm 3: CP
    t.add_row([
        'CPN + Baseline + CP', np.mean(l1_baseline_cpn_cp), np.mean(psnr_baseline_cpn_cp),
        np.mean(ssim_baseline_cpn_cp), np.mean(lpips_baseline_cpn_cp)
    ])
    t.add_row([
        'DFPN + Baseline + CP', np.mean(l1_baseline_dfpn_cp), np.mean(psnr_baseline_dfpn_cp),
        np.mean(ssim_baseline_dfpn_cp), np.mean(lpips_baseline_dfpn_cp)
    ])
    t.add_row([
        'Ours (CPN + CHN + CP)', np.mean(l1_cpn_chn_cp), np.mean(psnr_cpn_chn_cp), np.mean(ssim_cpn_chn_cp),
        np.mean(lpips_cpn_chn_cp)
    ])
    t.add_row([
        'Ours (DFPN + CHN + CP)', np.mean(l1_dfpn_chn_cp), np.mean(psnr_dfpn_chn_cp), np.mean(ssim_dfpn_chn_cp),
        np.mean(lpips_dfpn_chn_cp)
    ])

    # CPN
    t.add_row(['CPN', np.mean(l1_cpn), np.mean(psnr_cpn), np.mean(ssim_cpn), np.mean(lpips_cpn)])

    # Print results table
    print('Dataset: {}'.format(dataset_name))
    print(t)