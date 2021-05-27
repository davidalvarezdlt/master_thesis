import torch.optim
import models.thesis_alignment
import torch.nn.functional as F
import utils.losses
import utils.draws
import utils.flow
import thesis.runner
import models.vgg_16
import utils.correlation
import utils.transforms
import torch.utils.data
import matplotlib.pyplot as plt
import os.path


class ThesisAlignmentRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    utils_losses = None
    losses_items_ids = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)

    @staticmethod
    def init_model_with_state(model, experiments_path, experiment_name, epoch, device):
        experiment_path = os.path.join(experiments_path, experiment_name)
        checkpoint_path = os.path.join(experiment_path, 'checkpoints', '{}.checkpoint.pkl'.format(epoch))
        with open(checkpoint_path, 'rb') as checkpoint_file:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model'])
        return model.to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )
        self.utils_losses = utils.losses.LossesUtils(self.model_vgg, device)
        self.losses_items_ids = [
            'corr_loss', 'flow_16', 'flow_64', 'flow_256', 'alignment_recons_64', 'alignment_recons_256'
        ]
        super().init_others(device)

    def train_step(self, it_data, device):
        (x, m), y, info = it_data

        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
        t, r_list = self.get_indexes(x.size(2))

        # Propagate through the model
        corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, v_maps, \
        v_maps_gt = ThesisAlignmentRunner.train_step_propagate(self.model, x, m, y, flow_gt, flows_use, t, r_list)

        # Get both total loss and loss items
        loss, loss_items = ThesisAlignmentRunner.compute_loss(
            self.model_vgg, self.utils_losses, corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def test(self, epoch, device):
        self.model.eval()
        if epoch is not None:
            self.load_states(epoch, device)

        # Compute the losses on the test set
        self.test_losses(self.test_losses_handler, self.losses_items_ids, device)

        # Inpaint individual frames on the test set
        if self.counters['epoch'] % 1 == 0:
            self.test_frames(self.test_frames_handler, 'validation', device, include_y_hat=False,
                             include_y_hat_comp=False)
            self.test_frames(self.test_frames_handler, 'test', device, include_y_hat=False, include_y_hat_comp=False)

    def test_losses_handler(self, x, m, y, flows_use, flow_gt, t, r_list):
        with torch.no_grad():
            corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, \
            v_maps, v_maps_gt = ThesisAlignmentRunner.train_step_propagate(
                self.model, x, m, y, flow_gt, flows_use, t, r_list
            )
        return ThesisAlignmentRunner.compute_loss(
            self.model_vgg, self.utils_losses, corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list
        )

    def test_frames_handler(self, x, m, y, t, r_list):
        x_ref_aligned, v_ref_aligned, v_map = ThesisAlignmentRunner.infer_step_propagate(
            self.model, x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        return x_ref_aligned, v_ref_aligned, v_map, None, None

    @staticmethod
    def train_step_propagate(model, x, m, y, flow_gt, flows_use, t, r_list):
        corr, flow_16, flow_64, flow_256 = model(x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list])

        # Resize the data to multiple resolutions
        x_16, v_16, y_16 = utils.transforms.resize_set(x, 1 - m, y, 16)
        x_64, v_64, y_64 = utils.transforms.resize_set(x, 1 - m, y, 64)
        x_256, v_256, y_256 = x, 1 - m, y

        # Resize GT flows to multiple resolutions
        flow_16_gt = utils.flow.resize_flow(flow_gt[:, r_list], (16, 16))
        flow_64_gt = utils.flow.resize_flow(flow_gt[:, r_list], (64, 64))
        flow_256_gt = flow_gt[:, r_list]

        # Align the data in multiple resolutions with GT dense flow
        x_16_aligned_gt, v_16_aligned_gt = utils.flow.align_set(x_16[:, :, r_list], v_16[:, :, r_list], flow_16_gt)
        x_64_aligned_gt, v_64_aligned_gt = utils.flow.align_set(x_64[:, :, r_list], v_64[:, :, r_list], flow_64_gt)
        x_256_aligned_gt, v_256_aligned_gt = utils.flow.align_set(x_256[:, :, r_list], v_256[:, :, r_list], flow_256_gt)

        # Compute target v_maps
        v_map_64_gt, v_map_256_gt = torch.zeros_like(v_64_aligned_gt), torch.zeros_like(v_256_aligned_gt)
        v_map_64_gt[flows_use] = (
                v_64_aligned_gt[flows_use] - v_64[flows_use, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)
        v_map_256_gt[flows_use] = (
                v_256_aligned_gt[flows_use] - v_256[flows_use, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)

        # Align the data in multiple resolutions
        x_16_aligned, v_16_aligned = utils.flow.align_set(x_16[:, :, r_list], v_16[:, :, r_list], flow_16)
        x_64_aligned, v_64_aligned = utils.flow.align_set(x_64[:, :, r_list], v_64[:, :, r_list], flow_64)
        x_256_aligned, v_256_aligned = utils.flow.align_set(x_256[:, :, r_list], v_256[:, :, r_list], flow_256)

        # Compute predicted visual maps
        v_map_64 = (v_64_aligned - v_64[:, :, t].unsqueeze(2)).clamp(0, 1)
        v_map_256 = (v_256_aligned - v_256[:, :, t].unsqueeze(2)).clamp(0, 1)

        # Pack variables to return
        xs, vs, ys = (x_16, x_64, x_256), (v_16, v_64, v_256), (y_16, y_64, y_256)
        xs_aligned = (x_16_aligned, x_64_aligned, x_256_aligned)
        xs_aligned_gt = (x_16_aligned_gt, x_64_aligned_gt, x_256_aligned_gt)
        vs_aligned = (v_16_aligned, v_64_aligned, v_256_aligned)
        vs_aligned_gt = (v_16_aligned_gt, v_64_aligned_gt, v_256_aligned_gt)
        flows, flows_gt = (flow_16, flow_64, flow_256), (flow_16_gt, flow_64_gt, flow_256_gt)
        v_maps, v_maps_gt = (v_map_64, v_map_256), (v_map_64_gt, v_map_256_gt)

        # Return packed data
        return corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, \
               v_maps, v_maps_gt

    @staticmethod
    def infer_step_propagate(model, x_target, m_target, x_ref, m_ref):
        with torch.no_grad():
            *_, flow_256 = model(x_target, m_target, x_ref, m_ref)
        x_ref_aligned, v_ref_aligned = utils.flow.align_set(x_ref, (1 - m_ref), flow_256)
        v_map = (v_ref_aligned - (1 - m_target).unsqueeze(2)).clamp(0, 1)
        return x_ref_aligned, v_ref_aligned, v_map

    @staticmethod
    def compute_loss(model_vgg, utils_losses, corr, xs, vs, ys, xs_aligned, flows, flows_gt, flows_use, t, r_list):

        # Get the features of the frames from VGG
        b, c, f, h, w = ys[2].size()
        with torch.no_grad():
            if h == 256 and w == 256:
                y_vgg_input = ys[2].transpose(1, 2).reshape(b * f, c, h, w)
            else:
                y_vgg_input = F.interpolate(ys[2].transpose(1, 2).reshape(b * f, c, h, w), (256, 256), mode='bilinear')
            y_vgg_feats = model_vgg(y_vgg_input)
        y_vgg_feats = y_vgg_feats[3].reshape(b, f, -1, 16, 16).transpose(1, 2)

        # Compute L1 loss between correlation volumes
        corr_y = utils.correlation.compute_masked_4d_correlation(
            y_vgg_feats[:, :, t], None, y_vgg_feats[:, :, r_list], None
        )
        corr_loss = F.l1_loss(corr, corr_y)

        # Compute flow losses
        flow_loss_16 = utils_losses.masked_l1(flows[0], flows_gt[0], torch.ones_like(flows[0]), flows_use)
        flow_loss_64 = utils_losses.masked_l1(flows[1], flows_gt[1], torch.ones_like(flows[1]), flows_use)
        flow_loss_256 = utils_losses.masked_l1(flows[2], flows_gt[2], torch.ones_like(flows[2]), flows_use)

        # Compute out-of-frame regions from flows
        mask_out_64 = ((flows[1] < -1).float() + (flows[1] > 1).float()).sum(4).clamp(0, 1).unsqueeze(1)
        mask_out_256 = ((flows[2] < -1).float() + (flows[2] > 1).float()).sum(4).clamp(0, 1).unsqueeze(1)

        # Compute alignment reconstruction losses
        alignment_recons_64 = utils_losses.masked_l1(
            xs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[1],
            vs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1) * (1 - mask_out_64),
            reduction='sum'
        )
        alignment_recons_256 = utils_losses.masked_l1(
            xs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[2],
            vs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1) * (1 - mask_out_256),
            reduction='sum'
        )

        # Compute sum of losses and return them
        total_loss = corr_loss + flow_loss_16 + flow_loss_64 + flow_loss_256
        total_loss += alignment_recons_64 + alignment_recons_256
        return total_loss, [
            corr_loss, flow_loss_16, flow_loss_64, flow_loss_256, alignment_recons_64, alignment_recons_256
        ]
