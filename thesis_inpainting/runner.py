import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import models.thesis_inpainting
import torch
import utils.losses
import thesis_dfpn.runner
import utils.flow
import utils.draws
import matplotlib.pyplot as plt
import thesis_cpn.runner
import models.cpn_original
import os.path
import progressbar


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model = models.thesis_inpainting.ThesisInpaintingVisible().to(device)
        if self.experiment.configuration.get('model', 'alignment_network') == 'cpn':
            self.model_alignment = thesis_cpn.runner.ThesisCPNRunner.init_model_with_state(
                models.cpn_original.CPNOriginal().to(device), device
            )
        else:
            self.model_alignment = thesis_dfpn.runner.ThesisAlignmentRunner.init_model_with_state(
                models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device),
                os.path.dirname(self.experiment.paths['experiment']),
                self.experiment.configuration.get('model', 'alignment_experiment_name'),
                self.experiment.configuration.get('model', 'alignment_experiment_epoch'),
                device
            )

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
        self.losses_items_ids = ['loss_nh', 'loss_vh', 'loss_nvh', 'loss_perceptual', 'loss_grad']
        super().init_others(device)

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
        t, r_list = self.get_indexes(x.size(2))

        # Compute t and r_list
        x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.train_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    @staticmethod
    def train_step_propagate(model_alignment, model, x_target, m_target, x_ref, m_ref):
        with torch.no_grad():
            if isinstance(model_alignment, models.thesis_alignment.ThesisAlignmentModel):
                x_ref_aligned, v_ref_aligned, v_map = thesis_dfpn.runner.ThesisAlignmentRunner.infer_step_propagate(
                    model_alignment, x_target, m_target, x_ref, m_ref
                )
            else:
                x_ref_aligned, v_ref_aligned, v_map = thesis_cpn.runner.ThesisCPNRunner.infer_alignment_step_propagate(
                    model_alignment, x_target, m_target, x_ref, m_ref
                )
        y_hat, y_hat_comp = model(x_target, 1 - m_target, x_ref_aligned, v_ref_aligned, v_map)
        return x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp

    @staticmethod
    def infer_step_propagate(model_alignment, model, x_target, m_target, x_ref, m_ref):
        with torch.no_grad():
            if isinstance(model_alignment, models.thesis_alignment.ThesisAlignmentModel):
                x_ref_aligned, v_ref_aligned, v_map = thesis_dfpn.runner.ThesisAlignmentRunner.infer_step_propagate(
                    model_alignment, x_target, m_target, x_ref, m_ref
                )
            else:
                x_ref_aligned, v_ref_aligned, v_map = thesis_cpn.runner.ThesisCPNRunner.infer_alignment_step_propagate(
                    model_alignment, x_target, m_target, x_ref, m_ref
                )
            y_hat, y_hat_comp = model(x_target, 1 - m_target, x_ref_aligned, v_ref_aligned, v_map)
        return x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp

    @staticmethod
    def compute_loss(utils_losses, y_target, v_target, y_hat, y_hat_comp, v_map):
        b, c, h, w = y_target.size()
        target_img = y_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        nh_mask = v_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        vh_mask = v_map
        nvh_mask = (1 - nh_mask) - vh_mask
        loss_nh = utils_losses.masked_l1(y_hat, target_img, nh_mask, reduction='sum', weight=0.50)
        loss_vh = utils_losses.masked_l1(y_hat, target_img, vh_mask, reduction='sum', weight=2)
        loss_nvh = utils_losses.masked_l1(y_hat_comp, target_img, nvh_mask, reduction='sum', weight=1)
        loss_perceptual, *_ = utils_losses.perceptual(
            y_hat.transpose(1, 2).reshape(-1, c, h, w), target_img.transpose(1, 2).reshape(-1, c, h, w), weight=0.50
        )
        loss_grad = utils_losses.grad(y_hat.squeeze(2), target_img.squeeze(2), reduction='mean', weight=1)
        loss = loss_nh + loss_vh + loss_nvh + loss_perceptual + loss_grad
        return loss, [loss_nh, loss_vh, loss_nvh, loss_perceptual, loss_grad]

    def test(self, epoch, device):
        self.model.eval()

        # If epoch != 0, loadit
        if epoch is not None:
            self.load_states(epoch, device)

        # # Compute the losses on the test set
        # self.test_losses(self.test_losses_handler, self.losses_items_ids, device)
        #
        # # Inpaint individual frames on the test set
        # if self.counters['epoch'] % 5 == 0:
        #     self.test_frames(self.test_frames_handler, 'validation', device)
        #     self.test_frames(self.test_frames_handler, 'test', device)

        # Inpaint test sequences every 10 epochs
        if epoch is not None or self.counters['epoch'] % 50 == 0:
            # self.test_sequence(
            #     self.inpainting_algorithm_ff, 'algorithm_ff', self.model_alignment, self.model, device
            # )
            # self.test_sequence(
            #     self.inpainting_algorithm_ip, 'algorithm_ip', self.model_alignment, self.model, device
            # )
            self.test_sequence(
                self.inpainting_algorithm_cp, 'algorithm_cp', self.model_alignment, self.model, device
            )

    def test_losses_handler(self, x, m, y, flows_use, flow_gt, t, r_list):
        x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        return ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )

    def test_frames_handler(self, x, m, y, t, r_list):
        return ThesisInpaintingRunner.infer_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

    @staticmethod
    def inpainting_algorithm_ff(x, m, model_alignment, model, s=1, D=20, e=1):
        print('Inpainting sequence...')
        fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        y_inpainted = torch.zeros_like(x)
        for t in range(x.size(1)):
            x_target, m_target, y_hat_comp = x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), None
            t_candidates = ThesisInpaintingRunner.inpainting_algorithm_ff_indexes(t, x.size(1), s=s, D=D)
            while (len(t_candidates) > 0 and torch.sum(m_target) * 100 / m_target.numel() > e) or y_hat_comp is None:
                r_list = [t_candidates.pop(0)]
                _, _, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
                    model_alignment, model, x_target, m_target, x[:, r_list].unsqueeze(0),
                    m[:, r_list].unsqueeze(0)
                )
                m_target = m_target - v_map[:, :, 0]
                x_target = (1 - m_target) * y_hat_comp[:, :, 0] + m_target.repeat(1, 3, 1, 1) * fill_color
            y_inpainted[:, t] = y_hat_comp[:, :, 0]
        return y_inpainted

    @staticmethod
    def inpainting_algorithm_ff_indexes(t, max_t, s, D):
        ref_candidates = list(range(max_t))
        ref_candidates.pop(t)
        ref_candidates_dist = list(map(lambda x: abs(x - t), ref_candidates))
        ref_candidates_sorted = [r[1] for r in sorted(zip(ref_candidates_dist, ref_candidates))]
        return list(
            filter(lambda x: abs(x - t) <= D and abs(x - t) % s == 0, ref_candidates_sorted)
        )

    @staticmethod
    def inpainting_algorithm_ip(x, m, model_alignment, model, s=1, D=20, e=1):
        print('Inpainting sequence...')
        fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        y_inpainted, m_inpainted = x.unsqueeze(0), m.unsqueeze(0)
        t_list = sorted(list(range(x.size(1))), key=lambda xi: abs(xi - x.size(1) // 2))
        for t in t_list:
            t_candidates = ThesisInpaintingRunner.inpainting_algorithm_ip_indexes(t, t_list, s, D)
            y_hat_comp = None
            while (len(t_candidates) > 0 and torch.sum(m_inpainted[:, :, t]) * 100 / m_inpainted[:, :, t].numel() > e) \
                    or y_hat_comp is None:
                r_list = [t_candidates.pop(0)]
                _, _, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
                    model_alignment, model, y_inpainted[:, :, t], m_inpainted[:, :, t], y_inpainted[:, :, r_list],
                    m_inpainted[:, :, r_list]
                )
                m_inpainted[:, :, t] = m_inpainted[:, :, t] - v_map[:, :, 0]
                y_inpainted[:, :, t] = (1 - m_inpainted[:, :, t]) * y_hat_comp[:, :, 0] + \
                                       m_inpainted[:, :, t].repeat(1, 3, 1, 1) * fill_color
            m_inpainted[:, :, t] = 0
            y_inpainted[:, :, t] = y_hat_comp[:, :, 0]
        return y_inpainted[0]

    @staticmethod
    def inpainting_algorithm_ip_indexes(t, t_list, s, D):
        t_list_inpainted = list(reversed(t_list[:t_list.index(t)]))
        t_list_ff = ThesisInpaintingRunner.inpainting_algorithm_ff_indexes(t, len(t_list), s, D)
        t_list_ff = [t_item for t_item in t_list_ff if t_item not in t_list_inpainted]
        return t_list_inpainted + t_list_ff

    @staticmethod
    def inpainting_algorithm_cp(x, m, model_alignment, model, N=20, s=1, e=1):
        print('Inpainting sequence...')
        fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        y_inpainted, m_inpainted = x.unsqueeze(0), m.unsqueeze(0)
        for i in range(N):
            t_list = [t for t in range(y_inpainted.size(2)) if (t // s) % (s if s > 1 else 2) == i % 2]
            for t in t_list:
                if m_inpainted[:, :, t].sum() == 0:
                    continue
                for delta_t in [-s, s]:
                    if not 0 <= t + delta_t < y_inpainted.size(2):
                        continue
                    r_list = [t + delta_t]
                    _, _, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
                        model_alignment, model, y_inpainted[:, :, t], m_inpainted[:, :, t], y_inpainted[:, :, r_list],
                        m_inpainted[:, :, r_list]
                    )
                    m_inpainted[:, :, t] = m_inpainted[:, :, t] - v_map[:, :, 0]
                    y_inpainted[:, :, t] = (1 - m_inpainted[:, :, t]) * y_hat_comp[:, :, 0] + \
                                           m_inpainted[:, :, t].repeat(1, 3, 1, 1) * fill_color
                    if torch.sum(m_inpainted[:, :, t]) * 100 / m_inpainted[:, :, t].numel() < e or i >= N - 2:
                        m_inpainted[:, :, t] = 0
                        y_inpainted[:, :, t] = y_hat_comp[:, :, 0]
        return y_inpainted[0]
