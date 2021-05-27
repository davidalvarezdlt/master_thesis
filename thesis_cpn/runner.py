import torch.optim
import numpy as np
import thesis.runner
import models.cpn
import models.cpn_original
import torch.utils.data
import copy
import utils.measures
import utils.losses
import utils.draws
import models.vgg_16
import matplotlib.pyplot as plt
import os.path


class ThesisCPNRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    scheduler = None
    utils_losses = None
    utils_measures = None
    losses_items_ids = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        if self.experiment.configuration.get('model', 'version') == 'original':
            self.model = ThesisCPNRunner.init_model_with_state(models.cpn_original.CPNOriginal(), device)
        else:
            self.model = models.cpn.CPNet(self.experiment.configuration.get('model', 'mode'), None).to(device)

    @staticmethod
    def init_model_with_state(model, device):
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'cpn', 'cpn.pth')
        checkpoint_data = dict(torch.load(checkpoint_path, map_location=device))
        model_state = model.state_dict()
        for ck_item, k_data in checkpoint_data.items():
            if ck_item.replace('module.', '') in model_state:
                model_state[ck_item.replace('module.', '')].copy_(k_data)
        model.load_state_dict(model_state)
        for param in model.parameters():
            param.requires_grad = False
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
        self.utils_measures = utils.measures.UtilsMeasures()
        self.losses_items_ids = ['alignment', 'vh', 'nvh', 'nh', 'perceptual', 'style', 'tv', 'grad']
        super().init_others(device)

    def train_step(self, it_data, device):
        raise NotImplementedError
        # (x, m), y, info = it_data
        # x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
        # t, r_list = self.get_indexes(x.size(2))
        #
        # # Propagate through the model
        # x_refs_aligned, v_refs_aligned, v_map = ThesisCPNRunner.train_step_propagate(
        #     self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        # )
        #
        # # Get visibility map of aligned frames and target frame
        # visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * v_refs_aligned
        #
        # # Get both total loss and loss items
        # loss, loss_items = ThesisCPNRunner.compute_loss(
        #     self.utils_losses, None, y[:, :, t], y_hat, y_hat_comp, x[:, :, t], x_refs_aligned, visibility_maps,
        #     m[:, :, t], None
        # )
        #
        # # Append loss items to epoch dictionary
        # e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        # for i, loss_item in enumerate(self.losses_items_ids):
        #     e_losses_items[loss_item].append(loss_items[i].item())
        #
        # # Append loss items to epoch dictionary
        # e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        # for i, loss_item in enumerate(self.losses_items_ids):
        #     e_losses_items[loss_item].append(loss_items[i].item())
        #
        # # Return combined loss
        # return loss

    # @staticmethod
    # def train_step_propagate(model, x_target, m_target, y_target, x_refs, m_refs):
    #     return model(x_target, m_target, y_target, x_refs, m_refs)

    @staticmethod
    def infer_alignment_step_propagate(model, x_target, m_target, x_refs, m_refs):
        return model.align(x_target, m_target, x_refs, m_refs)

    # @staticmethod
    # def compute_loss(utils_losses, loss_weights, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):
    #     reduction = 'mean'
    #     x_extended = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1)
    #     loss_alignment = utils_losses.masked_l1(x_extended, x_aligned, v_map, 'sum', loss_weights[0])
    #     loss_vh = utils_losses.masked_l1(y_t, y_hat, m * (1 - c_mask), reduction, loss_weights[1])
    #     loss_nvh = utils_losses.masked_l1(y_t, y_hat, m * c_mask, reduction, loss_weights[2])
    #     loss_nh = utils_losses.masked_l1(y_t, y_hat, 1 - m, reduction, loss_weights[3])
    #     loss_perceptual, vgg_y, vgg_y_hat_comp = utils_losses.perceptual(y_t, y_hat_comp, loss_weights[4])
    #     loss_style = utils_losses.style(vgg_y, vgg_y_hat_comp, loss_weights[5])
    #     loss_tv = utils_losses.tv(y_hat, loss_weights[6])
    #     loss_grad = utils_losses.grad(y_t, y_hat, 1, reduction, loss_weights[7])
    #     loss = loss_vh + loss_nvh + loss_nh + loss_perceptual + loss_style + loss_tv + loss_grad
    #     return loss, [loss_alignment, loss_vh, loss_nvh, loss_nh, loss_perceptual, loss_style, loss_tv, loss_grad]

    def test(self, epoch, device):
        self.model.eval()
        self.test_sequence(
            ThesisCPNRunner.inpainting_algorithm, 'algorithm_cpn', None, self.model, device
        )

    @staticmethod
    def inpainting_algorithm(x, m, model_alignment, model):
        c, f, h, w = x.size()
        x, m = x.unsqueeze(0), m.unsqueeze(0)

        # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
        y_inpainted = np.zeros((2, c, f, h, w), dtype=np.float32)

        # Use the model twice: forward (0) and backward (1)
        for d in range(2):
            x_copy, m_copy = x.clone(), m.clone()

            # Iterate over all the frames of the video
            for t in (list(range(f)) if d == 0 else reversed(list(range(f)))):
                r_list = ThesisCPNRunner.get_reference_frame_indexes(t, f)

                # Replace input_frames and input_masks with previous predictions to improve quality
                with torch.no_grad():
                    _, x_copy[:, :, t] = model(
                        x_copy[:, :, t], m_copy[:, :, t], x_copy[:, :, r_list], m_copy[:, :, r_list]
                    )
                    m_copy[:, :, t] = 0
                    y_inpainted[d, :, t] = x_copy[:, :, t].squeeze(0).detach().cpu().numpy()

        # Combine forward and backward predictions
        forward_factor = np.arange(start=0, stop=y_inpainted.shape[2]) / f
        backward_factor = (f - np.arange(start=0, stop=y_inpainted.shape[2])) / f
        y_inpainted = (
                y_inpainted[0].transpose(0, 2, 3, 1) * forward_factor +
                y_inpainted[1].transpose(0, 2, 3, 1) * backward_factor
        ).transpose(0, 3, 1, 2)
        return torch.from_numpy(y_inpainted).float().to(x.device)

    @staticmethod
    def get_reference_frame_indexes(t, n_frames, p=2, r_list_max_length=120):
        # Set start and end frames
        start = t - r_list_max_length
        end = t + r_list_max_length

        # Adjust frames in case they are not in the limit
        if t - r_list_max_length < 0:
            end = (t + r_list_max_length) - (t - r_list_max_length)
            end = n_frames - 1 if end > n_frames else end
            start = 0
        elif t + r_list_max_length > n_frames:
            start = (t - r_list_max_length) - (t + r_list_max_length - n_frames)
            start = 0 if start < 0 else start
            end = n_frames - 1

        # Return list of reference_frames every n=2 frames
        return [i for i in range(start, end, p) if i != t]
