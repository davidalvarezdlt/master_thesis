import skeltorch
import numpy as np
import os
import utils
import torch
import torch.utils
import torch.utils.data
from PIL import Image
import utils.draws
import matplotlib.pyplot as plt


class ThesisRunner(skeltorch.Runner):
    losses_items_ids = None
    losses_it_items = None
    losses_epoch_items = None
    e_train_losses_items = None
    e_validation_losses_items = None
    scheduler = None

    def init_model(self, device):
        raise NotImplemented

    def init_optimizer(self, device):
        raise NotImplemented

    def init_others(self, device):
        self.losses_it_items = {
            'train': {loss_item_id: {} for loss_item_id in self.losses_items_ids},
            'validation': {loss_item_id: {} for loss_item_id in self.losses_items_ids}
        }
        self.losses_epoch_items = {
            'train': {loss_item_id: {} for loss_item_id in self.losses_items_ids},
            'validation': {loss_item_id: {} for loss_item_id in self.losses_items_ids}
        }
        self.logger.info('Number of model parameters: {}'.format(self.get_n_params(self.model)))

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])
        self.losses_it_items = checkpoint_data['losses_it_items']
        self.losses_epoch_items = checkpoint_data['losses_epoch_items']

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict(), 'losses_it_items': self.losses_it_items,
                'losses_epoch_items': self.losses_epoch_items}

    def train_step(self, it_data, device):
        raise NotImplemented

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.e_train_losses_items = {loss_item_id: [] for loss_item_id in self.losses_items_ids}
        self.e_validation_losses_items = {loss_item_id: [] for loss_item_id in self.losses_items_ids}
        self.experiment.tbx.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.counters['epoch'])
        self.experiment.data.regenerate_loader_indexes(self.counters['epoch'])

    def train_iteration_log(self, e_train_losses, log_period, device):
        super().train_iteration_log(e_train_losses, log_period, device)
        self._log_iteration(self.counters['train_it'], self.losses_it_items['train'], self.e_train_losses_items,
                            log_period, 'iteration', 'train')

    def validation_iteration_log(self, e_validation_losses, log_period, device):
        super().validation_iteration_log(e_validation_losses, log_period, device)
        self._log_iteration(self.counters['validation_it'], self.losses_it_items['validation'],
                            self.e_validation_losses_items, log_period, 'iteration', 'validation')

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self._log_iteration(self.counters['epoch'], self.losses_epoch_items['train'], self.e_train_losses_items, 0,
                            'epoch', 'train')
        self._log_iteration(self.counters['epoch'], self.losses_epoch_items['validation'],
                            self.e_validation_losses_items, 0, 'epoch', 'validation')
        self.test(None, device)
        self.scheduler.step(epoch=self.counters['epoch'])

    def _log_iteration(self, it_n, losses_it_items_split, e_losses_items, log_period, ot, split):
        for loss_item_id in self.losses_items_ids:
            losses_it_items_split[loss_item_id][it_n] = np.mean(e_losses_items[loss_item_id][-log_period:])
            self.experiment.tbx.add_scalar(
                'loss_items_{}_{}/{}'.format(split, ot, loss_item_id), losses_it_items_split[loss_item_id][it_n], it_n
            )

    def test(self, epoch, device):
        raise NotImplemented

    def test_losses(self, test_losses_handler, losses_items_ids, device):
        loss_t, losses_items_t = [], [[] for _ in range(len(losses_items_ids))]

        for it_data in self.experiment.data.loaders['test']:
            (x, m), y, info = it_data
            x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
            t, r_list = self.get_indexes(x.size(2))
            loss, loss_items = test_losses_handler(x, m, y, flows_use, flow_gt, t, r_list)
            loss_t.append(loss.item())
            for i, loss_item in enumerate(loss_items):
                losses_items_t[i].append(loss_item.item())
        self.experiment.tbx.add_scalar('loss/epoch/test', np.mean(loss_t), self.counters['epoch'])
        for i, loss_item_id in enumerate(losses_items_ids):
            self.experiment.tbx.add_scalar(
                'loss_items_test_epoch/{}'.format(loss_item_id), np.mean(losses_items_t[i]), self.counters['epoch'])
        self.logger.info('Epoch: {} | Average Test Loss: {}'.format(self.counters['epoch'], np.mean(loss_t)))

    def _test_objective_measures(self, samples_dataset, samples_indexes, device):
        subset_dataset = torch.utils.data.Subset(samples_dataset, samples_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Initialize LPSIS model to proper device
        self.utils_measures.init_lpips(device)

        # Create variables to store PSNR, SSIM and LPIPS
        psnr, ssim, lpips = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, _ = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (_, y_hat_comp, _, _, (_, _)), t, _ = self.train_step_propagate(x, m, y)
                psnr += self.utils_measures.psnr(y[:, :, t].cpu(), y_hat_comp.cpu())
                ssim += self.utils_measures.ssim(y[:, :, t].cpu(), y_hat_comp.cpu())
                lpips += self.utils_measures.lpips(y[:, :, t], y_hat_comp)

        # Remove LPSIS model from memory
        self.utils_measures.destroy_lpips()

        # Log measures in TensorBoard
        self.experiment.tbx.add_scalar('test_measures/psrn', np.mean(psnr), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/ssim', np.mean(ssim), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/lpips', np.mean(lpips), global_step=self.counters['epoch'])

    def test_frames(self, test_handler, split, device, include_y_hat=True, include_y_hat_comp=True,
                    include_y_hat_trivial=True):
        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        if split == 'validation':
            subset_dataset = torch.utils.data.Subset(
                self.experiment.data.datasets['validation'], self.experiment.data.validation_frames_indexes
            )
        else:
            subset_dataset = torch.utils.data.Subset(
                self.experiment.data.datasets['test'], self.experiment.data.test_frames_indexes
            )

        # Create a loader using the previous subset
        batch_size = self.experiment.configuration.get('training', 'batch_size')
        loader = torch.utils.data.DataLoader(subset_dataset, batch_size)

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, m_tbx, y_tbx, x_aligned_tbx, v_map_tbx = [], [], [], [], []
        y_hat_trivial_tbx, y_hat_tbx, y_hat_comp_tbx = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

            # Use middle frame as the target frame
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)

            # Propagate using the handler
            x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp = test_handler(x, m, y, t, r_list)

            # Add items to the lists
            x_tbx.append(x.cpu().numpy())
            m_tbx.append(m.cpu().numpy())
            y_tbx.append(y.cpu().numpy())
            x_aligned_tbx.append(x_ref_aligned.cpu().numpy())
            v_map_tbx.append(v_map.cpu().numpy())
            if include_y_hat:
                y_hat_tbx.append(y_hat.cpu().numpy())
            if include_y_hat_comp:
                y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())
            if include_y_hat_trivial:
                y_hat_trivial_tbx.append(self._test_frames_trivial(x[:, :, t], x_ref_aligned, v_map).cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx)
        m_tbx = np.concatenate(m_tbx)
        y_tbx = np.concatenate(y_tbx)
        x_aligned_tbx = np.concatenate(x_aligned_tbx)
        v_map_tbx = np.concatenate(v_map_tbx)
        y_hat_tbx = np.concatenate(y_hat_tbx) if len(y_hat_tbx) > 0 else y_hat_tbx
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx) if len(y_hat_comp_tbx) > 0 else y_hat_tbx
        y_hat_trivial_tbx = np.concatenate(y_hat_trivial_tbx) if len(y_hat_trivial_tbx) > 0 else y_hat_tbx

        # Add each batch item individually
        for b in range(x_tbx.shape[0]):
            x_aligned_sample = np.insert(arr=x_aligned_tbx[b], obj=t, values=x_tbx[b, :, t], axis=1)
            x_aligned_sample = utils.draws.add_border(x_aligned_sample, m_tbx[b, :, t])
            v_map_rep, m_rep = v_map_tbx[b].repeat(3, axis=0), m_tbx[b, :, t].repeat(3, axis=0)
            v_map_sample = np.insert(arr=v_map_rep, obj=t, values=m_rep, axis=1)
            sample = np.concatenate((x_tbx[b], x_aligned_sample, v_map_sample), axis=2)
            if include_y_hat_trivial:
                y_hat_trivial_sample = np.insert(arr=y_hat_trivial_tbx[b], obj=t, values=y_tbx[b, :, t], axis=1)
                sample = np.concatenate((sample, y_hat_trivial_sample), axis=2)
            if include_y_hat:
                y_hat_sample = np.insert(arr=y_hat_tbx[b], obj=t, values=y_tbx[b, :, t], axis=1)
                sample = np.concatenate((sample, y_hat_sample), axis=2)
            if include_y_hat_comp:
                y_hat_comp_sample = np.insert(arr=y_hat_comp_tbx[b], obj=t, values=y_tbx[b, :, t], axis=1)
                sample = np.concatenate((sample, y_hat_comp_sample), axis=2)
            self.experiment.tbx.add_images(
                'frames_{}/{}'.format(split, b + 1), sample.transpose(1, 0, 2, 3), global_step=self.counters['epoch']
            )

    def test_sequence(self, handler, folder_name, model_alignment, model, device):
        for it_data in self.experiment.data.datasets['test_sequences']:
            (x, m), y, info = it_data
            x, m = x.to(device), m.to(device)
            self._save_sample(handler(x, m, model_alignment, model).cpu().numpy() * 255, folder_name, info[0])

    def _test_frames_trivial(self, x_target, x_ref_aligned, v_map):
        b, c, ref_n, h, w = x_ref_aligned.size()
        return x_target.unsqueeze(2).repeat(1, 1, ref_n, 1, 1) * (1 - v_map) + x_ref_aligned * v_map

    @staticmethod
    def inpainting_hard_copy(x_target, v_target, x_ref_aligned, v_ref_aligned, v_map):
        y_hat = y_hat_comp = (1 - v_map[:, :, 0]) * x_target + v_map[:, :, 0] * x_ref_aligned[:, :, 0]
        return y_hat.unsqueeze(2), y_hat_comp.unsqueeze(2)

    def get_indexes(self, size):
        t, r_list = size // 2, list(range(size))
        r_list.pop(t)
        return t, r_list

    def _save_sample(self, y_hat_comp, folder_name, file_name, save_as_video=True):
        """Saves a set of samples.

        Args:
            y_hat_comp (np.Array): array of size (C,F,H,W) containing the frames to save.
            folder_name (str): name of the folder where the results should be stored inside.
            file_name (str): name of the sequences to be stored.
            save_as_video (bool): whether to store the sequence as a video.
        """
        path = os.path.join(self.experiment.paths['results'], folder_name, 'epoch-{}'.format(self.counters['epoch']))
        if not os.path.exists(path):
            os.makedirs(path)

        # Permute the Array to be (F,H,W,C)
        y_hat_comp = y_hat_comp.transpose(1, 2, 3, 0)

        # Save sequence as video
        if save_as_video:
            frames_to_video = utils.FramesToVideo(0, 10, None)
            frames_to_video.add_sequence(y_hat_comp)
            frames_to_video.save(path, file_name)
        else:
            for t in range(y_hat_comp.shape[1]):
                bt_path = os.path.join(path, file_name)
                if not os.path.exists(bt_path):
                    os.makedirs(bt_path)
                pil_img = Image.fromarray(y_hat_comp[t])
                pil_img.save(os.path.join(bt_path, '{}.jpg'.format(t)))

        # Log progress
        self.logger.info('Epoch {} | Test generated as {} for sequence {}'.format(
            self.counters['epoch'], 'video' if save_as_video else 'frames', file_name
        ))
