import json
import os
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import numpy as np
import torch
import torch.nn as nn

from augments.utils import util_dataset
from utils import util_general, util_path, util_easydict, util_url

# ------------------------------------------------------------------------------------------------------------------
# General helper Functions
# ------------------------------------------------------------------------------------------------------------------

def init_module(module, gpu_ids=[]):
    """Initialize a network: register CPU/GPU device (with multi-GPU support);
    Parameters:
        module    -- the module to be initialized
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        module.to(gpu_ids[0])
        module = torch.nn.DataParallel(module, gpu_ids)  # multi-GPUs

    return module

def load_vgg():
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    print(f"Loading VGG16 from: {url}")
    with util_url.open_url(url) as f:
        vgg16 = torch.jit.load(f)

    vgg16 = vgg16.eval()
    print('Done.')
    return vgg16

def define_latentaugment(module_name, phase, opt, save_dir, gpu_ids=[]):
    """Create a LatentAugment Policy

    Parameters:
        module_name
        phase
        opt
        save_dir
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a module for LatentAgment Policy
    """

    if module_name == 'latent_aug':
        module = LatentAug(phase, opt, save_dir, gpu_ids)
    elif netG == 'latent_net':
        raise NotImplementedError
    else:
        raise NotImplementedError('Module name [%s] is not recognized' % module_name)
    return init_module(module, gpu_ids)

# ------------------------------------------------------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------------------------------------------------------

class LatentAug(nn.Module):

    def __init__(self, phase, opt, save_dir, gpu_ids):
        super(LatentAug, self).__init__()

        self.save_dir = save_dir
        self.model_dir = opt.model_dir
        self.interim_dir = opt.interim_dir
        self.phase = phase
        if len(gpu_ids) > 0:
            self.world_size = len(gpu_ids)
        else:
            self.world_size = 1

        # Common parameters.
        self.dataset = opt.dataset_aug                         # Common.
        self.dataset_name = opt.dataset_name_aug
        self.modalities = util_general.parse_comma_separated_list(opt.modalities_aug)
        self.res = opt.img_resolution
        self.batch_size = opt.batch_size

        self.exp_stylegan = opt.exp_stylegan                   # StyleGAN.
        self.network_pkl_stylegan = opt.network_pkl_stylegan

        self.dataset_w_name = opt.dataset_w_name               # Projector.
        self.exp_inv = opt.exp_inv

        # Augmentation parameters.
        self.num_epochs = opt.opt_num_epochs
        self.opt_lr = opt.opt_lr
        self.lpips_script = opt.lpips_script
        self.truncation_psi = opt.truncation_psi

        self.w_pix = opt.w_pix                                  # Losses weights.
        self.w_lpips = opt.w_lpips                              # ...
        self.w_latent = opt.w_latent                            # ...
        self.w_disc = opt.w_disc                                # ...

        self.crop_size = opt.crop_size_aug                      # Transformation parameters.
        self.preprocess =  opt.preprocess_aug

        self.soft_aug = opt.soft_aug                            # Gate parameters.
        self.alpha = opt.alpha

        self.verbose_log = opt.verbose_log

        # Load Generator and Discriminator.
        self.G, self.D = self.load_stylegan()
        # Define latent space parameters.
        self.z_dim = self.G.z_dim
        self.w_dim = self.G.w_dim
        self.num_ws = self.G.num_ws

        # Losses definition.
        if self.w_pix > 0.0:
            self.mse_loss = torch.nn.MSELoss().eval()
        if self.w_lpips > 0.0:
            if self.lpips_script == 'lpips_script':
                self.vgg16 = load_vgg()
            else:
                from augments.criteria.lpips.lpips import LPIPS
                self.lpips = LPIPS(net_type='vgg', report_dir=self.save_dir).eval()  # work in [-1 1]
        # Save results for each optimization.
        self.stats_loss = util_easydict.EasyDict()
        self.stats_time = util_easydict.EasyDict()

        # Create or load dataset from which calculate the distances.
        cache_dir = os.path.join(self.interim_dir, self.dataset, 'cache_dir')

        # Create dataset.
        self.stats_dataset_w = util_dataset.LatentCodeDataset(
            path=os.path.join(self.interim_dir, self.dataset, self.dataset_w_name + '.zip'), split=self.phase,
            w_dim=self.w_dim, num_ws=self.num_ws
        )
        if self.w_latent > 0.0:
            stats = self.compute_stats(
                dataset=self.stats_dataset_w, manifold='latent', cache_dir=cache_dir, step=opt.step_w,
            )
            self.register_buffer('W', stats.get_all_torch())

        if self.w_pix > 0.0:
            # Create dataset.
            stats_dataset_img = util_dataset.ImgDataset(
                path=os.path.join(self.interim_dir, self.dataset, self.dataset_name + '.zip'), modalities=self.modalities, split=self.phase, resolution=self.res
            )
            stats = self.compute_stats(
                dataset=stats_dataset_img, manifold='img', cache_dir=cache_dir, step=opt.step_img,
            )
            self.register_buffer('X', stats.get_all_torch())

        if self.w_lpips > 0.0:
            if self.lpips_script == 'lpips_script':
                # Create dataset.
                stats_dataset_img = util_dataset.ImgDataset(
                    path=os.path.join(self.interim_dir, self.dataset, self.dataset_name + '.zip'), modalities=self.modalities, split=self.phase, resolution=self.res
                )

                for mode_id, mode in enumerate(self.modalities):
                    stats = self.compute_stats(
                        dataset=stats_dataset_img, manifold='features_jit', cache_dir=cache_dir, cache_tag=mode+"-"+str(opt.crop_size_aug), step=opt.step_img, mode_id=mode_id
                    )
                    self.register_buffer(f'fea_{mode}', stats.get_all_torch())
            else:
                # Create dataset.
                stats_dataset_img = util_dataset.ImgDataset(
                    path=os.path.join(self.interim_dir, self.dataset, self.dataset_name + '.zip'),   modalities=self.modalities, split=self.phase, resolution=self.res
                )

                for mode_id, mode in enumerate(self.modalities):
                    stats = self.compute_stats(
                        dataset=stats_dataset_img, manifold='features', cache_dir=cache_dir, cache_tag=mode + "-" + str(opt.crop_size_aug), step=opt.step_img, mode_id=mode_id
                    )
                    fea_mode = stats.get_all_list()
                    # Across layers.
                    for i, target_layer in enumerate(self.lpips.target_layers):
                        self.register_buffer(f'fea_{mode}_{target_layer}', fea_mode[i])

        # Visualization
        if self.verbose_log:
            self.verbose_flag = True
        else:
            self.verbose_flag = False

    def get_fea(self, mode):
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/3
        for target_layer in self.lpips.target_layers:
            yield self.__getattr__(f'fea_{mode}_{target_layer}')


    # ------------------------------------------------------------------------------------------------------------------
    # Forward Functions

    def forward_ganrand(self, w):
        w_aug = self.G.mapping(w, c=None, truncation_psi=self.truncation_psi)
        imgAB_aug = self.synthetize(w_aug)
        return imgAB_aug, w_aug

    def forward(self, w, fname):

        if w.ndim == 2:
            w = self.z_to_w(w)

        w_opt = torch.tensor(w, dtype=torch.float32, requires_grad=True)  # [batch_size, 1, dim_w]
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=self.opt_lr)

        # Start optimization
        params = util_dataset.get_params(load_size=self.res, crop_size=self.crop_size, preprocess=self.preprocess) # Set a transformation pipeline on images.
        transform_aug = util_dataset.get_transform(load_size=self.res, crop_size=self.crop_size, preprocess=self.preprocess, params=params)
        transform_center_crop = util_dataset.get_center_crop(load_size=self.res)

        for epoch in range(self.num_epochs):  # epoch cycle.
            tick_epoch = time.time()
            loss_dict = util_general.list_dict()
            time_dict = util_general.list_dict()

            # Create synthetic image.
            ws = self.broadcasting(w_opt)  # [batch_size, num_ws, dim_w]
            x_tilde = self.G.synthesis(ws, noise_mode='const')  # [batch_size x n_modes x res x res]

            # Compute latent loss.
            tik = time.time()
            loss_latent = 0.0
            if self.w_latent > 0.0:
                loss_latent = self.calc_loss_latent(ws, self.W)
                loss_dict[f'loss_latent'].append(float(loss_latent.item()))
            time_dict['time_latent'].append(time.time() - tik)

            # --------------------------------------------------------
            # Image dependent loss.
            # --------------------------------------------------------
            # Compute discriminator loss.
            tik = time.time()
            loss_disc = 0.0
            if self.w_disc > 0.0:
                loss_disc = self.calc_loss_disc(x_tilde)
                loss_dict[f'loss_disc'].append(float(loss_disc.item()))
            time_dict['time_disc'].append(time.time() - tik)

            # Mode dependent losses.
            # Compute pixel loss.
            tik = time.time()
            loss_pix = 0.0
            if self.w_pix > 0.0:
                loss_pix = self.calc_loss_pix(transform_center_crop(x_tilde), transform_center_crop(self.X))
                loss_dict[f'loss_pix'].append(float(loss_pix.item()))
            time_dict['time_pix'].append(time.time() - tik)

            # Compute perceptual loss.
            tik = time.time()
            loss_lpips = 0.0
            if self.w_lpips > 0.0:
                if self.lpips_script == 'lpips_script':
                    loss_lpips = self.calc_loss_lpips_torchscript(transform_aug(x_tilde))
                else:
                    loss_lpips = self.calc_loss_lpips_tr(transform_aug(x_tilde))

                loss_dict[f'loss_lpips'].append(float(loss_lpips.item()))
            time_dict['time_lpips'].append(time.time() - tik)

            # Compute total loss
            loss = -loss_latent - loss_pix - loss_lpips + loss_disc
            loss_dict['loss'].append(float(loss.item()))
            time_dict['time_epoch'].append(time.time() - tick_epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose_flag:
                # Register the loss and times for every epoch.
                desc = ""
                self.stats_loss[f'epoch_{epoch}'] = util_easydict.EasyDict()
                for (loss_name, loss_val) in loss_dict.items():
                    self.stats_loss[f'epoch_{epoch}'][loss_name] = np.mean(loss_val)
                    desc += f"{loss_name} {np.mean(loss_val):<4.2f} "

                desc += "||| "
                self.stats_time[f'epoch_{epoch}'] = util_easydict.EasyDict()
                for (time_name, time_val) in time_dict.items():
                    self.stats_time[f'epoch_{epoch}'][time_name] = np.sum(time_val)
                    desc += f"{time_name} {np.sum(time_val):<4.3f} "
                logprint(f'epoch {epoch + 1:>4d}/{self.num_epochs}, {desc}')
                # Snap the current latent code and image to the disk.
                if x_tilde.shape[0] == 1: # only with batch size of 1
                    self.snap_w(w, epoch, fname[0])
                    self.snap_img(x_tilde, epoch, fname[0])

        if self.verbose_flag:
            self.snapshot_stats(self.stats_loss, title='losses')
            self.snapshot_stats(self.stats_time, title='times [s]')
            self.verbose_flag = False

        # Merge real and moved latent code schedule.
        if self.soft_aug:
            w_aug = self.smooth_aug(w, w_opt)
        else:
            w_aug = self.hard_aug(w, w_opt)

        imgAB_aug = self.synthetize(w_aug)

        return imgAB_aug, w_aug

    # ------------------------------------------------------------------------------------------------------------------
    # Loss Functions

    @staticmethod
    def l2_loss_vectorized(X, Y, compute_mean=True):

        if X.ndim == 4:
            assert Y.ndim == 4
            m, c, h, w = Y.shape
            n, _, _, _ = X.shape
            YYt = torch.sum(torch.square(Y), (1, 2, 3))
            XXt = torch.sum(torch.square(X), (1, 2, 3))
            YXt = torch.einsum('nchw, mchw -> nm', [Y, X])

            D = (YYt.unsqueeze(dim=-1) + XXt) - (2 * YXt)

            if compute_mean:
                D = torch.sum(D) / (m * n)  # L2 distance
                D = D / (c * h * w)  # normalization factor

        elif X.ndim == 3:
            assert Y.ndim == 3
            m, c, h = Y.shape
            n, _, _ = X.shape
            YYt = torch.sum(torch.square(Y), (1, 2))
            XXt = torch.sum(torch.square(X), (1, 2))
            YXt = torch.einsum('nch, mch -> nm', [Y, X])

            D = (YYt.unsqueeze(dim=-1) + XXt) - (2 * YXt)

            if compute_mean:
                D = torch.sum(D) / (m * n)  # L2 distance
                D = D / (c * h)  # normalization factor

        elif X.ndim == 2:
            assert Y.ndim == 2
            m, c = Y.shape
            n, _ = X.shape
            YYt = torch.sum(torch.square(Y), 1)
            XXt = torch.sum(torch.square(X), 1)
            YXt = torch.einsum('nc, mc -> nm', [Y, X])

            D = (YYt.unsqueeze(dim=-1) + XXt) - (2 * YXt)
            if compute_mean:
                D = torch.sum(D) / (m * n)  # L2 distance
                D = D / c  # normalization factor
        else:
            raise NotImplementedError

        return D

    def calc_loss_disc(self, x: torch.Tensor):
        assert x.shape == (self.batch_size // self.world_size, len(self.modalities), self.res, self.res)

        # Mode-INdependent loss.
        logits = self.D(x, c=None)
        loss = torch.nn.functional.softplus(-logits).mean()
        loss *= self.w_disc

        return loss

    def calc_loss_pix(self, x: torch.Tensor, x_tr: torch.Tensor):

        loss = 0.0
        for idx_mode, _ in enumerate(self.modalities):
            loss_mode = self.l2_loss_vectorized(
                x[:, idx_mode, :, :].unsqueeze(dim=1),
                x_tr[:, idx_mode, :, :].unsqueeze(dim=1)
            )
            loss_mode *= self.w_pix
            loss += loss_mode

        loss /= len(self.modalities)
        return loss

    def calc_loss_lpips_torchscript(self, x_crop: torch.Tensor):
        loss = 0.0

        for idx_mode, mode in enumerate(self.modalities):
            # Select the tensor from the buffer of the module.
            target_features = self._buffers[f'fea_{mode}']
            # Extract the features.
            x = x_crop[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1])
            synth_features = self.vgg16(x, resize_images=False, return_lpips=True)

            # Compute the losses.
            distance_mode = self.l2_loss_vectorized(
                synth_features,
                target_features,
                compute_mean=False
            )
            loss_mode = distance_mode.sum() / (synth_features.shape[0] * target_features.shape[0])

            loss_mode *= self.w_lpips
            loss += loss_mode

        loss /= len(self.modalities)
        return loss

    def calc_loss_lpips_tr(self, x_crop:  torch.Tensor):
        loss = 0.0
        for idx_mode, mode in enumerate(self.modalities):

            # Extract the features and compute the losses.
            x = x_crop[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1])
            feat = [*self.get_fea(mode)]
            loss_mode = self.lpips.forward_tr(x=x, feat=feat)

            loss_mode *= self.w_lpips
            loss += loss_mode

        loss /= len(self._modalities)
        return loss


    def calc_loss_latent(self, w: torch.Tensor, w_tr: torch.Tensor):
        assert w.shape == (self.batch_size // self.world_size, self.num_ws, self.w_dim)

        loss = self.l2_loss_vectorized(w, w_tr)
        loss *= self.w_latent

        return loss

    #  ------------------------------------------------------------------------------------------------------------------
    # Gate Functions

    def smooth_aug(self, w, w_tilde):
        assert isinstance(w, torch.Tensor)
        assert isinstance(w_tilde, torch.Tensor)
        assert w.shape == (self.batch_size // self.world_size, 1, self.w_dim)
        assert w_tilde.shape == (self.batch_size // self.world_size, 1, self.w_dim)

        w_aug = (self.alpha * w_tilde) + ((1 - self.alpha) * w)

        return self.broadcasting(w_aug)

    def hard_aug(self, w, w_tilde):
        assert isinstance(w, torch.Tensor)
        assert isinstance(w_tilde, torch.Tensor)
        assert w.shape == (self.batch_size // self.world_size, 1, self.w_dim)
        assert w_tilde.shape == (self.batch_size // self.world_size, 1, self.w_dim)

        return self.broadcasting(w_tilde)

    # ------------------------------------------------------------------------------------------------------------------
    # GANs functions

    def z_to_w(self, z):
        w = self.G.mapping(z, c=None, truncation_psi=self.truncation_psi)
        w = self.reverse_broadcasting(w)

        assert w.shape == (self.batch_size, 1, self.w_dim)
        return w

    def load_stylegan(self):
        # Create path to StyleGAN.
        dir_model = os.path.join(self.model_dir, self.dataset, "training-runs", self.dataset_name, util_general.parse_separated_list_comma(self.modalities))
        exp_name = [x for x in os.listdir(dir_model) if self.exp_stylegan in x]  # search for model
        assert len(exp_name) == 1
        path = os.path.join(dir_model, exp_name[0], self.network_pkl_stylegan)

        # Load the network.
        print(f'Loading stylegan from "{path}"...')
        with open(path, 'rb') as f:
            network_dict = pickle.load(f)
            G = network_dict['G_ema']  # subclass of torch.nn.Module
            D = network_dict['D']

        G = G.eval().requires_grad_(False)
        D = D.eval().requires_grad_(False)
        print('Done.')

        return G, D

    def synthetize(self, w_aug):
        assert w_aug.shape == (self.batch_size // self.world_size, self.num_ws, self.w_dim)
        img_aug = self.G.synthesis(w_aug)
        return img_aug #return img_aug.detach().cpu()

    #  -----------------------------------------------------------------------------------------------------------------
    # Helper.
    def broadcasting(self, latent):
        return latent.repeat([1, self.num_ws, 1])  # broadcasting operation [batch_size, 512] -> [batch_size, 14, 512]

    @staticmethod
    def reverse_broadcasting(latent):
        return latent[:, :1, :]  # Reverse the broadcasting operation [batch_size, 14, 512] -> [batch_size, 1, 512]

    # ------------------------------------------------------------------------------------------------------------------
    # Dataset functions.

    def compute_stats(self, dataset, manifold, cache_dir, cache_tag="",  step=10, max_items=100000, **kwargs):

        # Dataloader.
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        assert next(iter(data_loader))[0].shape[0] == 1

        # Define the maximum number of elements to load.
        num_items = len(dataset)
        if max_items is not None:
            num_items = min(num_items, max_items)

        # Try to lookup from cache.
        util_path.create_dir(cache_dir)

        # Define the tag.
        if cache_tag != "":
            cache_tag += "-"
            cache_tag += f"{manifold}-step_{step}-maxitems_{num_items}"
        else:
            cache_tag = f"{manifold}-step_{step}-maxitems_{num_items}"

        # Define the cache directory.
        cache_file = os.path.join(cache_dir, cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file)

        # Load.
        if flag:
            print(f"{manifold} dataset already created in {cache_file}.")
            return util_dataset.DatasetStats.load(cache_file)

        # Initialize.
        print(f"{manifold} dataset initialization.")
        stats = util_dataset.DatasetStats(manifold=manifold, max_items=num_items, step=step)

        # Main loop.
        with tqdm(total=len(data_loader.dataset), unit=' img/latent/fea ') as pbar:
            for x, fname in data_loader:

                if manifold == 'img':
                    x = x / 127.5 - 1  # synthetic images are in the range [-1 1]
                    added_shape = stats.append_torch(x, fname)
                elif manifold == 'latent':
                    added_shape = stats.append_torch(x, fname)
                elif manifold == 'features':
                    fea_list = self.extract_features_mode(x, kwargs['mode_id'])
                    added_shape = stats.append_list(fea_list, fname)
                elif manifold == 'features_torchscript':
                    fea = self.extract_features_mode_torchscript(x, kwargs['mode_id'])
                    added_shape = stats.append_torch(fea, fname)
                else:
                    raise NotImplementedError

                if added_shape < 0:
                    break
                pbar.update(x.shape[0])

        # Save to cache.
        stats.save(cache_file)
        return stats

    def extract_features_mode_torchscript(self, img, mode_id):

        assert isinstance(img, torch.Tensor) and img.ndim == 4
        assert img.shape[0] == 1
        x = img[:, mode_id, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1])

        # Get params for crop.
        params = util_dataset.get_params(load_size=self.res, crop_size=self.crop_size, preprocess=self.preprocess)
        # Get transformations for crop.
        transform = util_dataset.get_transform(load_size=self.res, crop_size=self.crop_size, preprocess=self.preprocess, params=params)
        x = transform(x)

        # Extract features.
        out_f = self.vgg16(x, resize_images=False, return_lpips=True)

        return out_f

    def extract_features_mode(self, img, mode_id):

        assert isinstance(img, torch.Tensor) and img.ndim == 4
        assert img.shape[0] == 1
        x = img[:, mode_id, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1])

        # Get params for crop.
        params = util_dataset.get_params(load_size=self.res, crop_size=self.crop_size, preprocess=self.preprocess)
        # Get transformations for crop.
        transform = util_dataset.get_transform(load_size=self._res, crop_size=self.crop_size, preprocess=self.preprocess, params=params)
        x = transform(x)
        x = x.to(self.device)

        # Extract features.
        out_f = self.lpips.extract_features(x)

        assert out_f[0].shape[0] == img.shape[0] # equal number of samples
        return out_f

    @staticmethod
    def extract_image_mode(img, mode_id):
        assert isinstance(img, torch.Tensor) and img.ndim == 4
        x = img[:, mode_id, :, :].unsqueeze(dim=1)
        assert x.shape[0] == img.shape[0]
        assert x.shape[1] == 1
        return x

    @staticmethod
    def adjust_channel(x):
        if x.shape[1] == 1:
            return x.repeat([1, 3, 1, 1])
        elif x.shape[1] == 3:  # already in the correct format
            return x
        else:
            raise IndexError('Wrong channel shape in the input.')

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization.
    def snapshot_stats(self, stats, title='losses'):
        stats_ticks = list(stats.values())
        stats_keys = list(stats_ticks[0].keys())
        for key in stats_keys:
            fig = plt.figure()
            plt.plot([x[key] for x in stats_ticks], label=key)
            plt.xlabel('epochs')
            plt.ylabel(f'{title}')
            plt.legend()
            fig.subplots_adjust(left=0.15)  # or whatever
            fig.savefig(os.path.join(self.save_dir, f"{title}_{key}.png"), dpi=400, format='png')
            plt.show()
        jsonl_line = json.dumps(stats, indent=2)
        # print(jsonl_line)
        with open(os.path.join(self.save_dir, f'{title}.jsonl'), 'w') as f:
            f.write(jsonl_line + '\n')

    def snap_w(self, w, epoch, fname):
        fname = util_path.get_filename_without_extension(fname)
        w_snap = w.detach().cpu().numpy()
        w_snap = w_snap.squeeze()
        with open(os.path.join(self.save_dir, f'w_{fname}_{epoch}.pkl'), 'wb') as f:
            pickle.dump(w_snap, f, pickle.HIGHEST_PROTOCOL)
    def snap_img(self, img, epoch, fname):
        fname = util_path.get_filename_without_extension(fname)
        img_snap = img.detach().cpu().numpy()

        imgA = img_snap[:, 0, :, :]
        imgB = img_snap[:, 1, :, :]

        img = np.concatenate([imgA[0, :, :], imgB[0, :, :]], axis=1) # 256 x 512

        if img.min() < -1 or img.max() > 1:
            img = np.clip(img, -1.0, 1.0)
        img = ((img + 1) / 2 * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_dir, f"{fname}_{epoch}.png"), img)

    # ------------------------------------------------------------------------------------------------------------------

def logprint(*args):
    print(*args)
