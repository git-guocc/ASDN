import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch3d.ops
from pytorch3d.ops import knn_points
import pytorch3d.loss.chamfer as cd_loss
import numpy as np
from .feature import FeatureExtraction
# from .feature import FeatureExtraction2
import pytorch_lightning as pl
from datasets.pcl import *
from datasets.patch import *
from utils.misc import *
from utils.transforms import *
from models.utils import chamfer_distance_unit_sphere
from models.utils import farthest_point_sampling
from torch.cuda.amp import autocast
from .InfoCD import *
# from models.utils import get_entropy_B

class DenoiseNetCD(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.feature_nets = FeatureExtraction()
        self.console_logger = logging.getLogger('pytorch_lightning.core')
        
        self.val_out = []
        self.train_out = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.feature_nets.parameters(),
            lr=self.args.lr,
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.args.sched_patience, factor=self.args.sched_factor,
                                           min_lr=self.args.min_lr),
            'interval': 'epoch',
            'frequency': 5,
            'monitor': 'val_loss',
        }
        return [optimizer], [scheduler]
        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def train_dataloader(self):
        # Datasets and loaders
        train_dset = PairedPatchDataset(
            datasets=[
                PointCloudDataset(
                    root=self.args.dataset_root,
                    dataset=self.args.dataset,
                    split='train',
                    resolution=resl,
                    transform=standard_train_transforms(noise_std_max=self.args.noise_max,
                                                        noise_std_min=self.args.noise_min, rotate=self.args.aug_rotate)
                ) for resl in self.args.resolutions
            ],
            split='train',
            patch_size=self.args.patch_size,
            num_patches=self.args.patches_per_shape_per_epoch,
            patch_ratio=self.args.patch_ratio,
            on_the_fly=True  # Currently, we only support on_the_fly=True
        )

        return DataLoader(train_dset, batch_size=self.args.train_batch_size, num_workers=8, pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        # Datasets and loaders
        val_dset = PointCloudDataset(
            root=self.args.dataset_root,
            dataset=self.args.dataset,
            split='test',
            resolution=self.args.resolutions[0],  # 'x0000_poisson'
            transform=standard_train_transforms(noise_std_max=self.args.val_noise, noise_std_min=self.args.val_noise,
                                                rotate=False),
        )

        return DataLoader(val_dset, batch_size=self.args.val_batch_size, num_workers=8, pin_memory=True, shuffle=False)

    def training_step(self, train_batch, batch_idx):
        pcl_noisy = train_batch['pcl_noisy']
        pcl_clean = train_batch['pcl_clean']
        pcl_seeds = train_batch['seed_pnts']
        pcl_std = train_batch['pcl_std']

        loss = self.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_seeds=pcl_seeds,
                                               pcl_std=pcl_std)
        # # Logging
        self.log('loss', loss, prog_bar=True)
        self.train_out.append(loss.item())
        return {"loss": loss, "loss_as_tensor": loss.clone().detach()}

    def validation_step(self, val_batch, batch_idx):
        pcl_clean = val_batch['pcl_clean']
        pcl_noisy = val_batch['pcl_noisy']

        all_clean = []
        all_denoised = []
        for i, data in enumerate(pcl_noisy):
            pcl_denoised = self.patch_based_denoise(data, seed_k_alpha=10)  
            all_clean.append(pcl_clean[i].unsqueeze(0))
            all_denoised.append(pcl_denoised.unsqueeze(0))
        
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)

        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        self.val_out.append(avg_chamfer)
        return torch.tensor(avg_chamfer)

    def on_train_epoch_end(self):
        if self.train_out:
            loss_all = sum(self.train_out) / len(self.train_out)
            self.console_logger.info(f'INFO: Current epoch training loss: {loss_all:.6f}')
            self.log('train_epoch_loss', loss_all, sync_dist=True)
            self.train_out.clear()  # 清空列表以备下次训练阶段使用
        else:
            if self.trainer.is_global_zero:
                self.console_logger.info('INFO: No training outputs to process.')
        
    def on_validation_epoch_end(self):
        if self.val_out:
            val_outs = torch.tensor(self.val_out, device=self.device)
            val_loss_all = val_outs.mean()

            self.console_logger.info(f'INFO: Current epoch validation loss: {val_loss_all:.6f}')
            self.log('val_loss', val_loss_all, sync_dist=True)
            self.val_out.clear()  # 清空列表以备下次验证阶段使用
        else:
            if self.trainer.is_global_zero:
                self.console_logger.info('INFO: No validation outputs to process.')

    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + torch.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float()

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.  e.g. M:1200 N:1000
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)

        losses = torch.zeros(1)

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)
        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        pcl_input = pcl_noisy

        feat = pcl_input.view(B * N_noisy, -1)[:, 3:].cuda()  
        offset = torch.tensor(np.array([(i + 1) * N_noisy for i in range(B)]), dtype=torch.int32).cuda() 
        pred_disp = self.feature_nets(pcl_input, feat, offset)
        
        pred_pcl = pcl_input + pred_disp
        
        InfoCD = calc_cd_like_InfoV2(pred_pcl, pcl_clean)
        losses = InfoCD.mean(dim=-1).mean(dim=-1)

        return losses.sum().mean() # , target, scores, noise_vecs


    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10):
        """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
        assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.size()
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size,
                                                                                return_nn=True)
        patches = patches[0]  # (N, K, 3)

        # Patch stitching preliminaries
        seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

        all_dists = torch.ones(num_patches, N) / 0
        all_dists = all_dists.cuda()
        all_dists = list(all_dists)
        patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)

        for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
            all_dist[patch_id] = patch_dist

        all_dists = torch.stack(all_dists, dim=0)
        weights = torch.exp(-1 * all_dists)

        best_weights, best_weights_idx = torch.max(weights, dim=0)
        patches_denoised = []

        # Denoising
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            # print("Processed {:d}/{:d} patches.".format(i, num_patches))
            curr_patches = patches[i:i + patch_step]  # [1, 1000, 3]

            patches_denoised_temp = self.denoise_langevin_dynamics(curr_patches)

            patches_denoised.append(patches_denoised_temp)
            i += patch_step

        patches_denoised = torch.cat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1

        # Patch stitching
        pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd] for
                        pidx_in_main_pcd, patch in enumerate(best_weights_idx)]

        pcl_denoised = torch.cat(pcl_denoised, dim=0)

        while pcl_denoised.shape[0] != N:
            pcl_denoised = torch.cat((pcl_denoised, pcl_denoised[pcl_denoised.shape[0]-1].unsqueeze(0)), dim=0)
            print(f'pcl_denoised.shape ===> {pcl_denoised.shape}')  # 

        return pcl_denoised

    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()  # 1, 1000, 3
        pred_disps = []

        with torch.no_grad():
            self.feature_nets.eval()

            feat = pcl_noisy.view(B * N, -1)[:, 3:].cuda()  # [B*N, 0]
            offset = torch.tensor(np.array([(i + 1) * N for i in range(B)]), dtype=torch.int32).cuda()

            pred_points = self.feature_nets(pcl_noisy, feat, offset)  
            pred_disps.append(pred_points)

        return pcl_noisy + pred_disps[-1]
