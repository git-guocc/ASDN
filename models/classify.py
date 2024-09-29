import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch3d.ops
from .classifyNet import ScaleNet
import pytorch_lightning as pl
from datasets.pcl import *
from datasets.patch import *
from utils.misc import *
from utils.transforms import *
from models.utils import farthest_point_sampling
from models.utils import get_entropy_B

class Classify(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn

        self.feature_nets = nn.ModuleList()
        self.console_logger = logging.getLogger('pytorch_lightning.core')
        # networks
        input_dim = 3
        z_dim = 0 
        self.feature_nets.append(ScaleNet(k=self.frame_knn, input_dim=input_dim, z_dim=z_dim, embedding_dim=256, output_dim=1))
        input_dim = 3 + z_dim

        self.val_out = []
        self.train_out = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                                        self.feature_nets.parameters(), 
                                        lr=self.args.lr, 
                                    )
        scheduler = {
                        'scheduler': ReduceLROnPlateau(optimizer, patience=self.args.sched_patience, factor=self.args.sched_factor, min_lr=self.args.min_lr),
                        'interval': 'epoch',
                        'frequency': 5,
                        'monitor': 'val_loss',
                    }
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        # Datasets and loaders
        train_dset = PairedPatchDataset(
            datasets=[
                PointCloudDataset(
                    root=self.args.dataset_root,
                    dataset=self.args.dataset,
                    split='train',
                    resolution=resl,
                    transform=standard_train_transforms(noise_std_max=self.args.noise_max, noise_std_min=self.args.noise_min, rotate=self.args.aug_rotate)
                ) for resl in self.args.resolutions
            ],
            split='train',
            patch_size=self.args.patch_size,
            num_patches=self.args.patches_per_shape_per_epoch,
            patch_ratio=self.args.patch_ratio,
            transform=None
        )

        return DataLoader(train_dset, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        # Datasets and loaders
        val_dset = PointCloudDataset(
                        root=self.args.dataset_root,
                        dataset=self.args.dataset,
                        split='test',
                        resolution=self.args.resolutions[2],
                        transform=standard_train_transforms(noise_std_max=self.args.val_noise, noise_std_min=self.args.val_noise, rotate=False, scale_d=0.0),
                    )

        return DataLoader(val_dset, batch_size=self.args.val_batch_size, num_workers=4, pin_memory=True, shuffle=False)
    

    def training_step(self, train_batch, batch_idx):
        pcl_noisy = train_batch['pcl_noisy']
        pcl_clean = train_batch['pcl_clean']
        pcl_seeds = train_batch['seed_pnts']
        pcl_std = train_batch['pcl_std']

        # Forward
        loss = self.get_supervised_loss_nn(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_seeds=pcl_seeds, pcl_std=pcl_std)  

        self.log('loss', loss, prog_bar=True)
        self.train_out.append(loss)
        return {"loss": loss, "loss_as_tensor": loss.clone().detach()} 
    
    def validation_step(self, val_batch, batch_idx):
        pcl_clean = val_batch['pcl_clean']
        pcl_noisy = val_batch['pcl_noisy']
        all_clean = []
        all_denoised = []

        for i, (pcl_noisy, pcl_clean) in enumerate(zip(pcl_noisy, pcl_clean)):
            pre_scale, scale_gt = self.patch_based_shang(pcl_noisy, pcl_clean) #返回整个点云
            all_clean.append(scale_gt.unsqueeze(0))
            all_denoised.append(pre_scale.unsqueeze(0))

        avg = (torch.abs(torch.cat(all_clean, dim=0).squeeze() - torch.cat(all_denoised, dim=0).squeeze())).mean().item()
        self.val_out.append(avg)
        return torch.tensor(avg)
    
    def on_train_epoch_end(self):
        if self.train_out:
            loss_all = torch.stack(self.train_out, dim=0)
            loss_all = loss_all.mean()
            if self.trainer.is_global_zero:
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
            if self.trainer.is_global_zero:
                self.console_logger.info(f'INFO: Current epoch validation loss: {val_loss_all:.6f}')
            self.log('val_loss', val_loss_all, sync_dist=True)
            self.val_out.clear()  # 清空列表以备下次验证阶段使用
        else:
            if self.trainer.is_global_zero:
                self.console_logger.info('INFO: No validation outputs to process.')

    def get_supervised_loss_nn(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        mse_loss = nn.MSELoss(reduction='sum')
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        
        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)

        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        NoiseEntropy = get_entropy_B(pcl_noisy)
        CleanEntropy = get_entropy_B(pcl_clean)

        entropy_ratio = NoiseEntropy / CleanEntropy

        pre, _ = self.feature_nets[0](pcl_noisy, None)
        loss = mse_loss(pre.squeeze(), entropy_ratio.squeeze())

        return loss
    
    def patch_based_shang(self, pcl_noisy, pcl_clean, patch_size=1000, seed_k=5, seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
        assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.size()
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        pcl_clean = pcl_clean.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, indices = farthest_point_sampling(pcl_noisy, num_patches)
        seed_clean_pnts = pcl_clean[0][indices].unsqueeze(0)
        _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        _, _, patches_clean = pytorch3d.ops.knn_points(seed_clean_pnts, pcl_clean, K=patch_size, return_nn=True)
        patches = patches[0]    # (N, K, 3)
        patches_clean = patches_clean[0]    # (N, K, 3)
        seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        seed_pnts_2 = seed_clean_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
        patches_clean = patches_clean - seed_pnts_2
        pre_scale = []

        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            # print("Processed {:d}/{:d} patches.".format(i, num_patches))
            curr_patches = patches[i:i+patch_step]

            try:
                if num_modules_to_use is None:
                    patches_denoised_temp, _ = self.feature_nets[0](curr_patches, None)
                else:
                    patches_denoised_temp, _ = self.feature_nets[0](curr_patches, None)

            except Exception as e:
                print("="*100)
                print(e)
                print("="*100)
                print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.") 
                print("Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
                print("="*100)
                return
            pre_scale.append(patches_denoised_temp)
            i += patch_step

        pre_scale = torch.cat(pre_scale, dim=0)
        shang_clean = get_entropy_B(patches_clean)
        shang_noise = get_entropy_B(patches)
        scale_gt = shang_noise / shang_clean

        return pre_scale, scale_gt