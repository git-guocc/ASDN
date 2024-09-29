import os
import argparse
from tqdm.auto import tqdm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from utils.misc import get_log_dir_name_tblogger, seed_all, str_list
import shutil
from pytorch_lightning.callbacks import LearningRateMonitor
from models.classify import Classify

def main(args):

    # Logging
    local_rank = os.environ.get('LOCAL_RANK', 0)

    if local_rank == 0:
        log_dir_name = get_log_dir_name_tblogger(name='D%s_' % (args.dataset))
        if args.resume_from_checkpoint is None:
            os.makedirs(os.path.join(args.log_root, log_dir_name))
        os.environ['LOG_DIR_NAME'] = log_dir_name
    else:
        log_dir_name = os.environ['LOG_DIR_NAME']

    log_dir = os.path.join(args.log_root, log_dir_name)
    if args.resume_from_checkpoint is not None:
        log_dir = os.path.dirname(args.resume_from_checkpoint)
        log_dir_name = os.path.basename(log_dir)

    # 复制脚本文件到日志目录
    if args.resume_from_checkpoint is None:
        files_to_save = ['./models/classifyNet.py','./models/utils.py', './models/classify.py']
        for file_ in files_to_save:
            shutil.copyfile(file_, os.path.join(log_dir, os.path.basename(file_)))
            

    # configure logging on module level, redirect to file
    logger = logging.getLogger('pytorch_lightning.core')
    logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'run.log')))

    # Model
    logger.info('INFO: Building model...')
    model = Classify(args)

    # # Use DistributedDataParallel
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    logger.info(repr(model))

    # Main loop
    try:
        logger.info('INFO: Start training...')
        # seed_everything(args.seed, workers=True)

        # 在创建Trainer实例时添加回调
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # 创建一个RichProgressBar实例
        trainer = Trainer(
            accelerator='gpu',
            devices=args.n_gpu,
            num_nodes=1,
            logger=TensorBoardLogger(args.log_root, name=log_dir_name),
            deterministic=False,
            max_epochs=800,
            check_val_every_n_epoch=args.save_interval,
            callbacks=[
                        ModelCheckpoint(
                             monitor='val_loss',
                             every_n_epochs=args.save_interval, 
                             save_on_train_epoch_end=False,
                             save_top_k = -1,
                             dirpath=log_dir,
                             filename='denoisenet-epoch{epoch:02d}-val_loss{val_loss:.6f}',
                             auto_insert_metric_name=False
                        ),
                        lr_monitor,  # 添加这行
                    ],
            # strategy="ddp"
        )
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.info('INFO: Terminating...')
        print('Terminating...')


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--changelog', type=str, default='')
    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'], choices=[['x0000_poisson'], ['x0000_poisson', 'y0000_poisson'], ['x0000_poisson', 'y0000_poisson', 'z0000_poisson']])
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--val_batch_size', type=int, default=12)
    parser.add_argument('--noise_lvs', type=list, default=None)

    # parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])

    ## Optimizer and scheduler
    parser.add_argument('--sched_patience', default=2, type=int, help='Ierativative scheduler patience')
    parser.add_argument('--sched_factor', default=0.7, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)  
    parser.add_argument('--lr', type=float, default=1e-4)

    ## Training
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_root', type=str, default='./logs/classify')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_noise', type=float, default=0.015)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # Ablation parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--frame_knn', type=int, default=32) # Neighbourhood side for graph convolution
    parser.add_argument('--num_modules', type=int, default=10)
    parser.add_argument('--noise_decay', type=int, default=10) # Noise decay is set to 16/T where T=num_modules or set to 1 for no decay

    args = parser.parse_args()
    
    args.n_gpu = [0 ,1]
    args.num_modules = 1
    args.noise_decay = 1
    args.patches_per_shape_per_epoch = 1000

    #固定噪声
    # args.noise_lvs = [0.005, 0.01, 0.015, 0.02]
    # args.dataset = 'pointfilter'
    # 
    # args.resume_from_checkpoint = "logs/DPUNet_2024_06_25__10_49_32/denoisenet-epoch49-val_loss0.000249.ckpt"
    # args.resolutions = ['10000_poisson', '30000_poisson', '50000_poisson']
    args.log_root = './logs/debug'
    args.dataset_root = './data/debug'
    main(args)
