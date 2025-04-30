# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
from tqdm import tqdm
import logging
import os
from transforms.apply_transforms import get_pretrain_transformation
from dataset.datamodule_handler import get_data_modules
from utils.labels import get_diffusion_classes
from models.dit import DiT_models
from models import create_diffusion
from download import find_model
from diffusers.models import AutoencoderKL
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter 

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def get_lr_lambda(warmup_epochs):
    """
    Returns a lambda function for linear warm-up.
    """
    return lambda epoch: (epoch + 1) / warmup_epochs

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("gloo")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(log_dir=experiment_dir)
    else:
        logger = create_logger(None)
        writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    if args.checkpoint:
        state_dict = find_model(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    logger.info(f"Using {args.scheduler} scheduler")
    """"OCT pre-training"""
    print('loading dataset..')
    data_modules = get_data_modules(batch_size=args.global_batch_size,
                                    classes=get_diffusion_classes(),
                                    train_transform=get_pretrain_transformation(args.image_size, channels=3, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    test_transform=get_pretrain_transformation(args.image_size, channels=3, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    )
    
    dataset = ConcatDataset([
                                    data_modules[0].data_train,
                                    data_modules[1].data_train,
                                    data_modules[0].data_val,
                                    data_modules[1].data_val,
                                    ])
    
    print("concatenated datasets")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    print("dataloder is built")
    logger.info(f"Dataset contains {len(dataset):,}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # configure scheduler
    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=1e-6)
    elif args.scheduler == "linear_decay":
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=args.epochs * len(loader))
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    print("start")
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # Wrap the loader with tqdm
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=(rank != 0))
        for i, (x, y, text) in enumerate(progress_bar):
            x = x.to(device)
            y = y.long().to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # model_kwargs = dict(y=text_embeddings, conditioning_type="text")
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Call scheduler step after each iteration
            if args.scheduler in ["cosine", "linear_decay"]:
                scheduler.step()
            elif args.scheduler == "plateau":
                scheduler.step(epoch_loss / len(loader))

            # Log loss values:
            running_loss += loss.item()
            epoch_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    writer.add_scalar('Learning Rate', opt.param_groups[0]['lr'], train_steps)

                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint periodically
            if train_steps % args.ckpt_every == 0 and rank == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    # Final model evaluation
    model.eval()
    logger.info("Training complete. Evaluating the final model...")
    # Add any sampling or FID calculation here with ema or model

    if rank == 0:
        writer.close()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=108)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=4000)
    parser.add_argument("--checkpoint", type=str, default="DiT-XL-2-256x256.pt")
    parser.add_argument(
    "--scheduler", type=str, choices=["none", "cosine", "linear_decay", "plateau"], default="cosine",
    help="Select the learning rate scheduler: 'none', 'cosine', 'linear_decay', 'plateau'"
)
    args = parser.parse_args()
    main(args)


    # CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nnodes=1 --nproc_per_node=3 train.py --model DiT-XL/2