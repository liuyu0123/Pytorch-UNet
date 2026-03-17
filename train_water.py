import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_checkpoint = Path('./checkpoints/')


def create_dataset(dir_img: Path, dir_mask: Path, img_scale: float, mask_suffix: str):
    """辅助函数：创建数据集，自动回退到 BasicDataset"""
    try:
        return CarvanaDataset(dir_img, dir_mask, img_scale, mask_suffix)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(dir_img, dir_mask, img_scale, mask_suffix)


def train_model(
        model,
        device,
        dir_img: Path,
        dir_mask: Path,
        dir_val_img: Path = None,
        dir_val_mask: Path = None,
        mask_suffix: str = '_mask',
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        num_workers: int = 0,
):
    # 1. 创建训练数据集
    train_set = create_dataset(dir_img, dir_mask, img_scale, mask_suffix)
    
    # 2. 创建验证数据集（优先使用独立的验证集目录）
    if dir_val_img is not None and dir_val_mask is not None:
        val_set = create_dataset(dir_val_img, dir_val_mask, img_scale, mask_suffix)
        n_val = len(val_set)
        n_train = len(train_set)
        logging.info(f'Using separate validation set: {n_train} train, {n_val} val')
    else:
        # 自动划分 train/val
        n_val = int(len(train_set) * val_percent)
        n_train = len(train_set) - n_val
        train_set, val_set = random_split(
            train_set, [n_train, n_val], 
            generator=torch.Generator().manual_seed(0)
        )
        logging.info(f'Auto-split dataset: {n_train} train, {n_val} val ({val_percent*100:.1f}%)')

    # 3. 创建 data loaders
    loader_args = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True if device.type == 'cuda' else False
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Num workers:     {num_workers}
    ''')

    # 4. 设置优化器、损失函数、学习率调度器
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay, 
        momentum=momentum, 
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    grad_scaler = torch.amp.GradScaler(device_type, enabled=amp)
    
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device_type, enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # 保存训练集的 mask_values（验证集可能类别不全）
            state_dict['mask_values'] = train_set.dataset.mask_values if hasattr(train_set, 'dataset') else train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100), ignored if --val-images is provided')
    
    # 训练集路径（必填）
    parser.add_argument('--images', '-i', type=str, required=True, 
                        help='Path to the training images directory')
    parser.add_argument('--masks', '-m', type=str, required=True, 
                        help='Path to the training masks directory')
    
    # 验证集路径（可选，提供后将使用独立验证集，不再自动划分）
    parser.add_argument('--val-images', type=str, default=None,
                        help='Path to the validation images directory (optional, overrides --validation)')
    parser.add_argument('--val-masks', type=str, default=None,
                        help='Path to the validation masks directory (optional, must be used with --val-images)')
    
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--mask-suffix', type=str, default='_mask',
                        help='Suffix for mask files (default: _mask)')
    parser.add_argument('--workers', '-w', type=int, default=0,
                        help='Number of data loading workers (default: 0, recommended for Windows)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dir_img = Path(args.images)
    dir_mask = Path(args.masks)
    
    # 处理可选的验证集路径
    dir_val_img = Path(args.val_images) if args.val_images else None
    dir_val_mask = Path(args.val_masks) if args.val_masks else None
    
    # 检查：如果提供了 val-images，必须也提供 val-masks
    if (dir_val_img is None) != (dir_val_mask is None):
        raise ValueError('Both --val-images and --val-masks must be provided together, or neither.')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    
    try:
        train_model(
            model=model,
            device=device,
            dir_img=dir_img,
            dir_mask=dir_mask,
            dir_val_img=dir_val_img,
            dir_val_mask=dir_val_mask,
            mask_suffix=args.mask_suffix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.workers
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            dir_img=dir_img,
            dir_mask=dir_mask,
            dir_val_img=dir_val_img,
            dir_val_mask=dir_val_mask,
            mask_suffix=args.mask_suffix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.workers
        )