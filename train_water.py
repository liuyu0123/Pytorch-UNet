import argparse
import logging
import os
import random
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

def compute_metrics(pred_mask, true_mask, num_classes):
    """
    计算分割指标：Precision, Recall, F1, IoU
    返回宏平均指标
    """
    # 展平
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    metrics_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        tp = (pred_cls * true_cls).sum()
        fp = (pred_cls * (1 - true_cls)).sum()
        fn = ((1 - pred_cls) * true_cls).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)

        if true_cls.sum() > 0:
            metrics_per_class.append({
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'iou': iou.item(),
            })

    if metrics_per_class:
        return {
            'precision': np.mean([m['precision'] for m in metrics_per_class]),
            'recall': np.mean([m['recall'] for m in metrics_per_class]),
            'f1': np.mean([m['f1'] for m in metrics_per_class]),
            'miou': np.mean([m['iou'] for m in metrics_per_class]),
        }
    return {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0}

@torch.no_grad()
def evaluate_metrics(model, dataloader, device, num_classes, amp=False):
    """评估模型，返回指标和平均推理时间"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    inference_times = []
    
    criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()

    for batch in dataloader:
        images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)

        # 测量推理时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=amp):
            masks_pred = model(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)

        # 计算loss
        if num_classes == 1:
            loss = criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
            preds = (F.sigmoid(masks_pred) > 0.5).squeeze(1).long()
        else:
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            preds = masks_pred.argmax(dim=1)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(preds.cpu())
        all_targets.append(true_masks.cpu())

    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets, num_classes)
    
    metrics['loss'] = total_loss / num_batches
    metrics['inference_time_ms'] = np.mean(inference_times) * 1000
    metrics['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    
    return metrics

def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'total_params': total_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }

def create_dataset(dir_img, dir_mask, img_scale, mask_suffix):
    try:
        return CarvanaDataset(dir_img, dir_mask, img_scale, mask_suffix)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(dir_img, dir_mask, img_scale, mask_suffix)

class CSVLogger:
    """CSV日志记录器，支持增量写入"""
    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.header = [
            'epoch', 'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        self.rows = []

    def log(self, epoch, train_metrics, val_metrics, lr):
        """记录一轮数据"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_metrics['loss']:.6f}",
            'train_precision': f"{train_metrics['precision']:.6f}",
            'train_recall': f"{train_metrics['recall']:.6f}",
            'train_f1': f"{train_metrics['f1']:.6f}",
            'train_miou': f"{train_metrics['miou']:.6f}",
            'val_loss': f"{val_metrics['loss']:.6f}",
            'val_precision': f"{val_metrics['precision']:.6f}",
            'val_recall': f"{val_metrics['recall']:.6f}",
            'val_f1': f"{val_metrics['f1']:.6f}",
            'val_miou': f"{val_metrics['miou']:.6f}",
            'inference_time_ms': f"{val_metrics['inference_time_ms']:.4f}",
            'fps': f"{val_metrics['fps']:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)

    def save(self):
        """保存CSV，覆盖写入以更新所有历史记录"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)

def train_model(
        model,
        device,
        dir_img,
        dir_mask,
        dir_val_img=None,
        dir_val_mask=None,
        mask_suffix='_mask',
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
        val_percent=0.1,
        save_checkpoint=True,
        img_scale=0.5,
        amp=False,
        weight_decay=1e-8,
        momentum=0.999,
        gradient_clipping=1.0,
        num_workers=0,
        log_dir=Path('./logs'),
        model_dir=Path('./checkpoints'),
        save_interval=0,
        model_name='checkpoint',
        log_name='training_log'
):
    # 1. 初始化日志记录器
    csv_path = log_dir / f'{log_name}.csv'
    logger = CSVLogger(csv_path)

    # 记录模型静态信息
    model_info = get_model_info(model)
    info_path = log_dir / f'{log_name}_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Number of Classes: {model.n_classes}\n")
        f.write(f"Bilinear Upsampling: {model.bilinear}\n")
    logging.info(f"Model info: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    # 2. 准备数据
    train_set = create_dataset(dir_img, dir_mask, img_scale, mask_suffix)
    if dir_val_img is not None and dir_val_mask is not None:
        val_set = create_dataset(dir_val_img, dir_val_mask, img_scale, mask_suffix)
        n_val, n_train = len(val_set), len(train_set)
    else:
        n_val = int(len(train_set) * val_percent)
        n_train = len(train_set) - n_val
        train_set, val_set = random_split(train_set, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 3. 优化器与调度器
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    grad_scaler = torch.amp.GradScaler(device_type, enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    best_miou = 0

    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                with torch.autocast(device_type, enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        preds = (F.sigmoid(masks_pred) > 0.5).squeeze(1).long()
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        preds = masks_pred.argmax(dim=1)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
                all_preds.append(preds.cpu())
                all_targets.append(true_masks.cpu())

                pbar.update(images.shape[0])

        # 计算训练指标
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_metrics = compute_metrics(all_preds, all_targets, model.n_classes)
        train_metrics['loss'] = epoch_loss / len(train_loader)

        # 验证
        val_metrics = evaluate_metrics(model, val_loader, device, model.n_classes, amp)
        scheduler.step(val_metrics['miou'])

        # 记录日志
        logger.log(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
        logger.save()  # 每个epoch立即保存CSV

        # 打印摘要
        logging.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, FPS: {val_metrics['fps']:.1f}"
        )

        # 保存最佳模型
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            if save_checkpoint:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = train_set.dataset.mask_values if hasattr(train_set, 'dataset') else train_set.mask_values
                state_dict['epoch'] = epoch
                state_dict['miou'] = best_miou
                torch.save(state_dict, Path(model_dir) / f'{model_name}_best.pth')

        # 定期保存检查点
        if save_checkpoint and save_interval > 0 and epoch % save_interval == 0:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.dataset.mask_values if hasattr(train_set, 'dataset') else train_set.mask_values
            torch.save(state_dict, Path(model_dir) / f'{model_name}_epoch{epoch}.pth')

    # 5. 训练结束，保存最终模型 (新增逻辑)
    if save_checkpoint:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict['mask_values'] = train_set.dataset.mask_values if hasattr(train_set, 'dataset') else train_set.mask_values
        state_dict['epoch'] = epochs
        state_dict['miou'] = val_metrics['miou']  # 保存最后一个epoch的miou
        final_path = Path(model_dir) / f'{model_name}_last.pth'
        torch.save(state_dict, final_path)
        logging.info(f"Final model saved to {final_path}")

    logging.info(f"Best Val mIoU: {best_miou:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1)
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=1e-5)
    parser.add_argument('--load', '-f', type=str, default=False)
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0)
    parser.add_argument('--images', '-i', type=str, required=True)
    parser.add_argument('--masks', '-m', type=str, required=True)
    parser.add_argument('--val-images', type=str, default=None)
    parser.add_argument('--val-masks', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--mask-suffix', type=str, default='_mask')
    parser.add_argument('--workers', '-w', type=int, default=0)
    
    # 路径与命名控制参数
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--model-dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-name', type=str, default='training_log', help='Filename for the CSV log')
    parser.add_argument('--model-name', type=str, default='checkpoint', help='Prefix name for saved model files')
    
    # 保存策略参数
    parser.add_argument('--save-interval', type=int, default=0, help='Save checkpoint every N epochs (0 to disable)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_img, dir_mask = Path(args.images), Path(args.masks)
    dir_val_img = Path(args.val_images) if args.val_images else None
    dir_val_mask = Path(args.val_masks) if args.val_masks else None

    if (dir_val_img is None) != (dir_val_mask is None):
        raise ValueError('Both --val-images and --val-masks must be provided together')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)

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
            num_workers=args.workers,
            log_dir=Path(args.log_dir),
            model_dir=Path(args.model_dir),
            save_interval=args.save_interval,
            model_name=args.model_name,
            log_name=args.log_name
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('OutOfMemoryError! Enabling checkpointing...')
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
            num_workers=args.workers,
            log_dir=Path(args.log_dir),
            model_dir=Path(args.model_dir),
            save_interval=args.save_interval,
            model_name=args.model_name,
            log_name=args.log_name
        )
