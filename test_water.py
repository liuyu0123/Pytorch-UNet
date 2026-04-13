import argparse
import logging
import time
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.serialization import safe_globals


def compute_metrics(pred_mask, true_mask, num_classes):
    """计算分割指标：Precision, Recall, F1, IoU（宏平均）"""
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


def create_dataset(dir_img, dir_mask, img_scale, mask_suffix):
    """创建数据集"""
    try:
        return CarvanaDataset(dir_img, dir_mask, img_scale, mask_suffix)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(dir_img, dir_mask, img_scale, mask_suffix)


@torch.no_grad()
def test_model(model, dataloader, device, num_classes, amp=False):
    """测试模型，返回各项指标"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    inference_times = []
    
    criterion = torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCEWithLogitsLoss()
    
    for batch in dataloader:
        images = batch['image'].to(device=device, dtype=torch.float32, 
                                   memory_format=torch.channels_last)
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
        
        # 计算 loss
        if num_classes == 1:
            loss = criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), 
                            true_masks.float(), multiclass=False)
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
    metrics['total_images'] = len(all_preds)
    
    return metrics


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'total_params': total_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def save_results(metrics, model_info, save_path, args):
    """保存测试结果到 CSV"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建结果字典
    result = {
        'model_path': args.model,
        'test_images': args.images,
        'test_masks': args.masks,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'num_classes': args.classes,
        'img_scale': args.scale,
        'amp': args.amp,
        'test_loss': f"{metrics['loss']:.6f}",
        'test_precision': f"{metrics['precision']:.6f}",
        'test_recall': f"{metrics['recall']:.6f}",
        'test_f1': f"{metrics['f1']:.6f}",
        'test_miou': f"{metrics['miou']:.6f}",
        'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
        'fps': f"{metrics['fps']:.2f}",
        'total_images': metrics['total_images'],
    }
    
    # 写入 CSV
    header = list(result.keys())
    file_exists = save_path.exists()
    
    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    logging.info(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test the UNet on test set')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained model .pth file')
    parser.add_argument('--images', '-i', type=str, required=True,
                        help='Path to the test images directory')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to the test masks directory')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('--mask-suffix', type=str, default='_mask',
                        help='Suffix for mask files')
    parser.add_argument('--classes', '-c', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use mixed precision')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--workers', '-w', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--output', '-o', type=str, default='./logs/test_results.csv',
                        help='Path to save test results CSV')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # 加载模型
    logging.info(f'Loading model from {args.model}')
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    
    # with safe_globals([np.core.multiarray.scalar]):
    #     state_dict = torch.load(args.model, map_location=device, weights_only=True)
    state_dict = torch.load(args.model, map_location=device, weights_only=False)
    # 移除不需要的键
    keys_to_remove = ['mask_values', 'epoch', 'miou', 'metrics']
    for key in keys_to_remove:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict)
    model.to(device=device)
    
    # 获取模型信息
    model_info = get_model_info(model)
    logging.info(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 准备测试数据
    test_set = create_dataset(Path(args.images), Path(args.masks), args.scale, args.mask_suffix)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True if device.type == 'cuda' else False)
    
    logging.info(f'Test set size: {len(test_set)} images')
    
    # 测试
    logging.info('Starting evaluation...')
    metrics = test_model(model, test_loader, device, args.classes, args.amp)
    
    # 打印结果
    logging.info('=' * 50)
    logging.info('TEST RESULTS')
    logging.info('=' * 50)
    logging.info(f"Loss:           {metrics['loss']:.6f}")
    logging.info(f"Precision:      {metrics['precision']:.6f}")
    logging.info(f"Recall:         {metrics['recall']:.6f}")
    logging.info(f"F1-Score:       {metrics['f1']:.6f}")
    logging.info(f"mIoU:           {metrics['miou']:.6f}")
    logging.info(f"Inference Time: {metrics['inference_time_ms']:.4f} ms")
    logging.info(f"FPS:            {metrics['fps']:.2f}")
    logging.info('=' * 50)
    
    # 保存结果
    save_results(metrics, model_info, args.output, args)
    
    # 同时保存详细文本报告
    report_path = Path(args.output).parent / 'test_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Images: {args.images}\n")
        f.write(f"Test Masks: {args.masks}\n")
        f.write(f"\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Number of Classes: {args.classes}\n")
        f.write(f"\n")
        f.write(f"Test Set Size: {metrics['total_images']} images\n")
        f.write(f"\n")
        f.write(f"Loss:           {metrics['loss']:.6f}\n")
        f.write(f"Precision:      {metrics['precision']:.6f}\n")
        f.write(f"Recall:         {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.6f}\n")
        f.write(f"mIoU:           {metrics['miou']:.6f}\n")
        f.write(f"Inference Time: {metrics['inference_time_ms']:.4f} ms\n")
        f.write(f"FPS:            {metrics['fps']:.2f}\n")
    
    logging.info(f"Text report saved to {report_path}")


if __name__ == '__main__':
    main()