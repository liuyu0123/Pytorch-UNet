import argparse
import logging
import os
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Input image file or folder')
    parser.add_argument('--output', '-o', metavar='OUTPUT', 
                        help='Output folder for overlay images and CSV results')
    parser.add_argument('--ground-truth', '-g', metavar='GT',
                        help='Ground truth mask file or folder for evaluation')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--alpha', '-a', type=float, default=0.4, 
                        help='Transparency of red water overlay (0.0-1.0), default 0.4')
    
    return parser.parse_args()


def calculate_metrics(pred_mask, gt_mask):
    """计算Precision、Recall、F1、mIoU"""
    pred = (pred_mask > 0).astype(np.uint8).flatten()
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    
    return precision, recall, f1, miou


def save_overlay(image_path, mask, output_path, alpha=0.4):
    """保存原图与红色水mask的叠加图"""
    # 加载原图并转为RGBA
    image = Image.open(image_path).convert('RGBA')
    
    # 创建红色蒙版 (水区域为红色半透明，非水透明)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask == 1] = [255, 0, 0, int(255 * alpha)]  # 红色半透明
    
    overlay_img = Image.fromarray(overlay, mode='RGBA')
    
    # 叠加并保存为RGB
    blended = Image.alpha_composite(image, overlay_img)
    blended.convert('RGB').save(output_path)
    logging.info(f'Saved overlay: {output_path}')


def get_image_files(path):
    """获取路径下的所有图片文件"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    p = Path(path)
    
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in exts]
        return sorted(files)
    else:
        raise ValueError(f"Invalid path: {path}")


def find_gt_file(input_file, gt_path):
    """匹配对应的真值文件"""
    if not gt_path:
        return None
    
    gt_path = Path(gt_path)
    stem = input_file.stem
    
    if gt_path.is_file():
        return gt_path
    elif gt_path.is_dir():
        # 尝试相同文件名，不同扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
            candidate = gt_path / (stem + ext)
            if candidate.exists():
                return candidate
    return None


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # 初始化模型
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device=device)
    
    # 加载模型权重（兼容完整检查点格式）
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        mask_values = checkpoint.pop('mask_values', None)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 过滤训练元数据，只保留权重
            ignore_keys = ['epoch', 'miou', 'optimizer_state_dict', 'scheduler_state_dict',
                          'global_step', 'best_score', 'best_epoch', 'scaler']
            state_dict = {k: v for k, v in checkpoint.items() 
                         if k not in ignore_keys and isinstance(v, torch.Tensor)}
    else:
        state_dict = checkpoint
        mask_values = None
    
    if mask_values is None:
        mask_values = [0, 1]  # 默认二分类
    
    net.load_state_dict(state_dict)
    logging.info(f'Model loaded: {args.model}')
    
    # 获取输入文件列表并判断输入类型
    input_path = Path(args.input)
    input_files = get_image_files(args.input)
    
    # 确定CSV基础名称：文件用文件名（无扩展名），文件夹用文件夹名
    if input_path.is_file():
        csv_base_name = input_path.stem
    else:
        csv_base_name = input_path.name
    
    logging.info(f'Found {len(input_files)} images to process')
    
    # 准备输出目录
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估数据收集
    results = []
    has_gt = args.ground_truth is not None
    
    # 批量推理
    for input_file in input_files:
        logging.info(f'Processing: {input_file.name}')
        
        # 加载并推理
        img = Image.open(input_file)
        mask = predict_img(net, img, device, args.scale, args.mask_threshold)
        
        # 保存叠加图
        if output_dir:
            out_file = output_dir / f"{input_file.stem}_overlay.png"
            save_overlay(input_file, mask, out_file, args.alpha)
        
        # 评估指标计算
        if has_gt:
            gt_file = find_gt_file(input_file, args.ground_truth)
            if gt_file and gt_file.exists():
                try:
                    # 加载真值并标准化
                    gt_img = Image.open(gt_file)
                    gt_arr = np.array(gt_img.convert('L'))  # 转为灰度
                    
                    # 处理尺寸不匹配
                    if gt_arr.shape != mask.shape:
                        gt_arr = np.array(Image.fromarray(gt_arr).resize(
                            (mask.shape[1], mask.shape[0]), Image.NEAREST))
                    
                    # 计算指标
                    p, r, f1, miou = calculate_metrics(mask, gt_arr)
                    results.append({
                        'filename': input_file.name,
                        'precision': round(p, 4),
                        'recall': round(r, 4),
                        'f1': round(f1, 4),
                        'miou': round(miou, 4)
                    })
                    logging.info(f'  Metrics: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, mIoU={miou:.4f}')
                except Exception as e:
                    logging.error(f'  Error evaluating {input_file.name}: {e}')
            else:
                logging.warning(f'  No GT found for {input_file.name}')
    
    # 生成评估报告
    if has_gt and results:
        # 计算平均值
        avg_p = np.mean([x['precision'] for x in results])
        avg_r = np.mean([x['recall'] for x in results])
        avg_f1 = np.mean([x['f1'] for x in results])
        avg_miou = np.mean([x['miou'] for x in results])
        
        # 添加平均行
        results.append({
            'filename': 'AVERAGE',
            'precision': round(avg_p, 4),
            'recall': round(avg_r, 4),
            'f1': round(avg_f1, 4),
            'miou': round(avg_miou, 4)
        })
        
        if output_dir:
            # 保存CSV，文件名与输入一致（如：image.csv 或 foldername.csv）
            csv_filename = f"{csv_base_name}.csv"
            csv_path = output_dir / csv_filename
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'precision', 'recall', 'f1', 'miou'])
                writer.writeheader()
                writer.writerows(results)
            logging.info(f'\nEvaluation CSV saved: {csv_path}')
        else:
            # 终端打印表格
            print('\n' + '='*70)
            print(f'Evaluation Results: {csv_base_name}')
            print(f'{"Filename":<35} {"Prec":>8} {"Rec":>8} {"F1":>8} {"mIoU":>8}')
            print('-'*70)
            for r in results[:-1]:  # 除平均外
                print(f"{r['filename']:<35} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['miou']:>8.4f}")
            print('-'*70)
            print(f"{'AVERAGE':<35} {avg_p:>8.4f} {avg_r:>8.4f} {avg_f1:>8.4f} {avg_miou:>8.4f}")
            print('='*70)
    elif has_gt:
        logging.warning('No evaluation results generated (no matching GT files)')
    
    logging.info('All done!')