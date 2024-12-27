import os
import logging
import pandas as pd
import numpy as np
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from dataset import ImageDataset
from config import Config
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from transformers import get_cosine_schedule_with_warmup
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_transforms(cfg, is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=cfg.image_size[0], width=cfg.image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(
                mean, std = ([0.8536320017130206, 0.8362727931350286, 0.8301507008641884],
                             [0.23491626922733028, 0.24977155061784578, 0.25436414956228226]),
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=cfg.image_size[0], width=cfg.image_size[1]),
            A.Normalize(
                mean, std = ([0.8536320017130206, 0.8362727931350286, 0.8301507008641884],
                             [0.23491626922733028, 0.24977155061784578, 0.25436414956228226]),
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ])

def create_model(cfg):
    """创建模型并加载本地权重"""
    try:
        logging.info(f"创建模型: {cfg.model_name}")
        model = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=cfg.num_classes
        )
        
        if cfg.pretrained and os.path.exists(cfg.local_weights_path):
            logging.info(f"加载本地权重文件: {cfg.local_weights_path}")
            state_dict = torch.load(cfg.local_weights_path)
            # 移除分类头的权重，因为类别数可能不同
            for key in list(state_dict.keys()):
                if 'head' in key or 'fc' in key or 'classifier' in key:
                    del state_dict[key]
            # 加载其余权重
            model.load_state_dict(state_dict, strict=False)
            logging.info("成功加载本地预训练权重")
        else:
            logging.warning("未找到本地权重文件或未启用预训练，使用随机初始化")
            
        return model
    except Exception as e:
        logging.error(f"创建模型时出错: {str(e)}")
        raise

def test_time_augmentation(model, image, device, cfg):
    """测试时增强"""
    model.eval()
    predictions = []
    
    # 原始图像预测
    with torch.no_grad():
        pred = model(image.to(device)).softmax(1)
        predictions.append(pred)
    
    # 水平翻转
    with torch.no_grad():
        pred = model(torch.flip(image.to(device), dims=[3])).softmax(1)
        predictions.append(pred)
    
    # 不同尺度
    scales = [0.9, 1.1]
    for scale in scales:
        size = (int(cfg.image_size[0] * scale), int(cfg.image_size[1] * scale))
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(cfg.image_size)
        ])
        with torch.no_grad():
            aug_image = transform(image)
            pred = model(aug_image.to(device)).softmax(1)
            predictions.append(pred)
    
    # 平均所有预测
    predictions = torch.stack(predictions).mean(0)
    return predictions

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, cfg, scaler=None):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            # 检查标签范围
            max_label = labels.max().item()
            min_label = labels.min().item()
            if max_label >= cfg.num_classes or min_label < 0:
                logging.error(f"训练集标签范围错误: 最小值={min_label}, 最大值={max_label}")
                logging.error(f"Batch {batch_idx}, Labels: {labels}")
                continue
            
            images = images.to(device)
            if not isinstance(labels, dict):
                labels = labels.long()
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                if cfg.gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if cfg.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # 计算准确率
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            
            # 更新指标
            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg*100:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            if (batch_idx + 1) % cfg.log_interval == 0:
                logging.info(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                           f'Loss: {losses.avg:.4f} '
                           f'Acc: {top1.avg*100:.2f}%')
        except Exception as e:
            logging.error(f"训练时发生错误: {str(e)}")
            continue
    
    return losses.avg, top1.avg

def predict(model, test_loader, device, cfg):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            # 检查是否存在tta_transforms属性并且值大于0
            if hasattr(cfg, 'tta_transforms') and cfg.tta_transforms > 0:
                outputs = test_time_augmentation(model, images, device, cfg)
            else:
                outputs = model(images).softmax(1)
            predictions.extend(outputs.argmax(1).cpu().numpy())
    
    return predictions

def main():
    cfg = Config()
    seed_everything(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 加载数据
    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.test_csv)
    
    # 检查标签分布
    label_counts = train_df['target'].value_counts()
    logging.info("标签分布:")
    for label, count in label_counts.items():
        logging.info(f"类别 {label}: {count} 样本")
    
    # 准备数据集
    train_dataset = ImageDataset(
        root_dir=cfg.train_dir,
        csv_file=train_df,
        transform=get_transforms(cfg, is_train=True),
        is_test=False,
        cfg=cfg
    )
    test_dataset = ImageDataset(
        root_dir=cfg.test_dir,
        csv_file=test_df,
        transform=get_transforms(cfg, is_train=False),
        is_test=True,
        cfg=cfg
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = create_model(cfg)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 学习率调度器
    num_training_steps = len(train_loader) * cfg.epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 混合精度训练
    scaler = GradScaler() if cfg.use_amp else None
    
    # SWA
    if hasattr(cfg, 'use_swa') and cfg.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
    
    # 训练循环
    best_loss = float('inf')
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        logging.info(f'\nEpoch [{epoch+1}/{cfg.epochs}]')
        
        # 训练一个epoch
        loss, acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, device, cfg, scaler
        )
        
        # 更新SWA
        if hasattr(cfg, 'use_swa') and cfg.use_swa and epoch >= cfg.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            os.makedirs(cfg.model_save_path, exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(cfg.model_save_path, 'best_model.pth'))
            logging.info(f'Saved best model with acc: {best_acc*100:.2f}%, loss: {best_loss:.4f}')
    
    # 如果使用了SWA，更新BN统计量
    if hasattr(cfg, 'use_swa') and cfg.use_swa:
        logging.info('Updating batch normalization statistics for SWA')
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
        model = swa_model
    
    # 加载最佳模型进行预测
    model.load_state_dict(torch.load(os.path.join(cfg.model_save_path, 'best_model.pth')))
    predictions = predict(model, test_loader, device, cfg)
    
    # 保存预测结果
    test_df['label'] = predictions
    test_df.to_csv(cfg.submission_path, index=False)
    logging.info(f'Predictions saved to {cfg.submission_path}')

if __name__ == '__main__':
    main()
