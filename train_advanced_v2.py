import os
import logging
import pandas as pd
import numpy as np
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
from dataset import ImageDataset
from config import Config
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
import copy
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
        logging.FileHandler('training_advanced.log'),
        logging.StreamHandler()
    ]
)

# 设置随机种子
def seed_everything(seed):
    """设置所有随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter:
    """跟踪指标的平均值和当前值"""
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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """获取带有warmup的余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, cfg, scaler=None):
    """训练一个epoch"""
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
            
            # 将数据移动到设备上
            images = images.to(device)
            if not isinstance(labels, dict):  # 添加类型检查
                labels = labels.long()  # 确保标签是长整型
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用混合精度训练
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
            logging.error(f"Batch {batch_idx}")
            logging.error(f"Labels shape: {labels.shape}")
            logging.error(f"Outputs shape: {outputs.shape if 'outputs' in locals() else 'not created'}")
            continue
    
    if losses.count == 0:
        logging.error("训练过程中所有样本都被跳过")
        return float('inf'), 0.0
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', ncols=100):
            try:
                images = images.to(device)
                if not isinstance(labels, dict):
                    labels = labels.long()
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 计算准确率
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                
                # 更新指标
                losses.update(loss.item(), images.size(0))
                top1.update(acc.item(), images.size(0))
                
            except Exception as e:
                logging.error(f"验证时发生错误: {str(e)}")
                continue
    
    if losses.count == 0:
        logging.error("验证过程中所有样本都被跳过")
        return float('inf'), 0.0
    
    return losses.avg, top1.avg

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

def predict(models, test_loader, device, cfg):
    """使用模型集成进行预测"""
    if not models:  # 如果没有训练好的模型
        logging.error("没有可用的模型进行预测")
        return [], []
        
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting', ncols=100)
        for images, img_names in pbar:
            batch_preds = []
            
            # 对每个模型进行预测
            for model in models:
                if cfg.tta_transforms > 0:
                    pred = test_time_augmentation(model, images, device, cfg)
                else:
                    pred = model(images.to(device)).softmax(1)
                batch_preds.append(pred)
            
            # 合并所有模型的预测
            if batch_preds:  # 确保有预测结果
                batch_preds = torch.stack(batch_preds).mean(0)  # 使用平均而不是加权和
                _, predicted = batch_preds.max(1)
                
                predictions.extend(predicted.cpu().numpy())
                image_ids.extend(img_names)
            
            pbar.set_postfix({'Processed': len(predictions)})
    
    return image_ids, predictions

def prepare_loaders(train_df, train_idx, val_idx, train_dir, batch_size, num_workers=4):
    """准备数据加载器"""
    # 分割数据
    train_subset = train_df.iloc[train_idx].reset_index(drop=True)
    val_subset = train_df.iloc[val_idx].reset_index(drop=True)
    
    # 创建数据集
    train_dataset = ImageDataset(train_dir, train_subset, transform=get_train_transforms())
    val_dataset = ImageDataset(train_dir, val_subset, transform=get_val_transforms())
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_train_transforms():
    """获取训练数据增强"""
    return A.Compose([
        A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    """获取验证数据增强"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ])

def create_model(cfg):
    """创建模型"""
    try:
        logging.info(f"创建模型: {cfg.model_name}")
        # 尝试直接创建模型
        try:
            model = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=cfg.num_classes
            )
            logging.info("成功创建模型")
            return model
        except RuntimeError as e:
            if "Failed to download weights" in str(e):
                # 如果下载失败，尝试使用本地缓存
                logging.warning("无法下载预训练权重，尝试使用本地缓存...")
                model = timm.create_model(
                    cfg.model_name,
                    pretrained=False,
                    num_classes=cfg.num_classes
                )
                logging.info("使用未预训练的模型继续")
                return model
            raise
    except Exception as e:
        logging.error(f"创建模型时出错: {str(e)}")
        raise

def create_optimizer(model, cfg):
    """创建优化器"""
    return optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

def create_scheduler(optimizer, num_training_steps, cfg):
    """创建学习率调度器"""
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def train_k_fold(cfg, device):
    """K折交叉验证训练"""
    try:
        # 读取训练数据
        train_df = pd.read_csv(cfg.train_csv)
        logging.info(f"加载了 {len(train_df)} 条训练数据")
        
        # 检查标签分布
        label_counts = train_df['target'].value_counts()
        logging.info("标签分布:")
        for label, count in label_counts.items():
            logging.info(f"类别 {label}: {count} 样本")
        
        # 创建数据增强
        train_transform = A.Compose([
            A.RandomResizedCrop(height=cfg.image_size[0], width=cfg.image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
            ], p=0.2),  # 减少噪声增强的概率
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 减小色彩增强的强度
            A.Normalize(),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(height=cfg.image_size[0], width=cfg.image_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # 初始化KFold
        kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
        
        # 存储所有训练好的模型
        models = []
        
        # K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df, train_df['target'])):
            logging.info(f'开始训练第 {fold + 1} 折 (共 {cfg.num_folds} 折)...')
            
            try:
                # 准备数据集
                train_fold = train_df.iloc[train_idx].reset_index(drop=True)
                val_fold = train_df.iloc[val_idx].reset_index(drop=True)
                
                logging.info(f"训练集大小: {len(train_fold)}, 验证集大小: {len(val_fold)}")
                
                train_dataset = ImageDataset(cfg.train_dir, train_fold, transform=train_transform, cfg=cfg)
                val_dataset = ImageDataset(cfg.train_dir, val_fold, transform=val_transform, cfg=cfg)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=cfg.batch_size * 2,  # 验证时用更大的batch_size
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=True
                )
                
                # 创建模型
                model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
                model = model.to(device)
                
                # 创建SWA模型
                swa_model = AveragedModel(model)
                
                # 创建优化器和调度器
                optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                criterion = nn.CrossEntropyLoss()
                
                # 计算总步数和预热步数
                num_training_steps = len(train_loader) * cfg.epochs
                num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
                
                # 创建学习率调度器
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
                
                # 创建SWA调度器
                swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
                
                # 创建梯度缩放器
                scaler = GradScaler() if cfg.use_amp else None
                
                # 记录最佳验证准确率
                best_val_acc = 0.0
                best_model = None
                patience_counter = 0
                
                # 训练循环
                for epoch in range(cfg.epochs):
                    try:
                        logging.info(f'Epoch {epoch + 1}/{cfg.epochs}')
                        
                        # 训练一个epoch
                        train_loss, train_acc = train_one_epoch(
                            model, train_loader, criterion, optimizer,
                            scheduler, device, cfg, scaler
                        )
                        
                        # 验证
                        val_loss, val_acc = validate(model, val_loader, criterion, device)
                        
                        logging.info(f'Fold {fold + 1}, Epoch {epoch + 1}: '
                                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, '
                                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
                        
                        # 更新最佳模型
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model = copy.deepcopy(model)
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        # 早停
                        if patience_counter >= cfg.patience:
                            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                            break
                        
                        # 更新SWA模型
                        if epoch >= cfg.swa_start:
                            swa_model.update_parameters(model)
                            swa_scheduler.step()
                            
                    except Exception as e:
                        logging.error(f"训练epoch {epoch + 1}时发生错误: {str(e)}")
                        continue
                
                # 保存最佳模型
                if best_model is not None:
                    models.append(best_model)
                    model_path = os.path.join(cfg.model_save_path, f'model_fold_{fold}.pth')
                    torch.save(best_model.state_dict(), model_path)
                    logging.info(f"保存模型到: {model_path}")
                
                logging.info(f'Fold {fold + 1} 最佳验证准确率: {best_val_acc*100:.2f}%')
                
            except Exception as e:
                logging.error(f"训练第 {fold + 1} 折时发生错误: {str(e)}")
                continue
        
        return models
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise

def main():
    """主训练流程"""
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'使用设备: {device}')
        
        # 创建配置
        cfg = Config()
        
        # 确保模型保存目录存在
        os.makedirs(cfg.model_save_path, exist_ok=True)
        
        # 读取数据
        test_df = pd.read_csv(cfg.test_csv)
        
        # 创建测试数据集和加载器
        test_transform = A.Compose([
            A.Resize(height=cfg.image_size[0], width=cfg.image_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        test_dataset = ImageDataset(cfg.test_dir, test_df, transform=test_transform, is_test=True, cfg=cfg)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        # 训练模型
        models = train_k_fold(cfg, device)
        
        if not models:
            logging.error("训练失败，没有可用的模型")
            return
            
        logging.info("开始预测...")
        # 进行预测
        image_ids, predictions = predict(models, test_loader, device, cfg)
        
        if not predictions:
            logging.error("预测失败，没有预测结果")
            return
            
        # 创建提交文件
        submission = pd.DataFrame({
            'id': image_ids,
            'target': predictions
        })
        
        # 保存预测结果
        submission.to_csv(cfg.submission_path, index=False)
        logging.info(f"预测结果已保存到: {cfg.submission_path}")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
