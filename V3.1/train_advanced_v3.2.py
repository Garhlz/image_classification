import os
import random
import logging
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR

import timm
from timm.models import create_model as timm_create_model

from config import Config, get_config
from dataset import ImageDataset
from utils import ModelEMA, FocalLoss, save_checkpoint, load_checkpoint
from transforms import get_train_transforms, get_val_transforms

def seed_everything(seed):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class LearningRateMonitor:
    def __init__(self):
        self.lrs = []
        self.steps = []
        
    def update(self, lr, step):
        self.lrs.append(lr)
        self.steps.append(step)
    
    def plot(self, save_path='learning_rate.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.lrs)
        plt.xlabel('训练步数')
        plt.ylabel('学习率')
        plt.title('学习率变化曲线')
        plt.grid(True)
        plt.yscale('log')
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path)
        plt.close()
        logging.info(f'学习率变化曲线已保存至: {save_path}')

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, cfg, scaler=None, ema=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    scores = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{cfg.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 计算准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        # 更新统计信息
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        scores.update(acc.item(), batch_size)
        
        # 反向传播和优化
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.gradient_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            
        # 更新EMA
        if ema is not None:
            ema.update()
            
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{scores.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 定期记录
        if batch_idx % cfg.log_interval == 0:
            logging.info(
                f'Epoch [{epoch+1}/{cfg.epochs}][{batch_idx}/{len(train_loader)}] '
                f'Loss: {losses.avg:.4f} Acc: {scores.avg:.4f} '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
    
    return losses.avg, scores.avg

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean()
            
            # 更新统计信息
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            scores.update(acc.item(), batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{scores.avg:.4f}'
            })
    
    return losses.avg, scores.avg

def prepare_loaders(df, train_idx, val_idx, image_dir, batch_size, num_workers, cfg):
    """准备数据加载器"""
    try:
        # 创建训练集和验证集的DataFrame
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        logging.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
        
        # 统一标签列名
        if 'target' in train_df.columns:
            train_df = train_df.rename(columns={'target': 'label'})
            val_df = val_df.rename(columns={'target': 'label'})
        
        # 创建数据集
        train_dataset = ImageDataset(
            root_dir=image_dir,
            df=train_df,
            transform=get_train_transforms(cfg.image_size),
            cfg=cfg
        )
        
        val_dataset = ImageDataset(
            root_dir=image_dir,
            df=val_df,
            transform=get_val_transforms(cfg.image_size),
            cfg=cfg
        )
        
        # 创建采样器
        if cfg.use_sampler:
            try:
                # 计算类别权重
                class_counts = Counter(train_df['label'].values)
                total_samples = len(train_df)
                class_weights = {label: 1.0 / count for label, count in class_counts.items()}
                
                # 归一化权重
                weight_sum = sum(class_weights.values())
                class_weights = {label: weight / weight_sum * len(class_weights) 
                               for label, weight in class_weights.items()}
                
                # 为每个样本分配权重
                weights = [class_weights[label] for label in train_df['label'].values]
                weights = torch.DoubleTensor(weights)
                
                # 创建采样器
                sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(train_df),  # 确保采样数量等于数据集大小
                    replacement=True  # 允许重复采样
                )
                logging.info("成功创建加权采样器")
                shuffle = False
            except Exception as e:
                logging.error(f"创建采样器失败: {str(e)}")
                sampler = None
                shuffle = True
        else:
            sampler = None
            shuffle = True
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,  # 根据是否使用采样器决定是否shuffle
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,  # 设置worker的随机种子
            generator=torch.Generator().manual_seed(cfg.seed)  # 设置采样器的随机种子
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,  # 设置worker的随机种子
            generator=torch.Generator().manual_seed(cfg.seed)  # 设置采样器的随机种子
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logging.error(f"创建数据加载器时发生错误: {str(e)}")
        raise

def seed_worker(worker_id):
    """为DataLoader的worker设置随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_model(cfg):
    """
    创建模型
    Args:
        cfg: 配置对象，包含模型配置信息
    Returns:
        model: 创建的模型
    """
    try:
        logging.info(f"创建模型: {cfg.model_name}")
        
        # 检查必要的配置
        if not hasattr(cfg, 'model_name') or not cfg.model_name:
            raise ValueError("未指定模型名称")
        if not hasattr(cfg, 'num_classes') or cfg.num_classes <= 0:
            raise ValueError("无效的类别数量")
            
        # 创建未初始化的模型
        model = timm_create_model(
            cfg.model_name,
            pretrained=False,  # 先创建未预训练的模型
            num_classes=cfg.num_classes,
            drop_rate=getattr(cfg, 'dropout_rate', 0.0),  # 可选的dropout率
            drop_path_rate=getattr(cfg, 'drop_path_rate', 0.0)  # 可选的drop path率
        )
        
        # 如果配置了使用预训练权重
        if getattr(cfg, 'pretrained', False):
            # 优先使用本地权重
            if hasattr(cfg, 'pretrained_path') and cfg.pretrained_path:
                if os.path.exists(cfg.pretrained_path):
                    logging.info(f"加载本地预训练权重: {cfg.pretrained_path}")
                    try:
                        checkpoint = torch.load(cfg.pretrained_path, map_location='cpu')
                        
                        # 处理不同格式的checkpoint
                        if isinstance(checkpoint, dict):
                            if 'state_dict' in checkpoint:
                                checkpoint = checkpoint['state_dict']
                            elif 'model' in checkpoint:
                                checkpoint = checkpoint['model']
                        
                        # 移除分类器权重
                        classifier_keywords = ['head', 'fc', 'classifier', 'predictions']
                        for k in list(checkpoint.keys()):
                            if any(keyword in k for keyword in classifier_keywords):
                                logging.info(f"移除分类器权重: {k}")
                                del checkpoint[k]
                        
                        # 处理权重键名不匹配的情况
                        if hasattr(cfg, 'remove_prefix') and cfg.remove_prefix:
                            checkpoint = {k.replace(cfg.remove_prefix, ''): v for k, v in checkpoint.items()}
                        
                        # 加载权重
                        msg = model.load_state_dict(checkpoint, strict=False)
                        logging.info(f"成功加载的权重: {msg.missing_keys}")
                        if msg.unexpected_keys:
                            logging.warning(f"未使用的权重: {msg.unexpected_keys}")
                            
                    except Exception as e:
                        logging.error(f"加载预训练权重时出错: {str(e)}")
                        if not getattr(cfg, 'ignore_load_errors', False):
                            raise
                else:
                    logging.warning(f"预训练权重文件不存在: {cfg.pretrained_path}")
            
            # 如果没有本地权重或加载失败，尝试使用预训练权重
            elif getattr(cfg, 'use_pretrained', True):
                logging.info("使用预训练权重初始化模型")
                model = timm_create_model(
                    cfg.model_name,
                    pretrained=True,
                    num_classes=cfg.num_classes,
                    drop_rate=getattr(cfg, 'dropout_rate', 0.0),
                    drop_path_rate=getattr(cfg, 'drop_path_rate', 0.0)
                )
        
        # 冻结特定层（如果配置了）
        if hasattr(cfg, 'freeze_layers') and cfg.freeze_layers:
            for name, param in model.named_parameters():
                if any(layer in name for layer in cfg.freeze_layers):
                    param.requires_grad = False
                    logging.info(f"冻结层: {name}")
        
        # 打印模型信息
        logging.info(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
        
    except Exception as e:
        logging.error(f"创建模型时出错: {str(e)}")
        raise

def create_optimizer(model, cfg):
    """创建优化器"""
    return torch.optim.AdamW(
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

def compute_sample_weights(labels):
    """计算样本权重用于处理类别不平衡"""
    try:
        # 计算类别计数
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)
        
        # 计算每个类别的权重
        weights = {label: total_samples / (num_classes * count) 
                  for label, count in label_counts.items()}
        
        # 为每个样本分配权重
        sample_weights = [weights[label] for label in labels]
        
        # 归一化权重
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        # 转换为PyTorch张量
        return torch.DoubleTensor(sample_weights)
        
    except Exception as e:
        logging.error(f"计算样本权重时出错: {str(e)}")
        raise

def create_weighted_sampler(labels):
    """创建加权采样器"""
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler

def create_criterion(cfg):
    """创建损失函数"""
    if cfg.use_weighted_loss:
        # 计算类别权重
        df = pd.read_csv(cfg.train_csv)
        class_counts = Counter(df[cfg.target_col])
        total_samples = len(df)
        weights = torch.FloatTensor([
            total_samples / (len(class_counts) * class_counts[i])
            for i in range(cfg.num_classes)
        ]).to(cfg.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif cfg.use_focal_loss:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

def setup_logging(log_file=None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler()
        ]
    )

def main():
    """主函数"""
    try:
        # 获取配置
        cfg = get_config()
        
        # 设置随机种子
        seed_everything(cfg.seed)
        
        # 设置设备
        device = torch.device(cfg.device)
        
        # 创建输出目录
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        # 设置日志
        setup_logging(os.path.join(cfg.output_dir, 'train.log'))
        
        # 读取数据
        df = pd.read_csv(cfg.train_csv)
        
        # 处理类别不平衡问题
        label_counts = df[cfg.target_col].value_counts()
        min_samples_per_class = 2  # 每个类别至少需要2个样本
        
        # 对样本数量少于2的类别进行过采样
        additional_samples = []
        for label, count in label_counts.items():
            if count < min_samples_per_class:
                class_df = df[df[cfg.target_col] == label]
                samples_needed = min_samples_per_class - count
                additional_df = class_df.sample(n=samples_needed, replace=True, random_state=cfg.seed)
                additional_samples.append(additional_df)
                
        if additional_samples:
            df = pd.concat([df] + additional_samples, ignore_index=True)
            logging.info("对小类别进行了过采样处理")
            
        # 创建训练集和验证集索引
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=cfg.val_size,
            random_state=cfg.seed,
            stratify=df[cfg.target_col] if cfg.stratify else None
        )
        
        # 准备数据加载器
        train_loader, val_loader = prepare_loaders(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            image_dir=cfg.train_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            cfg=cfg
        )
        
        # 创建模型
        model = create_model(cfg).to(device)
        
        # 创建优化器
        optimizer = create_optimizer(model, cfg)
        
        # 创建损失函数
        criterion = create_criterion(cfg)
        
        # 创建学习率调度器
        scheduler = create_scheduler(optimizer, len(train_loader), cfg)
        
        # 创建混合精度训练的scaler
        scaler = torch.cuda.amp.GradScaler('cuda') if cfg.use_amp else None
        
        # 创建EMA
        if cfg.use_ema:
            ema = ModelEMA(model, cfg.ema_decay)
        else:
            ema = None
        
        # 训练模型
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(cfg.epochs):
            logging.info(f"\nEpoch {epoch + 1}/{cfg.epochs}")
            
            # 训练一个epoch
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                scaler=scaler,
                ema=ema,
                cfg=cfg
            )
            
            # 验证
            val_loss, val_acc = validate(
                model=ema.module if ema else model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                cfg=cfg
            )
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                save_checkpoint(
                    model=ema.module if ema else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    cfg=cfg,
                    filename='best_model.pth'
                )
                
            # 记录训练信息
            logging.info(f"Epoch {epoch + 1} - "
                        f"Train Loss: {train_loss:.4f} - "
                        f"Train Acc: {train_acc:.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Val Acc: {val_acc:.4f}")
            
            if epoch + 1 - best_epoch >= cfg.early_stopping:
                logging.info(f"Early stopping triggered. Best epoch: {best_epoch}")
                break
                
        logging.info(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
