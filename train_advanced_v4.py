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
from dataset import ImageDataset, get_train_transforms, get_valid_transforms, MixUpCutMixDataset
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

class DeepSupervisionModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        
        # 获取中间特征图的通道数
        if hasattr(base_model, 'stages'):  # ConvNeXt
            channels = []
            for stage in base_model.stages[1:]:  # 跳过第一个stage
                # 获取stage的最后一个block的输出通道数
                if hasattr(stage, 'blocks'):
                    last_block = stage.blocks[-1]
                    if hasattr(last_block, 'conv_dw'):
                        channels.append(last_block.conv_dw.out_channels)
                    elif hasattr(last_block, 'dwconv'):
                        channels.append(last_block.dwconv.out_channels)
                    else:
                        raise NotImplementedError("无法获取block的输出通道数")
        else:
            raise NotImplementedError("Deep supervision only implemented for ConvNeXt")
            
        # 辅助分类器
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, num_classes)
            ) for dim in channels
        ])
        
    def forward(self, x):
        # 存储中间特征图
        features = []
        
        # 第一个stage
        x = self.base_model.stem(x)
        
        # 后续stages
        for i, stage in enumerate(self.base_model.stages):
            x = stage(x)
            if i > 0:  # 跳过第一个stage
                features.append(x)
        
        # 主分类器输出
        x = self.base_model.head.global_pool(x)
        x = self.base_model.head.norm(x)
        x = self.base_model.head.flatten(x)
        main_out = self.base_model.head.fc(x)
        
        # 辅助分类器输出
        aux_outs = []
        for feature, classifier in zip(features, self.aux_classifiers):
            aux_out = classifier(feature)
            aux_outs.append(aux_out)
            
        return [main_out] + aux_outs

class DeepSupervisionLoss(nn.Module):
    def __init__(self, main_weight=0.6):
        super().__init__()
        self.main_weight = main_weight
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, outputs, target):
        if not isinstance(outputs, list):
            if len(target.shape) == 2:  # one-hot编码的标签
                target = target.argmax(dim=1)
            return self.criterion(outputs, target)
            
        # 主分类器损失
        if len(target.shape) == 2:  # one-hot编码的标签
            target = target.argmax(dim=1)
        main_loss = self.criterion(outputs[0], target)
        
        # 辅助分类器损失
        aux_losses = []
        aux_weight = (1 - self.main_weight) / (len(outputs) - 1)
        for aux_output in outputs[1:]:
            aux_losses.append(self.criterion(aux_output, target))
        
        # 总损失
        total_loss = self.main_weight * main_loss
        for aux_loss in aux_losses:
            total_loss += aux_weight * aux_loss
            
        return total_loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """获取带有warmup的余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, cfg, scaler=None, ema=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            # 检查标签范围
            if isinstance(labels, torch.Tensor) and len(labels.shape) == 1:  # 处理普通标签
                max_label = labels.max().item()
                min_label = labels.min().item()
                if max_label >= cfg.num_classes or min_label < 0:
                    logging.error(f"标签范围错误: min={min_label}, max={max_label}")
                    continue
            
            # 将数据移动到设备上
            images = images.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if isinstance(outputs, list) and not isinstance(criterion, DeepSupervisionLoss):
                        outputs = outputs[0]  # 如果不是深度监督，只使用主分类器输出
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    if cfg.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                        
                    if ema is not None:
                        ema.update()
            else:
                outputs = model(images)
                if isinstance(outputs, list) and not isinstance(criterion, DeepSupervisionLoss):
                    outputs = outputs[0]  # 如果不是深度监督，只使用主分类器输出
                loss = criterion(outputs, labels)
                
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                        
                    if ema is not None:
                        ema.update()
            
            # 计算准确率（仅在非mixup/cutmix时计算）
            if isinstance(labels, torch.Tensor) and len(labels.shape) == 1:
                if isinstance(outputs, list):
                    outputs = outputs[0]  # 使用主分类器的输出
                _, predicted = outputs.max(1)
                acc = predicted.eq(labels).float().mean()
                top1.update(acc.item(), images.size(0))
            
            # 更新损失
            losses.update(loss.item(), images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg*100:.2f}%' if top1.count > 0 else 'N/A',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        except Exception as e:
            logging.error(f"训练批次 {batch_idx} 出错: {str(e)}")
            continue
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 如果是深度监督模型，只使用主分类器的输出
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # 计算损失和准确率
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            
            val_loss.update(loss.item(), images.size(0))
            val_acc.update(acc.item(), images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{val_loss.avg:.4f}',
                'Acc': f'{val_acc.avg:.2%}'
            })
    
    return val_loss.avg, val_acc.avg

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
    val_dataset = ImageDataset(train_dir, val_subset, transform=get_valid_transforms())
    
    # 使用MixUpCutMixDataset包装训练数据集
    train_dataset = MixUpCutMixDataset(
        dataset=train_dataset,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=0.5,
        num_classes=44  # 根据实际类别数设置
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_fold(train_loader, valid_loader, model, criterion, optimizer, scheduler, device, cfg, fold):
    """训练单个fold"""
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    
    # 创建EMA
    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None
    
    # 创建梯度缩放器
    scaler = GradScaler() if cfg.use_amp else None
    
    for epoch in range(cfg.epochs):
        # 训练阶段
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            scaler=scaler,
            ema=ema
        )
        
        # 验证阶段
        if ema:
            ema.apply_shadow()
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        if ema:
            ema.restore()
        
        # 记录最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ema:
                ema.apply_shadow()
            best_model = copy.deepcopy(model)
            if ema:
                ema.restore()
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'val_acc': best_val_acc,
                'fold': fold,
            }, f'{cfg.model_save_path}/best_model_fold{fold}.pth')
        else:
            patience_counter += 1
        
        logging.info(f'Fold {fold}, Epoch {epoch+1}/{cfg.epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logging.info(f'Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}')
        logging.info(f'Best Valid Acc: {best_val_acc:.4f}')
        
        if patience_counter >= cfg.patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return best_model, best_val_acc

def create_model(cfg):
    """创建模型"""
    try:
        logging.info(f"创建模型: {cfg.model_name}")
        base_model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes
        )
        
        if cfg.use_deep_supervision:
            model = DeepSupervisionModel(base_model, cfg.num_classes)
        else:
            model = base_model
            
        return model
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
    if cfg.use_cosine_schedule:
        # 计算预热步数
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=cfg.min_lr
        )
    else:
        return None

def oversample_minority_classes(df, target_col='target', min_samples=20):
    """对小类别进行过采样
    Args:
        df: 数据框
        target_col: 目标列名
        min_samples: 每个类别的最小样本数
    Returns:
        过采样后的数据框
    """
    # 获取类别分布
    class_counts = df[target_col].value_counts()
    
    # 找出需要过采样的类别
    classes_to_oversample = class_counts[class_counts < min_samples]
    
    if len(classes_to_oversample) == 0:
        return df
    
    # 过采样
    oversampled_dfs = [df]
    for class_label, count in classes_to_oversample.items():
        # 获取当前类别的样本
        class_samples = df[df[target_col] == class_label]
        # 计算需要复制的次数
        n_copies = (min_samples - count) // len(class_samples) + 1
        # 复制样本
        oversampled_class = pd.concat([class_samples] * n_copies)
        # 随机选择所需数量的样本
        oversampled_class = oversampled_class.sample(min_samples - count, replace=False)
        oversampled_dfs.append(oversampled_class)
    
    # 合并所有数据
    oversampled_df = pd.concat(oversampled_dfs, ignore_index=True)
    return oversampled_df

def train_k_fold(cfg, device):
    """K折交叉验证训练"""
    train_df = pd.read_csv(cfg.train_csv)
    
    # 分析类别分布
    class_counts = train_df['target'].value_counts()
    logging.info("类别分布:")
    for class_label, count in class_counts.items():
        logging.info(f"类别 {class_label}: {count} 样本")
    
    kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
    best_models = []
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df, train_df['target']), 1):
        logging.info(f'\n开始训练第 {fold} 折 (共 {cfg.num_folds} 折)...')
        
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        valid_data = train_df.iloc[valid_idx].reset_index(drop=True)
        
        logging.info(f'训练集大小: {len(train_data)}, 验证集大小: {len(valid_data)}')
        
        try:
            train_dataset = ImageDataset(
                cfg.train_dir,
                train_data,
                transform=get_train_transforms(cfg),
                cfg=cfg
            )
            
            if cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0:
                from dataset import MixUpCutMixDataset
                train_dataset = MixUpCutMixDataset(
                    train_dataset,
                    mixup_alpha=cfg.mixup_alpha,
                    cutmix_alpha=cfg.cutmix_alpha,
                    prob=0.5,
                    num_classes=cfg.num_classes
                )
            
            valid_dataset = ImageDataset(
                cfg.train_dir,
                valid_data,
                transform=get_valid_transforms(cfg),
                cfg=cfg
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True
            )
            
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=cfg.batch_size * 2,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True
            )
            
            model = create_model(cfg).to(device)
            optimizer = create_optimizer(model, cfg)
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
            
            num_training_steps = len(train_loader) * cfg.epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_training_steps * cfg.warmup_ratio),
                num_training_steps=num_training_steps,
                min_lr=cfg.min_lr
            )
            
            best_model, best_acc = train_fold(
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                cfg=cfg,
                fold=fold
            )
            
            best_models.append(best_model)
            logging.info(f'Fold {fold} 完成，最佳验证准确率: {best_acc:.4f}')
            
        except Exception as e:
            logging.error(f'训练第 {fold} 折时发生错误: {str(e)}')
            continue
    
    return best_models

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
