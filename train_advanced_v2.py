import os
import logging
import pandas as pd
import numpy as np
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
from dataset import ImageDataset
from config import Config
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_advanced.log'),
        logging.StreamHandler()
    ]
)

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
        factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        factor = factor * (1.0 - min_lr) + min_lr
        return factor
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EarlyStopping:
    """早停策略"""
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf
    
    def __call__(self, val_loss):
        if self.mode == 'min':
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
        else:  # mode == 'max'
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    for batch_idx, (images, labels) in enumerate(pbar):
        # 添加调试信息
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels content: {labels}")
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast():
                outputs = model(images)  # [batch_size, num_classes]
                if isinstance(labels, dict):  # Mixup标签
                    loss = criterion(outputs, labels['targets']) * labels['lam'] + \
                           criterion(outputs, labels['targets_b']) * (1 - labels['lam'])
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if cfg.gradient_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(labels, dict):  # Mixup标签
                loss = criterion(outputs, labels['targets']) * labels['lam'] + \
                       criterion(outputs, labels['targets_b']) * (1 - labels['lam'])
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            if cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 计算准确率
        if isinstance(labels, dict):
            acc = (outputs.argmax(dim=1) == labels['targets']).float().mean()
        else:
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
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', ncols=100)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 使用argmax而不是max
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                
                losses.update(loss.item(), images.size(0))
                top1.update(acc.item(), images.size(0))
                
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{top1.avg*100:.2f}%'
                })
            except Exception as e:
                logging.error(f"验证时发生错误: {str(e)}")
                logging.error(f"Labels: {labels}")
                logging.error(f"Labels shape: {labels.shape}")
                logging.error(f"Outputs shape: {outputs.shape}")
                raise
    
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
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting', ncols=100)
        for images, img_names in pbar:
            batch_preds = []
            
            # 对每个模型进行预测
            for model, weight in zip(models, cfg.ensemble_weights):
                if cfg.tta_transforms > 0:
                    pred = test_time_augmentation(model, images, device, cfg)
                else:
                    pred = model(images.to(device)).softmax(1)
                batch_preds.append(pred * weight)
            
            # 合并所有模型的预测
            batch_preds = torch.stack(batch_preds).sum(0)
            _, predicted = batch_preds.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend(img_names)
            
            pbar.set_postfix({'Processed': len(predictions)})
    
    return image_ids, predictions

def train_k_fold(cfg, device):
    """使用K折交叉验证训练模型"""
    # 读取训练数据
    train_df = pd.read_csv(cfg.train_csv)
    
    # 创建K折交叉验证
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 存储每个折的最佳模型
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        logging.info(f'\n=== Fold {fold + 1}/3 ===')
        
        # 创建数据加载器
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = ImageDataset(cfg.train_dir, train_subset, is_test=False, cfg=cfg)
        val_dataset = ImageDataset(cfg.train_dir, val_subset, is_test=True, cfg=cfg)
        
        # 创建带权重的采样器
        samples_weights = torch.zeros(len(train_dataset))
        for idx, (_, label) in enumerate(train_dataset):
            samples_weights[idx] = train_dataset.class_weights[label]
        
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )
        
        # 创建模型
        model = timm.create_model(cfg.model_name, pretrained=True,
                                num_classes=cfg.num_classes)
        model = model.to(device)
        
        # 创建SWA模型
        swa_model = AveragedModel(model)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.loss['label_smoothing'])
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer['lr'],
            weight_decay=cfg.optimizer['weight_decay'],
            betas=cfg.optimizer['betas'],
            eps=cfg.optimizer['eps']
        )
        
        # 创建学习率调度器
        num_training_steps = len(train_loader) * cfg.num_epochs
        num_warmup_steps = len(train_loader) * cfg.warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            cfg.min_lr
        )
        
        # 创建SWA调度器
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
        
        # 创建早停策略
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping['patience'],
            min_delta=cfg.early_stopping['min_delta'],
            mode=cfg.early_stopping['mode']
        )
        
        # 创建混合精度训练的scaler
        scaler = GradScaler()
        
        # 训练循环
        best_acc = 0
        for epoch in range(cfg.num_epochs):
            # 训练一个epoch
            train_loss, train_acc = train_one_epoch(
                model if epoch < cfg.swa_start else swa_model,
                train_loader,
                criterion,
                optimizer,
                scheduler if epoch < cfg.swa_start else swa_scheduler,
                device,
                cfg,
                scaler
            )
            
            # 如果达到SWA起始轮数，开始使用SWA
            if epoch >= cfg.swa_start and (epoch - cfg.swa_start) % cfg.swa_freq == 0:
                swa_model.update_parameters(model)
            
            # 验证
            val_loss, val_acc = validate(
                swa_model if epoch >= cfg.swa_start else model,
                val_loader,
                criterion,
                device
            )
            
            # 记录结果
            logging.info(f'Epoch {epoch + 1}/{cfg.num_epochs}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
            logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = os.path.join(cfg.model_save_path, f'best_model_fold{fold}.pth')
                if epoch >= cfg.swa_start:
                    torch.save(swa_model.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                logging.info(f'保存最佳模型，验证准确率: {best_acc*100:.2f}%')
            
            # 早停检查
            early_stopping(val_acc)
            if early_stopping.early_stop:
                logging.info("触发早停")
                break
        
        # 加载最佳模型
        if os.path.exists(os.path.join(cfg.model_save_path, f'best_model_fold{fold}.pth')):
            if epoch >= cfg.swa_start:
                swa_model.load_state_dict(torch.load(os.path.join(cfg.model_save_path, f'best_model_fold{fold}.pth')))
                fold_models.append(swa_model)
            else:
                model.load_state_dict(torch.load(os.path.join(cfg.model_save_path, f'best_model_fold{fold}.pth')))
                fold_models.append(model)
    
    return fold_models

def main():
    """主训练流程"""
    cfg = Config()
    device = torch.device(cfg.device)
    
    # 创建���存模型的目录
    os.makedirs(cfg.model_save_path, exist_ok=True)
    
    # 记录训练开始时间和配置信息
    start_time = time.time()
    logging.info("=== 训练开始 ===")
    logging.info(f"设备: {cfg.device}")
    logging.info(f"批次大小: {cfg.batch_size}")
    logging.info(f"学习率: {cfg.learning_rate}")
    
    # 训练K折交叉验证模型
    models = train_k_fold(cfg, device)
    
    # 创建测试集数据加载器
    test_dataset = ImageDataset(cfg.test_dir, cfg.test_csv, is_test=True, cfg=cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    # 使用模型集成进行预测
    logging.info("\n开始预测...")
    image_ids, predictions = predict(models, test_loader, device, cfg)
    
    # 保存预测结果
    submission = pd.DataFrame({'id': image_ids, 'predict': predictions})
    submission.to_csv(cfg.submission_path, index=False)
    logging.info(f"预测结果已保存到: {cfg.submission_path}")
    
    # 记录总训练时间
    total_time = time.time() - start_time
    logging.info(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")

if __name__ == '__main__':
    main()
