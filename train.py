import os
import logging
import pandas as pd
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import time
from dataset import ImageDataset
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_one_epoch(model, train_loader, criterion, optimizer, device, cfg):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        cfg: 配置对象
    
    Returns:
        tuple: (平均损失, 准确率)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%'
        })
        
        # 定期记录训练状态
        if (batch_idx + 1) % cfg.log_interval == 0:
            logging.info(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                        f'Loss: {avg_loss:.4f} '
                        f'Acc: {acc:.2f}%')
    
    return total_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    """
    验证模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', ncols=100)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            avg_loss = total_loss / total
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
    
    return total_loss/len(val_loader), 100.*correct/total

def predict(model, test_loader, device):
    """
    对测试集进行预测
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        tuple: (图片ID列表, 预测结果列表)
    """
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting', ncols=100)
        for images, img_names in pbar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend(img_names)
            
            pbar.set_postfix({'Processed': len(predictions)})
    
    return image_ids, predictions

def main():
    """主训练流程"""
    # 加载配置
    cfg = Config()
    device = torch.device(cfg.device)
    
    # 创建保存模型的目录
    os.makedirs(cfg.model_save_path, exist_ok=True)
    
    # 记录训练开始时间和配置信息
    start_time = time.time()
    logging.info("=== 训练开始 ===")
    logging.info(f"设备: {cfg.device}")
    logging.info(f"批次大小: {cfg.batch_size}")
    logging.info(f"学习率: {cfg.learning_rate}")
    
    # 创建数据集
    logging.info("正在加载训练集...")
    train_dataset = ImageDataset(cfg.train_dir, cfg.train_csv)
    
    # 创建带权重的采样器
    logging.info("正在创建带权重的采样器...")
    samples_weights = torch.zeros(len(train_dataset))
    for idx, (_, label) in enumerate(train_dataset):
        samples_weights[idx] = train_dataset.class_weights[label]
    
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers
    )
    
    logging.info("正在加载测试集...")
    test_dataset = ImageDataset(cfg.test_dir, cfg.test_csv, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False, 
        num_workers=cfg.num_workers
    )
    
    # 创建模型
    logging.info(f"正在创建模型 {cfg.model_name}...")
    model = timm.create_model(cfg.model_name, pretrained=True,
                            num_classes=cfg.num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience, verbose=True
    )
    
    # 训练循环
    best_acc = 0
    for epoch in range(cfg.num_epochs):
        epoch_start_time = time.time()
        logging.info(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )
        
        # 记录训练信息
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} 结果:")
        logging.info(f"训练损失: {train_loss:.4f}")
        logging.info(f"训练准确率: {train_acc:.2f}%")
        logging.info(f"耗时: {epoch_time:.2f}秒")
        
        # 更新学习率
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            model_path = os.path.join(cfg.model_save_path, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f"保存最佳模型，准确率: {best_acc:.2f}%")
    
    # 加载最佳模型进行预测
    logging.info("\n开始预测...")
    model.load_state_dict(torch.load(os.path.join(cfg.model_save_path,
                                                 'best_model.pth')))
    image_ids, predictions = predict(model, test_loader, device)
    
    # 保存预测结果
    submission = pd.DataFrame({'id': image_ids, 'predict': predictions})
    submission.to_csv(cfg.submission_path, index=False)
    logging.info(f"预测结果已保存到: {cfg.submission_path}")
    
    # 记录总训练时间
    total_time = time.time() - start_time
    logging.info(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
    logging.info(f"最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
