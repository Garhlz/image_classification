import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch import nn
from dataset import ImageDataset
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)

def create_model(cfg):
    """创建模型实例"""
    if cfg.use_deep_supervision:
        base_model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
        model = DeepSupervisionModel(base_model, cfg.num_classes)
    else:
        model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
    return model

class DeepSupervisionModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        
        # 获取中间特征图的通道数
        if hasattr(base_model, 'stages'):  # ConvNeXt
            channels = []
            for stage in base_model.stages[1:]:  # 跳过第一个stage
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
        features = []
        x = self.base_model.stem(x)
        for i, stage in enumerate(self.base_model.stages):
            x = stage(x)
            if i > 0:
                features.append(x)
        
        x = self.base_model.head.global_pool(x)
        x = self.base_model.head.norm(x)
        x = self.base_model.head.flatten(x)
        main_out = self.base_model.head.fc(x)
        
        aux_outs = []
        for feature, classifier in zip(features, self.aux_classifiers):
            aux_out = classifier(feature)
            aux_outs.append(aux_out)
            
        return [main_out] + aux_outs

def predict(model, test_loader, device, cfg):
    """使用单个模型进行预测"""
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting')
        for images, img_names in pbar:
            images = images.to(device)
            outputs = model(images)
            
            # 如果是深度监督模型，只使用主分类器的输出
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # 获取预测结果
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend(img_names)
            
            pbar.set_postfix({'Processed': len(predictions)})
    
    return image_ids, predictions

def main():
    """主测试流程"""
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'使用设备: {device}')
        
        # 创建配置
        cfg = Config()
        
        # 读取测试数据
        test_df = pd.read_csv(cfg.test_csv)
        
        # 创建测试数据集和加载器
        test_transform = A.Compose([
            A.Resize(height=cfg.image_size[0], width=cfg.image_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        test_dataset = ImageDataset(
            cfg.test_dir, 
            test_df, 
            transform=test_transform, 
            is_test=True, 
            cfg=cfg
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        # 创建模型并加载权重
        model = create_model(cfg)
        model_path = os.path.join(cfg.model_save_path, 'best_model_fold1.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logging.info("开始预测...")
        image_ids, predictions = predict(model, test_loader, device, cfg)
        
        if not predictions:
            raise ValueError("预测失败，没有预测结果")
        
        # 创建提交文件
        submission = pd.DataFrame({
            'id': image_ids,
            'predict': predictions
        })
        
        # 保存预测结果
        submission.to_csv(cfg.submission_path, index=False)
        logging.info(f"预测结果已保存到: {cfg.submission_path}")
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()