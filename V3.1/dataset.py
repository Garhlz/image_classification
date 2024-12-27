import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import logging
import random
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset.log'),
        logging.StreamHandler()
    ]
)

class ImageDataset(Dataset):
    def __init__(self, root_dir, df, transform=None, cfg=None):
        """
        初始化数据集
        Args:
            root_dir: 图像文件所在目录
            df: 包含图像ID和标签的DataFrame
            transform: 数据增强转换
            cfg: 配置对象
        """
        self.root_dir = root_dir
        self.df = df.copy()  # 创建DataFrame的副本
        self.transform = transform
        self.cfg = cfg
        
        # 检查目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"图像目录不存在: {root_dir}")
            
        # 确保DataFrame包含所需的列
        required_cols = {'id'}
        if cfg and hasattr(cfg, 'target_col'):
            required_cols.add(cfg.target_col)
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame必须包含以下列: {required_cols}")
            
        # 确保所有图像ID都是字符串类型
        self.df['id'] = self.df['id'].astype(str)
        
        # 统计标签分布
        if cfg and hasattr(cfg, 'target_col') and cfg.target_col in self.df.columns:
            label_counts = self.df[cfg.target_col].value_counts()
            logging.info("\n标签分布:")
            for label, count in label_counts.items():
                logging.info(f"类别 {label}: {count} 样本")
                
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError(f"索引 {idx} 超出范围 (数据集大小: {len(self.df)})")
            
        # 获取图像ID和标签
        img_id = self.df.iloc[idx]['id']
        label = self.df.iloc[idx][self.cfg.target_col] if self.cfg and hasattr(self.cfg, 'target_col') else -1
        
        # 构建图像路径并尝试不同的扩展名
        img_extensions = ['.jpg', '.jpeg', '.png']
        img_path = None
        
        for ext in img_extensions:
            temp_path = os.path.join(self.root_dir, f"{img_id}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None:
            raise FileNotFoundError(f"找不到图像文件: {img_id} (尝试过的扩展名: {img_extensions})")
            
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"读取图像失败 {img_path}: {str(e)}")
            raise
            
        # 应用数据增强
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logging.error(f"数据增强失败 {img_path}: {str(e)}")
                raise
                
        return image, label

class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.copy()  # 创建DataFrame的副本
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError(f"索引 {idx} 超出范围 (数据集大小: {len(self.df)})")
            
        try:
            # 获取图像ID和标签
            img_id = str(self.df.iloc[idx]['id'])
            image_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用数据增强
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            # 准备返回数据
            if 'label' in self.df.columns:
                label = self.df.iloc[idx]['label']
                return {'image': image, 'label': label}
            else:
                return {'image': image}
                
        except Exception as e:
            logging.error(f"处理样本 {idx} 时发生错误: {str(e)}")
            raise
