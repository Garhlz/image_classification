import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import random
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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
    def __init__(self, root_dir, csv_file, transform=None, is_test=False, cfg=None):
        self.root_dir = root_dir
        self.is_test = is_test
        self.cfg = cfg
        self.transform = transform
        
        # 检查目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"目录不存在: {root_dir}")
        
        # 读取CSV文件或直接使用DataFrame
        logging.info("正在加载数据...")
        if isinstance(csv_file, str):
            if not os.path.exists(csv_file):
                raise ValueError(f"CSV文件不存在: {csv_file}")
            self.df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            raise ValueError("csv_file必须是字符串路径或DataFrame")
        
        # 确保id列为整数类型
        self.df['id'] = self.df['id'].astype(int).astype(str)  # 直接转换为字符串
            
        logging.info(f"数据集大小: {len(self.df)} 样本")
        
        # 验证CSV文件格式
        required_columns = ['id'] if is_test else ['id', 'target']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        if not is_test:
            # 检查标签范围
            if 'target' in self.df.columns:
                unique_labels = self.df['target'].unique()
                logging.info(f"数据集中的唯一标签: {sorted(unique_labels)}")
                if cfg and hasattr(cfg, 'num_classes'):
                    invalid_labels = [label for label in unique_labels 
                                    if label < 0 or label >= cfg.num_classes]
                    if invalid_labels:
                        raise ValueError(f"发现无效的标签值: {invalid_labels}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id']  # 已经是字符串了
        
        # 尝试不同的图片扩展名
        img_extensions = ['.jpg', '.jpeg', '.png']
        img_path = None
        
        for ext in img_extensions:
            temp_path = os.path.join(self.root_dir, f"{img_name}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            logging.error(f"找不到图片文件: {img_name} (尝试过的扩展名: {img_extensions})")
            # 创建一个随机的替代图像
            random_image = np.random.randint(0, 255, size=(*self.cfg.image_size, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=random_image)
                random_image = transformed["image"]
            if self.is_test:
                return random_image, img_name  # 返回图像和图像名
            else:
                label = self.df.iloc[idx]['target']
                return random_image, label
        
        try:
            # 使用cv2读取图像
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 转换BGR到RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            if self.is_test:
                return image, img_name  # 返回图像和图像名
            else:
                label = self.df.iloc[idx]['target']
                return image, label
                
        except Exception as e:
            logging.error(f"处理图像时出错 {img_path}: {str(e)}")
            # 返回随机图像作为替代
            random_image = np.random.randint(0, 255, size=(*self.cfg.image_size, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=random_image)
                random_image = transformed["image"]
            if self.is_test:
                return random_image, img_name  # 返回图像和图像名
            else:
                label = self.df.iloc[idx]['target']
                return random_image, label
