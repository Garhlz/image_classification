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
            
        logging.info(f"数据集大小: {len(self.df)} 样本")
        
        # 验证CSV文件格式
        required_columns = ['id'] if is_test else ['id', 'target']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要的列: {missing_columns}")
        
        # 确保target列的值在合法范围内
        if not is_test:
            if self.df['target'].max() >= cfg.num_classes or self.df['target'].min() < 0:
                raise ValueError(f"标签值超出范围[0, {cfg.num_classes-1}]")
        
        # 计算类别权重
        if not is_test:
            class_counts = self.df['target'].value_counts()
            total_samples = len(self.df)
            self.class_weights = torch.FloatTensor([
                total_samples / (len(class_counts) * count) 
                for count in class_counts
            ])
        
        # 数据增强
        if not is_test:
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    height=cfg.image_size[0],
                    width=cfg.image_size[1],
                    scale=cfg.aug_scale
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=cfg.aug_rotate,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=0.3),
                A.ColorJitter(
                    brightness=cfg.aug_brightness,
                    contrast=cfg.aug_contrast,
                    saturation=0.3,
                    hue=0.2,
                    p=0.5
                ),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(
                    height=cfg.image_size[0],
                    width=cfg.image_size[1]
                ),
                A.Normalize(),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.df.iloc[idx]['id']}.jpg")
        
        try:
            image = cv2.imread(img_name)
            if image is None:
                raise ValueError(f"无法读取图像: {img_name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"读取图像出错 {img_name}: {str(e)}")
            # 返回一个随机噪声图像作为替代
            image = np.random.randint(0, 255, size=(*self.cfg.image_size, 3), dtype=np.uint8)
        
        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                logging.error(f"数据增强出错 {img_name}: {str(e)}")
                # 返回原始图像转换为张量
                image = torch.from_numpy(image.transpose(2, 0, 1)) / 255.0
        
        if self.is_test:
            return image, self.df.iloc[idx]['id']
        else:
            label = self.df.iloc[idx]['target']
            return image, label
