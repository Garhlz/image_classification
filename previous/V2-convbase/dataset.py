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
        
        # 读取CSV文件
        logging.info(f"正在读取CSV文件: {csv_file}")
        if not os.path.exists(csv_file):
            raise ValueError(f"CSV文件不存在: {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        logging.info(f"数据集大小: {len(self.df)} 样本")
        
        # 验证CSV文件格式
        required_columns = ['id'] if is_test else ['id', 'target']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        # 高级数据增强
        if not is_test:
            # RandAugment配置
            self.rand_augment = rand_augment_transform(
                config_str='rand-m9-n3-mstd0.5',
                hparams={'translate_const': 117}
            )
            
            # Albumentations数据增强
            self.train_transform = A.Compose([
                A.RandomResizedCrop(
                    height=cfg.image_size[0], 
                    width=cfg.image_size[1],
                    scale=cfg.aug_scale,
                    ratio=(0.75, 1.333)
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
                A.OneOf([
                    A.OpticalDistortion(p=1),
                    A.GridDistortion(p=1),
                    A.ElasticTransform(p=1),
                ], p=0.3),
                A.ColorJitter(
                    brightness=cfg.aug_brightness,
                    contrast=cfg.aug_contrast,
                    saturation=0.3,
                    hue=0.1,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.test_transform = A.Compose([
                A.Resize(
                    height=cfg.image_size[0],
                    width=cfg.image_size[1]
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        if not is_test:
            self.labels = self.df['target'].values
            # 验证标签值的范围
            unique_labels = np.unique(self.labels)
            if len(unique_labels) > cfg.num_classes or np.min(self.labels) < 0:
                raise ValueError("标签值超出预期范围")
            # 计算并记录类别分布
            self._log_class_distribution()
            # 计算类别权重
            self.class_weights = self._calculate_class_weights()
            logging.info("类别权重计算完成")
            
            # 设置Mixup
            self.mixup = Mixup(
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=1.0,
                cutmix_minmax=None,
                prob=cfg.mixup_prob,
                switch_prob=cfg.cutmix_prob,
                mode='batch',
                label_smoothing=0.1,
                num_classes=cfg.num_classes
            )
            
        # 预先检查所有图片文件是否存在
        self._verify_images()
    
    def _verify_images(self):
        """验证所有图片文件是否存在"""
        missing_images = []
        for idx in range(len(self.df)):
            img_name = f"{self.df['id'].iloc[idx]}.jpg"
            img_path = os.path.join(self.root_dir, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_path)
        
        if missing_images:
            logging.error(f"发现 {len(missing_images)} 个缺失的图片文件")
            for path in missing_images[:10]:
                logging.error(f"缺失文件: {path}")
            if len(missing_images) > 10:
                logging.error("...")
            raise FileNotFoundError(f"数据集中有 {len(missing_images)} 个缺失的图片文件")
    
    def _log_class_distribution(self):
        """记录数据集的类别分布情况"""
        class_counts = self.df['target'].value_counts().sort_index()
        logging.info("\n类别分布情况:")
        logging.info(f"类别数量: {len(class_counts)}")
        logging.info(f"平均样本数: {class_counts.mean():.2f}")
        logging.info(f"最大样本数: {class_counts.max()}")
        logging.info(f"最小样本数: {class_counts.min()}")
        logging.info(f"标准差: {class_counts.std():.2f}")
    
    def _calculate_class_weights(self):
        """计算类别权重"""
        class_counts = self.df['target'].value_counts().sort_index()
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # 构建图片路径
            img_name = f"{self.df['id'].iloc[idx]}.jpg"
            img_path = os.path.join(self.root_dir, img_name)
            
            # 读取图片
            try:
                image = Image.open(img_path).convert('RGB')
                # 应用RandAugment（在PIL图像上）
                if not self.is_test and random.random() < 0.3:
                    image = self.rand_augment(image)
                # 转换为numpy数组
                image = np.array(image)
            except Exception as e:
                logging.error(f"读取图片失败: {img_path}")
                logging.error(f"错误信息: {str(e)}")
                raise
            
            # 应用Albumentations数据增强
            try:
                if self.is_test:
                    transformed = self.test_transform(image=image)
                else:
                    transformed = self.train_transform(image=image)
                image = transformed['image']
            except Exception as e:
                logging.error(f"转换图片失败: {img_path}")
                logging.error(f"错误信息: {str(e)}")
                raise
            
            if self.is_test:
                return image, self.df['id'].iloc[idx]
            else:
                return image, self.labels[idx]
                
        except Exception as e:
            logging.error(f"处理样本失败，索引: {idx}")
            logging.error(f"错误信息: {str(e)}")
            raise
