import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, img_dir, df, transform=None, is_test=False, cfg=None):
        self.img_dir = img_dir
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        
        # 确保id列是字符串格式
        if 'id' in self.df.columns:
            self.df['id'] = self.df['id'].astype(str)
            
        # 确保target列是整数类型
        if 'target' in self.df.columns:
            self.df['target'] = self.df['target'].astype(int)
            
        self.transform = transform
        self.is_test = is_test
        self.cfg = cfg
        
        # 记录所有图像文件名
        self.image_files = os.listdir(img_dir)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图像ID和标签
        if self.is_test:
            img_name = f"{self.df.iloc[idx]['id']}.jpg"
        else:
            img_name = f"{self.df.iloc[idx]['id']}.jpg"
            
        # 构建图像路径
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # 尝试读取图像
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"读取图像出错 {img_path}: {str(e)}")
            # 返回一个随机的黑色图像作为替代
            image = np.zeros((self.cfg.image_size[0], self.cfg.image_size[1], 3), dtype=np.uint8)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.is_test:
            return image, img_name
            
        # 确保标签是PyTorch张量
        target = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.long)
        return image, target

class MixUpCutMixDataset(Dataset):
    def __init__(self, dataset, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, num_classes=None):
        self.dataset = dataset
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.dataset)
        
    def mixup(self, data1, data2, target1, target2, alpha):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        # 转换目标为one-hot格式
        if not isinstance(target1, torch.Tensor):
            target1 = torch.tensor(target1)
        if not isinstance(target2, torch.Tensor):
            target2 = torch.tensor(target2)
            
        target1 = target1.long()
        target2 = target2.long()
            
        if len(target1.shape) == 0:
            target1 = target1.unsqueeze(0)
        if len(target2.shape) == 0:
            target2 = target2.unsqueeze(0)
            
        target1_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        target1_onehot[target1] = 1
        target2_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        target2_onehot[target2] = 1
            
        data = data1 * lam + data2 * (1 - lam)
        target = target1_onehot * lam + target2_onehot * (1 - lam)
        return data, target
        
    def cutmix(self, data1, data2, target1, target2, alpha):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        # 转换目标为one-hot格式
        if not isinstance(target1, torch.Tensor):
            target1 = torch.tensor(target1)
        if not isinstance(target2, torch.Tensor):
            target2 = torch.tensor(target2)
            
        target1 = target1.long()
        target2 = target2.long()
            
        if len(target1.shape) == 0:
            target1 = target1.unsqueeze(0)
        if len(target2.shape) == 0:
            target2 = target2.unsqueeze(0)
            
        target1_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        target1_onehot[target1] = 1
        target2_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        target2_onehot[target2] = 1
            
        H, W = data1.shape[1:]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        data = data1.clone()
        data[:, bby1:bby2, bbx1:bbx2] = data2[:, bby1:bby2, bbx1:bbx2]
        
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        target = target1_onehot * lam + target2_onehot * (1 - lam)
        
        return data, target
    
    def __getitem__(self, idx):
        data1, target1 = self.dataset[idx]
        
        # 确保target1是张量
        if not isinstance(target1, torch.Tensor):
            target1 = torch.tensor(target1, dtype=torch.long)
        else:
            target1 = target1.long()
            
        # 随机决定是否进行混合
        if np.random.random() > self.prob:
            if len(target1.shape) == 0:
                target1 = target1.unsqueeze(0)
            target1_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
            target1_onehot[target1] = 1
            return data1, target1_onehot
            
        # 随机选择另一个样本
        index2 = np.random.randint(len(self.dataset))
        data2, target2 = self.dataset[index2]
        
        # 确保target2是张量
        if not isinstance(target2, torch.Tensor):
            target2 = torch.tensor(target2, dtype=torch.long)
        else:
            target2 = target2.long()
        
        # 随机选择使用mixup还是cutmix
        if np.random.random() < 0.5 and self.mixup_alpha > 0:
            data, target = self.mixup(data1, data2, target1, target2, self.mixup_alpha)
        else:
            data, target = self.cutmix(data1, data2, target1, target2, self.cutmix_alpha)
            
        return data, target

def get_train_transforms(cfg, stage=0):
    """获取训练数据增强"""
    size = cfg.image_size
    
    transforms = [
        A.RandomResizedCrop(height=size[0], width=size[1], scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
            A.GridDistortion(distort_limit=0.1, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.ToGray(p=1.0),
        ], p=0.3),
    ]
        
    if hasattr(cfg, 'random_erasing_prob') and cfg.random_erasing_prob > 0:
        transforms.append(
            A.CoarseDropout(
                max_holes=8, 
                max_height=size[0]//8,
                max_width=size[1]//8,
                min_holes=1,
                min_height=size[0]//16,
                min_width=size[1]//16,
                p=cfg.random_erasing_prob
            )
        )
    
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)

def get_valid_transforms(cfg):
    """获取验证数据增强
    Args:
        cfg: 配置对象
    Returns:
        albumentations.Compose: 数据增强pipeline
    """
    return A.Compose([
        A.Resize(height=cfg.image_size[0], width=cfg.image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
