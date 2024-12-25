import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

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
    """
    图像分类数据集类
    支持训练集和测试集的数据加载，包含数据增强和类别平衡
    """
    
    def __init__(self, root_dir, csv_file, transform=None, is_test=False):
        """
        初始化数据集
        
        Args:
            root_dir (str): 图片根目录
            csv_file (str): CSV文件路径
            transform (callable, optional): 数据转换
            is_test (bool): 是否为测试集
        """
        self.root_dir = root_dir
        self.is_test = is_test
        
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
        
        # 默认数据增强
        train_transform = transforms.Compose([
            transforms.Resize((60, 80)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((60, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transform if transform else (test_transform if is_test else train_transform)
        
        if not is_test:
            self.labels = self.df['target'].values
            # 验证标签值的范围
            unique_labels = np.unique(self.labels)
            if len(unique_labels) > 44 or np.min(self.labels) < 0:
                raise ValueError("标签值超出预期范围")
            # 计算并记录类别分布
            self._log_class_distribution()
            # 计算类别权重
            self.class_weights = self._calculate_class_weights()
            logging.info("类别权重计算完成")
            
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
            for path in missing_images[:10]:  # 只显示前10个
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
        """
        计算类别权重
        使用逆频率作为权重，并进行归一化
        """
        class_counts = self.df['target'].value_counts().sort_index()
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label) 或 (image, image_id)
        """
        try:
            # 构建图片路径
            img_name = f"{self.df['id'].iloc[idx]}.jpg"
            img_path = os.path.join(self.root_dir, img_name)
            
            # 读取并转换图片
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logging.error(f"读取图片失败: {img_path}")
                logging.error(f"错误信息: {str(e)}")
                raise
            
            # 应用数据转换
            try:
                image = self.transform(image)
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
