from dataclasses import dataclass
import torch

@dataclass
class Config:
    # 数据配置
    data_dir: str = "data"  # 数据目录
    train_csv: str = "train.csv"  # 训练数据CSV文件
    val_csv: str = "val.csv"  # 验证数据CSV文件
    image_size: int = 384  # 图像大小
    num_classes: int = 44  # 类别数量
    
    # 训练配置
    model_name: str = "convnext_base"  # 模型名称
    pretrained: bool = True  # 是否使用预训练模型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
    seed: int = 42  # 随机种子
    epochs: int = 12  # 训练轮数 (减少轮数)
    batch_size: int = 16  # 批次大小
    num_workers: int = 4  # 数据加载线程数
    n_folds: int = 3  # 交叉验证折数 (减少折数)
    
    # 优化器配置
    optimizer: str = "AdamW"  # 优化器类型
    learning_rate: float = 1e-4  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    
    # 学习率调度器配置
    scheduler: str = "CosineAnnealingLR"  # 学习率调度器类型
    min_lr: float = 1e-6  # 最小学习率
    warmup_ratio: float = 0.1  # 预热比例
    
    # 数据增强配置
    mixup_alpha: float = 0.2  # Mixup alpha值 (降低增强强度)
    cutmix_alpha: float = 0.2  # CutMix alpha值 (降低增强强度)
    augmix_prob: float = 0.3  # AugMix概率 (降低增强概率)
    
    # 训练策略配置
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    gradient_clip_val: float = 1.0  # 梯度裁剪值
    use_amp: bool = True  # 是否使用混合精度训练
    use_ema: bool = False  # 关闭EMA以加快训练
    
    # 深度监督配置
    use_deep_supervision: bool = False  # 关闭深度监督以简化训练
    deep_supervision_weights: list = None
    
    # 保存配置
    output_dir: str = "outputs"  # 输出目录
    model_dir: str = "models"  # 模型保存目录
    save_best: bool = True  # 是否保存最佳模型
