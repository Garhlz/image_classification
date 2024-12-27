class Config:
    # 基础配置
    seed = 42  # 随机种子
    device = 'cuda'  # 设备
    output_dir = './output'  # 输出目录
    
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/image_data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/image_data/test'  
    train_csv = 'C:/code_in_laptop/d2l-zh/image_data/train.csv'  
    test_csv = 'C:/code_in_laptop/d2l-zh/image_data/sample_submission.csv'   
    submission_path = './submission.csv'
    target_col = 'target'  # 目标列名
    stratify = True  # 是否进行分层采样
    
    # 模型相关配置
    model_name = 'tf_efficientnetv2_s_in21ft1k'
    image_size = (224, 224)
    batch_size = 128
    epochs = 50
    lr = 1e-4
    
    # 通用训练配置
    num_classes = 44
    num_workers = 4
    weight_decay = 1e-2
    use_amp = True
    gradient_clip_val = 1.0
    
    # 验证和早停配置
    val_size = 0.1
    patience = 3
    early_stopping = 5  # 早停轮数
    
    # 学习率调度器配置
    warmup_ratio = 0.1
    scheduler = 'cosine'  # 学习率调度器类型
    min_lr = 1e-6  # 最小学习率
    
    # 模型保存和加载
    model_save_path = './models'
    pretrained = True
    pretrained_path = None  # 预训练模型路径
    ignore_load_errors = False  # 是否忽略加载错误
    
    # 日志配置
    log_interval = 20
    
    # 数据不平衡处理相关配置
    use_sampler = True
    use_weighted_loss = True
    use_focal_loss = False
    min_samples_per_class = 100
    max_samples_ratio = 2.0
    
    # SWA配置
    swa_start = 25
    swa_lr = 1e-5
    
    # EMA配置
    use_ema = True
    ema_decay = 0.9999
    
    # 数据增强配置
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    
    # TTA配置
    tta_transforms = 4
    
    # 模型特定配置
    dropout_rate = 0.2  # Dropout率
    drop_path_rate = 0.1  # Drop Path率
    freeze_layers = []  # 需要冻结的层

def get_config(device_type='4090'):
    return Config()