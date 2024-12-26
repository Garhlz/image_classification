class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/train.csv'  
    test_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/sample_submission.csv'   
    submission_path = './submission.csv'  
    num_classes = 44                
    
    # 图像配置
    image_size = (260, 260)       # 使用更大的输入尺寸
    progressive_resizing = False   # 关闭渐进式尺寸调整
    
    # 数据增强配置
    random_erasing_prob = 0.3    # 随机擦除概率
    mixup_alpha = 0.2           # MixUp增强参数
    cutmix_alpha = 1.0          # CutMix增强参数
    
    # 模型相关配置
    model_name = 'convnext_base'  # 使用ConvNeXt Base模型
    pretrained = True            # 使用预训练模型
    use_deep_supervision = True   # 使用深度监督
    aux_weight = 0.4             # 辅助损失权重
    
    # 训练相关配置
    batch_size = 32              
    epochs = 20                  
    num_folds = 3                # 3折交叉验证
    num_workers = 4              
    patience = 4                 # 早停轮数
    label_smoothing = 0.1        # 标签平滑
    
    # 优化器配置
    optimizer = 'AdamW'         
    lr = 1e-4                   
    weight_decay = 1e-2         
    gradient_accumulation_steps = 2  # 梯度累积
    
    # 学习率策略
    use_cosine_schedule = True    # 使用余弦退火
    warmup_ratio = 0.1           # 预热比例
    min_lr = 1e-6                # 最小学习率
    
    # 高级训练配置
    use_amp = True               # 混合精度训练
    use_ema = True               # 指数移动平均
    ema_decay = 0.9997          
    gradient_clip_val = 1.0      # 梯度裁剪
    
    # 日志配置
    log_interval = 100           # 每100个batch记录一次
    model_save_path = './models'  # 模型保存路径
