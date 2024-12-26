class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/train.csv'  
    test_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/sample_submission.csv'   
    
    # 模型相关配置
    model_name = 'convnext_base_in22k'  
    num_classes = 44                
    image_size = (60, 80)          
    
    # 训练相关配置
    batch_size = 64                
    num_workers = 4                
    num_epochs = 30               # 减少训练轮数
    learning_rate = 1e-4          
    weight_decay = 1e-2           
    
    # 数据增强配置
    aug_rotate = 15               
    aug_brightness = 0.3          
    aug_contrast = 0.3            
    aug_scale = (0.5, 1.0)        
    cutmix_prob = 0.5             
    mixup_prob = 0.5              
    mixup_alpha = 0.2             
    
    # 新增高级数据增强配置
    random_erasing_prob = 0.3     
    random_erasing_scale = (0.02, 0.33)  
    random_erasing_ratio = (0.3, 3.3)    
    
    color_jitter = dict(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    )
    
    # 学习率调度器配置
    scheduler_type = 'cosine'      
    warmup_epochs = 2             # 减少预热轮数
    warmup_start_lr = 1e-6        
    min_lr = 1e-7                 
    cycle_decay = 0.1             
    cycle_limit = 2               # 减少循环次数
    
    # 优化器配置
    optimizer = dict(
        type='AdamW',
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 损失函数配置
    loss = dict(
        type='CrossEntropy',
        label_smoothing=0.1,
        class_weights=True        
    )
    
    # 模型集成配置
    ensemble_models = [
        'convnext_base_in22k',
        'vit_base_patch16_224.augreg_in21k',
        'deit_base_patch16_224',
    ]  
    ensemble_weights = [0.4, 0.3, 0.3]  
    
    # 验证配置
    val_size = 0.1                
    
    # TTA配置
    tta_transforms = 4            # 减少TTA次数
    
    # 新增高级训练策略
    swa_start = 20               # 提前开始SWA
    swa_freq = 2                 # 减少SWA更新频率
    swa_lr = 1e-5                
    
    gradient_clip_val = 5.0      
    
    # Early Stopping配置
    early_stopping = dict(
        patience=5,               # 减少早停耐心值
        min_delta=1e-4,          
        mode='max'               
    )
    
    # 设备配置
    device = 'cuda'               
    
    # 保存相关配置
    model_save_path = './checkpoints'  
    submission_path = './submission.csv'  
    
    # 日志配置
    log_interval = 10             
    
    # 学习率调度器配置
    scheduler_patience = 5         # 增加学习率调度器的耐心值
    scheduler_factor = 0.2        # 调整学习率衰减因子
    
    # 验证配置
    n_splits = 5                  # 交叉验证折数
