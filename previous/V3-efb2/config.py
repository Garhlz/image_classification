class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/train.csv'  
    test_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/sample_submission.csv'   
    submission_path = './submission.csv'  # 预测结果保存路径
    
    # 模型相关配置
    model_name = 'efficientnet_b2'  # 使用timm本地模型名称
    num_classes = 44                
    image_size = (260, 260)        # EfficientNet-B2推荐输入大小
    
    # 训练相关配置
    batch_size = 48                # 由于模型更大，稍微减小batch_size
    num_workers = 4                
    epochs = 20                    # 每折至少15轮
    lr = 2e-4                      # 稍微降低学习率以适应新模型
    weight_decay = 1e-2            
    
    # 训练策略
    use_amp = True                 # 使用混合精度训练
    gradient_clip_val = 1.0        
    num_folds = 2                  # 使用2折交叉验证
    
    # 早停策略
    patience = 3                   # 给模型更多机会
    
    # SWA配置
    swa_start = int(epochs * 0.7)  # 在训练后期开始SWA
    swa_lr = 1e-5                  
    
    # 学习率调度器配置
    warmup_ratio = 0.1             
    
    # 模型保存和加载
    model_save_path = './models'   
    pretrained = True              # 使用预训练模型
    
    # 日志配置
    log_interval = 20              
    
    # 集成配置
    ensemble_weights = [1.0]       # 单模型权重
    tta_transforms = 0             # 默认不使用测试时增强
