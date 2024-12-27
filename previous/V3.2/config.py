class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/image_data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/image_data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/image_data/train.csv'  
    test_csv = 'C:/code_in_laptop/d2l-zh/image_data/sample_submission.csv'   
    submission_path = './submission.csv'  # 添加回提交文件路径
    
    # 模型相关配置
    model_name = 'convnext_base'  # 使用ConvNeXt-Base模型
    num_classes = 44
    image_size = (224, 224)  # ConvNeXt标准输入大小
    
    # 训练相关配置
    batch_size = 64  # 4090显存充足，增大batch_size
    num_workers = 4
    epochs = 30  # 增加训练轮数
    lr = 1e-4  # 适当降低学习率
    weight_decay = 1e-4
    
    # 训练策略
    use_amp = True  # 使用混合精度训练
    gradient_clip_val = 1.0
    
    # 模型保存和加载
    model_save_path = './models'
    pretrained = True
    local_weights_path = 'C:/code_in_laptop/d2l-zh/image_classification/previous/V3.2/convnext_base_1k_224_ema.pth'  # 本地权重文件路径
    
    # 日志配置
    log_interval = 20
    
    # 学习率调度器配置
    warmup_ratio = 0.1
    
    # 测试时增强
    tta_transforms = 1  # 启用测试时增强
