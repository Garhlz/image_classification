class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/train.csv'  # 训练集CSV文件
    test_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/sample_submission.csv'   # 测试集CSV文件
    
    # 模型相关配置
    model_name = 'convnext_base_in22k'  # 使用更轻量的ConvNeXt Base模型
    num_classes = 44                # 类别数量
    image_size = (60, 80)          # 图片尺寸
    
    # 训练相关配置
    batch_size = 64                # 增大批次大小
    num_workers = 4                # 数据加载线程数
    num_epochs = 50               # 增加训练轮数
    learning_rate = 1e-4          # 调整学习率
    weight_decay = 1e-2           # 调整权重衰减
    
    # 数据增强配置
    aug_rotate = 15               # 增加旋转角度范围
    aug_brightness = 0.3          # 增加亮度调整范围
    aug_contrast = 0.3            # 增加对比度调整范围
    aug_scale = (0.5, 1.0)        # 修改scale范围到[0,1]
    cutmix_prob = 0.5             # CutMix概率
    mixup_prob = 0.5              # Mixup概率
    mixup_alpha = 0.2             # Mixup alpha参数
    
    # 学习率调度器配置
    scheduler_patience = 5         # 增加学习率调度器的耐心值
    scheduler_factor = 0.2        # 调整学习率衰减因子
    warmup_epochs = 3             # 添加warmup轮数
    min_lr = 1e-7                 # 最小学习率
    
    # 设备配置
    device = 'cuda'               # 使用GPU训练
    
    # 保存相关配置
    model_save_path = './checkpoints'  # 模型保存路径
    submission_path = './submission.csv'  # 预测结果保存路径
    
    # 日志配置
    log_interval = 10             # 每多少个batch记录一次训练状态
    
    # 模型集成配置
    ensemble_models = [
        'convnext_base_in22k',
        'vit_base_patch16_224.augreg_in21k',
        'deit_base_patch16_224',
    ]  # 使用更轻量的模型进行集成
    ensemble_weights = [0.4, 0.3, 0.3]  # 模型集成权重
    
    # 验证配置
    val_size = 0.1                # 验证集比例
    n_splits = 5                  # 交叉验证折数
    
    # TTA配置
    tta_transforms = 8            # TTA变换数量
