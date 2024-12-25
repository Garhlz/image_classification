class Config:
    # 数据相关配置
    train_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/train'  
    test_dir = 'C:/code_in_laptop/d2l-zh/lab5/data/test'   
    train_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/train.csv'  # 训练集CSV文件
    test_csv = 'C:/code_in_laptop/d2l-zh/lab5/data/sample_submission.csv'   # 测试集CSV文件
    
    # 模型相关配置
    model_name = 'efficientnet_b0'  # timm模型名称
    num_classes = 44                # 类别数量
    image_size = (60, 80)          # 图片尺寸
    
    # 训练相关配置
    batch_size = 64                # 批次大小
    num_workers = 4                # 数据加载线程数
    num_epochs = 30               # 训练轮数
    learning_rate = 1e-3          # 初始学习率
    weight_decay = 1e-4           # AdamW优化器的权重衰减
    
    # 数据增强配置
    aug_rotate = 10               # 随机旋转角度范围
    aug_brightness = 0.2          # 亮度调整范围
    aug_contrast = 0.2            # 对比度调整范围
    
    # 学习率调度器配置
    scheduler_patience = 3         # 学习率调度器的耐心值
    scheduler_factor = 0.1        # 学习率调整因子
    
    # 设备配置
    device = 'cuda'               # 使用GPU训练，如果没有GPU请改为'cpu'
    
    # 保存相关配置
    model_save_path = './checkpoints'  # 模型保存路径
    submission_path = './submission.csv'  # 预测结果保存路径
    
    # 日志配置
    log_interval = 10             # 每多少个batch记录一次训练状态
