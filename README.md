# 图像分类项目优化版本 V4.0

## 项目概述
这是一个针对44类图像分类任务的深度学习解决方案的优化版本。本版本使用ConvNeXt Base作为主干网络，并引入了更多高级训练策略和优化方法。 
当前主页是V4的内容，先前版本在previous文件夹内 

## 主要特点与改进

### 1. 模型架构
- **基础模型**：ConvNeXt Base
  - 预训练权重：ImageNet-22K
  - 纯CNN架构，无需注意力机制
  - 更大的感受野
  - 更强的特征提取能力

### 2. 训练策略优化

#### 2.1 数据增强
- 高级数据增强组合
  - RandomResizedCrop
  - ColorJitter
  - GaussNoise/GaussianBlur/MotionBlur
  - GridDistortion/OpticalDistortion
  - RGBShift/HueSaturationValue
- MixUp (alpha=0.2)
- CutMix (alpha=1.0)
- 随机擦除 (prob=0.3)

#### 2.2 训练技巧
- 混合精度训练 (AMP)
- 指数移动平均 (EMA)
- 梯度裁剪
- 标签平滑
- 余弦学习率调度
- 2折交叉验证

### 3. 优化器配置
- AdamW优化器
- 学习率：1e-3
- 权重衰减：1e-2
- 梯度累积：2步

### 4. 训练监控
- 训练/验证损失追踪
- 准确率监控
- 早停机制
- 模型检查点保存
- 详细日志记录

## 环境要求
- Python 3.8+
- PyTorch 1.9+
- timm 0.9.2+
- albumentations 1.3.1+

## 使用方法
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
- 将训练图像放在 train_dir 目录
- 将测试图像放在 test_dir 目录
- 确保 train.csv 和 test.csv 文件就位

3. 开始训练：
```bash
python train_advanced_v4.py
```

## 性能指标
- 验证集准确率：91.5%
- 训练时间：~5小时 (单GPU)
- GPU显存占用：~7GB

## 未来改进方向
1. 模型集成策略优化
2. 更多的数据增强方法探索
3. 自适应学习率策略
4. 分布式训练支持 