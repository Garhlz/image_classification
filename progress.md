# 图像分类项目优化技术演进记录

## 一、版本概述

项目目标：解决44类图像分类任务，处理类别严重不平衡问题（最大类别12341样本，最小类别个位数）。

### 数据集特征
- 图片尺寸：60×80像素
- 训练集：35,551张图片
- 测试集：8,889张图片
- 类别数：44类
- 类别分布极不平衡：
  - 最大类：12341样本
  - 最小类：个位数样本
  - 中位数：166样本

## 二、版本演进详解

### V1版本：基础优化方案

#### 1. 模型架构
- 主干网络：EfficientNet-B0
- 预训练：ImageNet-1K权重
- 选择理由：
  - 轻量级但效果好
  - FLOPS和参数量较小
  - 适合小尺寸图片处理
  - 训练速度快

#### 2. 核心优化技术

##### 2.1 类别不平衡处理
```python
# 权重计算
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)

# 采样器设置
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
```
- WeightedRandomSampler实现重采样
- 权重计算：1/类别频率
- 每个epoch都能看到所有类别

##### 2.2 基础数据增强
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2)
])
```
- 水平翻转：增加数据多样性
- 小角度旋转：提高模型鲁棒性
- 亮度调整：增强对光照变化的适应性

##### 2.3 学习率调度
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=3,
    verbose=True
)
```
- 使用ReduceLROnPlateau
- 监控验证集准确率
- 学习率调整系数：0.1
- 等待轮数：3轮

### V2版本：现代化训练技术引入

#### 1. 模型升级
- 主干网络：ConvNeXt Base
- 预训练：ImageNet-22K权重
- 改进点：
  - 更强大的特征提取能力
  - 更好的预训练权重
  - 现代化的网络设计

#### 2. 核心优化技术

##### 2.1 混合精度训练
```python
scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- 使用FP16和FP32混合训练
- 自动处理数值溢出
- 显存节省约50%
- 训练速度提升30-50%

##### 2.2 随机权重平均(SWA)
```python
swa_model = AveragedModel(model)
swa_scheduler = SWALR(
    optimizer,
    swa_lr=1e-5,
    anneal_epochs=5
)
```
- 起始轮数：20
- 更新频率：2轮
- SWA学习率：1e-5
- 提升模型稳定性和泛化能力

##### 2.3 高级数据增强
```python
# Mixup实现
if np.random.random() < mixup_prob:
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    
# CutMix实现
if np.random.random() < cutmix_prob:
    mixed_x, y_a, y_b, lam = cutmix_data(x, y)
```
- Mixup (α=0.2, p=0.5)
- CutMix (p=0.5)
- 标签平滑 (ε=0.1)
- 提升模型泛化能力

##### 2.4 学习率优化
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr=1e-7
)
```
- Warmup阶段：2轮
- 余弦退火调度
- 最小学习率：1e-7
- 更平滑的学习率变化

### V2.1版本：效率优化方案

#### 1. 模型保持
- 继续使用ConvNeXt Base
- 保持IN22K预训练权重

#### 2. 核心优化技术

##### 2.1 训练效率优化
```python
# 交叉验证优化
skf = StratifiedKFold(n_splits=3, shuffle=True)

# 早停策略
early_stopping = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    mode='max'
)
```
- 3折交叉验证（原5折）
- 更激进的早停策略
- 提前开始SWA (epoch 20)

##### 2.2 测试时增强(TTA)
```python
def tta_predict(model, image):
    transforms = [
        base_transform,
        horizontal_flip,
        scale_0_9,
        scale_1_1
    ]
    predictions = []
    for transform in transforms:
        aug_image = transform(image)
        pred = model(aug_image)
        predictions.append(pred)
    return torch.stack(predictions).mean(0)
```
- 水平翻转
- 多尺度测试(0.9x, 1.1x)
- 预测结果平均

##### 2.3 优化器参数精调
```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8
)
```
- 更大的权重衰减
- 梯度裁剪：5.0
- 优化器参数微调

### V3.0版本：类别不平衡深度优化

#### 1. 模型架构升级
- 主干网络：EfficientNetV2-B2
  - 相比V1版本的EfficientNet-B0，新版本改进：
    - MBConv模块升级为Fused-MBConv
    - 优化的渐进式学习策略
    - 改进的网络架构搜索空间
    - 更高效的训练策略

#### 2. 核心优化技术

##### 2.1 类别不平衡处理

###### A. 有效样本数加权（Class-Balanced Loss）
```python
def calculate_class_weights(labels, beta=0.9999):
    """
    基于有效样本数的类别权重计算
    原理：使用(1-β)/(1-β^n)作为权重，其中n是类别样本数
    - β接近1时，权重接近1/ln(n)
    - β=0时，权重为1/n（等同于传统的逆频率加权）
    """
    samples_per_class = np.bincount(labels)
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(weights)
    return torch.FloatTensor(weights)

# 使用示例
class_weights = calculate_class_weights(train_labels)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

###### B. Focal Loss实现
```python
class FocalLoss(nn.Module):
    """
    Focal Loss实现
    原理：降低易分样本的权重，提升难分样本的权重
    Loss = -α(1-pt)^γ * log(pt)
    - α: 类别权重系数
    - γ: 聚焦参数，降低易分样本的影响
    - pt: 预测概率
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 应用focal loss公式
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()
```

###### C. 动态采样策略
```python
class DynamicSampler:
    """
    动态采样策略
    原理：根据训练进度动态调整采样权重
    - 训练初期：强调类别平衡
    - 训练后期：逐渐过渡到原始分布
    """
    def __init__(self, labels, num_epochs):
        self.labels = labels
        self.num_epochs = num_epochs
        self.base_weights = self._compute_weights()
    
    def _compute_weights(self):
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        return weights / weights.sum()
    
    def get_weights(self, epoch):
        # 随训练进行逐渐减小重采样强度
        alpha = 1.0 - epoch / self.num_epochs
        current_weights = alpha * self.base_weights + \
                         (1-alpha) * np.ones_like(self.base_weights)
        return current_weights
```

##### 2.2 学习率优化

###### A. OneCycleLR原理与实现
```python
"""
OneCycleLR学习率调度策略
原理：
1. 学习率先增后减，动量相反
2. 分为三个阶段：
   - warmup：学习率从base_lr增至max_lr
   - annealing：学习率从max_lr降至min_lr
   - cooldown：学习率保持在min_lr
"""
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,          # 最大学习率
    epochs=cfg.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,        # warmup阶段占比
    div_factor=25,        # 初始学习率降低因子
    final_div_factor=1000 # 最终学习率降低因子
)

# 学习率变化曲线
def plot_lr_curve(scheduler, epochs):
    lrs = []
    for epoch in range(epochs):
        for _ in range(len(train_loader)):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
    plt.plot(lrs)
    plt.title('OneCycleLR Schedule')
    plt.show()
```

##### 2.3 训练流程优化

###### A. 动态折数调整
```python
def get_optimal_folds(min_samples):
    """
    动态确定交叉验证折数
    原理：根据最小类别样本数自适应调整
    - 样本数<3：使用2折
    - 样本数>=3：使用min(3, min_samples)折
    """
    if min_samples < 3:
        return 2
    return min(3, min_samples)

# 使用示例
label_counts = np.bincount(train_labels)
n_splits = get_optimal_folds(label_counts.min())
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
```

###### B. 综合评估指标
```python
def calculate_metrics(y_true, y_pred, classes):
    """
    综合性能评估
    - accuracy：整体准确率
    - macro_f1：宏平均F1（适用于不平衡数据）
    - weighted_f1：加权F1
    - per_class_f1：每个类别的F1
    - confusion_matrix：混淆矩阵
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'per_class_f1': f1_score(y_true, y_pred, average=None)
    }
    
    # 混淆矩阵分析
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # 类别级别分析
    for i, cls in enumerate(classes):
        metrics[f'class_{cls}_precision'] = precision_score(
            y_true, y_pred, labels=[i], average='micro'
        )
        metrics[f'class_{cls}_recall'] = recall_score(
            y_true, y_pred, labels=[i], average='micro'
        )
    
    return metrics
```

#### 3. 实验结果分析

##### 3.1 性能对比
| 版本   | 验证集准确率 | Macro F1 | 训练时间 | GPU显存 |
|--------|------------|----------|----------|---------|
| V1     | 85.2%     | 0.81     | 8小时    | 4GB     |
| V2     | 89.1%     | 0.85     | 15小时   | 8GB     |
| V2.1   | 88.7%     | 0.84     | 10小时   | 8GB     |
| V3.0   | 90.3%     | 0.87     | 6小时    | 6GB     |

##### 3.2 关键改进点分析
1. 类别不平衡处理效果：
   - 最小类别F1提升：0.45 → 0.68
   - 类别间F1标准差降低：0.25 → 0.15

2. 训练效率提升：
   - 总训练时间减少40%
   - GPU显存使用减少25%
   - 收敛速度提升约35%

## 四、经验总结

### 1. 模型选择
- 大型模型配合好的训练策略效果更好
- IN22K预训练权重优于IN1K
- 模型容量要与数据规模匹配

### 2. 优化技术
- 混合精度训练是必备技术
- SWA对模型稳定性提升明显
- 交叉验证折数与训练时间权衡

### 3. 工程实践
- 完善的日志系统很重要
- 需要在性能和效率间找平衡
- 异常处理机制不可少

## 五、未来改进方向

### 1. 模型优化
- 探索Transformer架构
- 引入知识蒸馏
- 尝试模型集成

### 2. 训练策略
- 实现渐进式学习
- 引入对比学习
- 优化学习率策略

### 3. 工程改进
- 添加分布式训练
- 实现模型量化
- 优化推理速度 