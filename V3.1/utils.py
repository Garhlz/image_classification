import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelEMA:
    """模型指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        if len(self.shadow) == 0:
            self.register()
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha[target]
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, cfg, filename='checkpoint.pth'):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'config': cfg.__dict__
    }
    save_path = os.path.join(cfg.output_dir, filename)
    torch.save(checkpoint, save_path)
    logging.info(f"保存检查点到: {save_path}")

def load_checkpoint(model, optimizer, scheduler, scaler, cfg, filename='checkpoint.pth'):
    """加载检查点"""
    checkpoint_path = os.path.join(cfg.output_dir, filename)
    if not os.path.exists(checkpoint_path):
        logging.warning(f"检查点文件不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logging.info(f"从 {checkpoint_path} 加载检查点，epoch: {checkpoint['epoch']}")
    return checkpoint['epoch']
