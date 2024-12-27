# %%
import pandas as pd
import random
import os
import sys
import numpy as np
import timm
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import GradScaler
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from PIL import Image

# %% [markdown]
# # Tensorboard

# %%
# def gen_Log():
#     current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_dir = os.path.join("runs", f"experiment_{current_time}")
#     writer = SummaryWriter(log_dir=log_dir)
#     return writer

# writer = gen_Log()

# %% [markdown]
# # CFG

# %%
class CFG:
    # model
    model_name = 'tf_efficientnet_b2'
    # criterion
    criterion_name = 'CrossEntropyLoss'
    
    # optimizer
    optimizer_name = 'AdamW'
    max_grad_norm = 1000
    gradient_accumulation_steps = 1
    # CosineAnnealingLR
    T_max = 5
    
    # scheduler
    scheduler_name = 'CosineAnnealingLR'
    
    # global settings
    apex = True
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    n_folds = 5
    used_folds = 5
    epochs = 1
    batch_size = 64
    num_classes = 44
    print_freq = 200
    # image settings
    width_map= {
        'tf_efficientnet_b2' :260,
        'convnext_base': 224,
        'convnext_small': 224
    }
    height_map ={
        'tf_efficientnet_b2' :260,
        'convnext_base': 224,
        'convnext_small': 224
    }
    width = width_map[model_name]
    height = height_map[model_name]

    # learning settings
    lr = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed=CFG.seed)

# %% [markdown]
# # 训练数据读入

# %%
loc = 'C:/code_in_laptop/d2l-zh/lab5/data/'
csv_file = loc + "train.csv"

train = pd.read_csv(csv_file)
Fold = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

train['fold'] = int(-1)
for fold, (train_index, val_index) in enumerate(Fold.split(train, train['target'])):
    train.loc[val_index, 'fold'] = fold

# %% [markdown]
# # 计算训练集各通道的mean和std值 

# %%
# def calculate_mean_std(data):
#     means = []
#     stds = []
#     for id in tqdm(data['id']):
#         train_img_path =f"/kaggle/input/boolart-image-classification/train_image/{id}.jpg"
#         image = Image.open(train_img_path).convert("RGB")
#         image = np.array(image.resize((CFG.width, CFG.height)))
#         means.append(np.mean(image,axis=(0,1)))
#         stds.append(np.std(image, axis=(0,1)))
#     mean = np.mean(means, axis=0)
#     std = np.mean(stds,axis=0)
#     return mean / 255, std / 255
# mean, std = calculate_mean_std(train)
# %store mean
# %store std
mean, std = ([0.8536320017130206, 0.8362727931350286, 0.8301507008641884],
             [0.23491626922733028, 0.24977155061784578, 0.25436414956228226])

# %% [markdown]
# # 数据增强

# %%
def get_transform(isTransform):
    if isTransform:
        return A.Compose([
            A.Resize(CFG.width, CFG.height),
            # 翻转
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            # 高斯噪声
            A.GaussNoise(p=0.2),
            # 模糊
            A.OneOf([
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
            ], p=0.2),
            # 变形
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=0.1),
            #     A.IAAPiecewiseAffine(p=0.3),
            # ], p=0.2),
            # 归一化
            A.Normalize(mean, std),
            # 遮挡
            A.CoarseDropout(max_holes=8,
                            max_height=int(CFG.height * 0.1),
                            max_width=int(CFG.width * 0.1),
                            p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CFG.width, CFG.height),
            A.Normalize(mean, std),
            ToTensorV2()
        ])

# %%
def show_transform_images(imgs, rows, cols):
    _, axes = plt.subplots(rows, cols, figsize=(rows*2, cols*2))
    axes = axes.flatten()
    torch_mean = torch.tensor(mean).view(-1, 1, 1).clone().detach()
    torch_std = torch.tensor(std).view(-1, 1, 1).clone().detach()
    for i ,(ax, img) in enumerate(zip(axes, imgs)):
        img = img * torch_std + torch_mean
        img = img.permute(1,2,0).numpy()
        ax.imshow(img)

# %% [markdown]
# # TrainDataSet

# %%
class TrainDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.labels = data['target'].values
        self.id = data['id'].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = loc + f"train/{self.id[index]}.jpg"
        image = np.array(Image.open(file_path).convert("RGB"))
        image = self.transform(image=image)['image']
        
        label = torch.tensor(self.labels[index]).long()
        return image, label

# %% [markdown]
# # Model

# %%
def get_model(model_name, pretrained=False):
    model = timm.create_model(model_name,pretrained=pretrained)
    
    if 'convnext_base' == model_name:
        model.head.fc = nn.Linear(in_features=1024, out_features=CFG.num_classes, bias=True)
    if 'convnext_small' == model_name:
        model.head.fc = nn.Linear(in_features=1024, out_features=CFG.num_classes, bias=True)
    if 'tf_efficientnet_b2' == model_name:
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=CFG.num_classes,bias=True)
    return model

# %% [markdown]
# # Criterion

# %%
def get_criterion(criterion_name):
    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

# %% [markdown]
# # Scheduler & Optimizer

# %%
def get_optimizer_scheduler(model, optimizer_name, scheduler_name):
    optimizer = None
    scheduler = None
    
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    if optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    if scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    return optimizer, scheduler

# %% [markdown]
# # Train_fn

# %%
def train_fn(train_loader, model, criterion, optimizer, epoch, device):
    if CFG.apex:
        scaler = GradScaler()
    model.train()
    
    for step,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_pres = model(images)
        loss = criterion(y_pres, labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss /gradient_accumulation_steps
        if CFG.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss: {loss:.4f}'
                  .format(
                   epoch+1, step, len(train_loader),loss=loss.item(),
                   )) 

# %% [markdown]
# # Valid_fn

# %%
def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    preds = []
    acc = 0.0
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # TTA
        with torch.no_grad():
            outputs1 = model(images)
            outputs2 = model(images.flip(-1))
            outputs3 = model(images.flip(-2))
            outputs4 = model(images.flip([-2, -1]))
            outputs5 = model(images.flip(-1).flip([-2, -1]))
            outputs6 = model(images.flip(-2).flip([-2, -1]))
            outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5 + outputs6) / 6
            loss = criterion(outputs, labels.long())
            _, predict_y = torch.max(outputs, dim=1)
            acc += (predict_y.to(device) == labels.to(device)).sum().item()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Loss: {loss:.4f}'
                  .format(step, len(valid_loader), loss=loss.item(),
                   ))
    return loss, acc
        
    

# %% [markdown]
# # Train

# %%
def train_pipeline(model, criterion, optimizer, scheduler, device):
    model.to(device)
    best_score = 0.
    best_loss = np.inf
    for fold_id in range(CFG.used_folds):
        train_index = train[train['fold'] != fold_id].index
        valid_index = train[train['fold'] == fold_id].index

        train_folds = train.loc[train_index].reset_index(drop=True)
        valid_folds = train.loc[valid_index].reset_index(drop=True)

        valid_labels = valid_folds['target'].values

        train_dataset = TrainDataset(train_folds,
                                     transform=get_transform(isTransform=True))
        valid_dataset = TrainDataset(valid_folds,
                                     transform=get_transform(isTransform=False))

        train_loader = DataLoader(train_dataset,
                                  batch_size=CFG.batch_size,
                                  shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=CFG.batch_size * 2,
                                  shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)

        print(f"FOLD_ID:{fold_id + 1}")
        best_score=0.
        best_loss=np.inf
        for epoch in range(CFG.epochs):
            best_score = 0.
            best_loss = np.inf
            # train
            train_fn(train_loader=train_loader,
                     model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     epoch=epoch,
                     device=device)
            # eval
            avg_val_loss, acc = valid_fn(valid_loader=valid_loader,
                                         model=model,
                                         criterion=criterion,
                                         device=device)
            acc = acc / len(valid_dataset)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            print(f"epoch:{epoch + 1}，acc:{acc}")

            if acc > best_score:
                best_score = acc
                torch.save({'model': model.state_dict(),
                            'preds': acc},
                           loc + "model/" + f'{CFG.model_name}_fold{fold_id + 1}_best_score.pth')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({'model': model.state_dict(),
                            'preds': acc},
                           loc + "model/" + f'{CFG.model_name}_fold{fold_id + 1}_best_loss.pth')

# %%
model = get_model(CFG.model_name, pretrained=True)
criterion = get_criterion(CFG.criterion_name)
optimizer ,scheduler = get_optimizer_scheduler(model=model, optimizer_name=CFG.optimizer_name, scheduler_name=CFG.scheduler_name)

# %%
train_pipeline(model, criterion, optimizer, scheduler, CFG.device)

# %% [markdown]
# # Inference

# %%

test_data = pd.read_csv(loc + "sample_submission.csv")

class TestDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data['id'].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.file_path = loc + f"test/{self.data[idx]}.jpg"
        image = np.array(Image.open(self.file_path).convert("RGB"))
        image = self.transform(image=image)['image']
        return image,self.data[idx]

# %%
def inference(model, models_path, test_loader, device):
    pre = []
    image_id = []
    for i, (images,img_ids) in enumerate(test_loader):
        image_id += list(img_ids.numpy())
        images = images.to(device)
        for model_path in models_path:
            model.load_state_dict(torch.load(model_path)['model'])
            model.to(device)
            model.eval()
            with torch.no_grad():
                y_preds1 = F.softmax(model(images))
                y_preds2 = F.softmax(model(images.flip(-1)))
                y_preds3 = F.softmax(model(images.flip(-2)))
                y_preds4 = F.softmax(model(images.flip([-2, -1])))
                y_preds5 = F.softmax(model(images.flip(-1).flip([-2, -1])))
                y_preds6 = F.softmax(model(images.flip(-2).flip([-2, -1])))
            y_preds = (y_preds1.to('cpu').numpy() + y_preds2.to('cpu').numpy() +
                       y_preds3.to('cpu').numpy() + y_preds4.to('cpu').numpy() + y_preds5.to(
                        'cpu').numpy() + y_preds6.to('cpu').numpy()) / 6
        avg_preds = F.softmax(torch.from_numpy(y_preds),dim=1)
        _,predict_y = torch.max(avg_preds,dim = 1)
        predict_y = np.array(predict_y).tolist()
        pre += predict_y
    return pre,image_id

# %%
test_dataset = TestDataset(test_data, transform=get_transform(isTransform=False))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
def get_best_model(model_name):
    best_acc = 0.
    best_model_pth = ""
    for fold_id in range(CFG.used_folds):
        checkpoint = torch.load(loc + "model/" + f'{model_name}_fold{fold_id + 1}_best_score.pth')
        if(checkpoint['preds'] > best_acc):
            best_acc = checkpoint['preds']
            best_model_pth = loc + "model/" + f'{model_name}_fold{fold_id + 1}_best_score.pth'
    return best_model_pth
        
best_model_pth = get_best_model(CFG.model_name)

# %%
best_model_pth

# %%
predictions,img_id = inference(model, [best_model_pth], test_loader, CFG.device)

# %% [markdown]
# # Submission
# 

# %%
df = pd.DataFrame({
    "id": img_id,
    "predict": predictions
})
df.to_csv(loc + "sub/" + "submission.csv", index=False)
df


