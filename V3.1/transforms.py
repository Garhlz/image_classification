import torchvision.transforms as transforms

def get_train_transforms(image_size):
    """获取训练数据增强"""
    height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8536320017130206, 0.8362727931350286, 0.8301507008641884],
            std=[0.23491626922733028, 0.24977155061784578, 0.25436414956228226]
        )
    ])

def get_val_transforms(image_size):
    """获取验证数据增强"""
    height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8536320017130206, 0.8362727931350286, 0.8301507008641884],
            std=[0.23491626922733028, 0.24977155061784578, 0.25436414956228226]
        )
    ])
