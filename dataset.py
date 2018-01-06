import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


# for ECSSD dataset
class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform, t_transform):
        self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        self.label_path = list(map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, mode='train', num_thread=1):
    shuffle = False
    mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
    if mode == 'train':
        transform = transforms.zenCompose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x - mean)
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x - mean)
        ])
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
    dataset = ImageData(img_root, label_root, transform, t_transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader

