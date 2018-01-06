import os
import torch
from torchvision import models

# extract vgg features
if __name__ == '__main__':
    save_fold = '../weights'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    vgg = models.vgg16(pretrained=True)
    torch.save(vgg.features.state_dict(), os.path.join(save_fold, 'vgg16_feat.pth'))
