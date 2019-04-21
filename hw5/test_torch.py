import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169

# deprocess image
def deprocess_input(x):
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,0] * 0.229 + 0.485
    t[:,:,1] = x[0][:,:,1] * 0.224 + 0.456
    t[:,:,2] = x[0][:,:,2] * 0.225 + 0.406
    return np.floor(np.clip(t*255, 0, 255))

# inf norm function
def cal_inf_norm(img, input):
    return np.max(np.abs(img-input))

# set model
model = resnet50(pretrained = True)
model.eval()
cross_entropy = nn.CrossEntropyLoss()

# read true label
true_labels = pd.read_csv('labels.csv').values[:, 3]

# find image path
path = 'images/'
files = []
for r, d, f in os.walk(path):
    for file in f:
        files.append(file)
files.sort()
newdirs = ['vgg16/','vgg19/','res101/','d121/','d169/']

for newdir in newdirs:
    acc = infnorm = 0
    for num, file in enumerate(files):
        print(num)
        label = true_labels[num]
        img = Image.open(path + file)
        img_attack = Image.open(newdir + file)
        trans = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        imArr = np.fromstring(img.tobytes(), dtype=np.uint8)
        imArr = imArr.reshape((img.size[1], img.size[0], 3)).astype('int')
        imArr_attack = np.fromstring(img_attack.tobytes(), dtype=np.uint8)
        imArr_attack = imArr_attack.reshape((img_attack.size[1], img_attack.size[0], 3)).astype('int')
        norm = cal_inf_norm(imArr, imArr_attack)
        infnorm += norm

        img_attack = trans(img_attack)
        img_attack = img_attack.unsqueeze(0)
        attack = model(img_attack).cuda()
        res = (label != (torch.argmax(attack)).cpu().numpy())
        acc += res
        # print(num, norm, res)
    print(newdir)
    print("   ATTACK ACC:", acc/200)
    print("   INF NORM:", infnorm/200)


