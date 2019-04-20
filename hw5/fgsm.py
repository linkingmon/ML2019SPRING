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
import sys

# set model
model = resnet50(pretrained = True)

model.eval()

cross_entropy = nn.CrossEntropyLoss()

# read true label
true_labels = pd.read_csv('labels.csv').values[:, 3]

# find image path
path = sys.argv[1]
files = []
for r, d, f in os.walk(path):
    for file in f:
        if(file[3:7] == '.png'):
            files.append('/' + file)
files.sort()

import os
newdir = sys.argv[2]
if not os.path.exists(newdir):
    os.makedirs(newdir)

def fgsm(model, num, file, epsilon):
    label = true_labels[num]
    label = torch.tensor(label)
    img = Image.open(path + file)

    trans = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = trans(img)
    img = img.unsqueeze(0)
    img.requires_grad = True

    label = label.unsqueeze(0).cuda()

    zero_gradients(img)
    output = model(img).cuda()
    loss = cross_entropy(output, label).cuda()
    loss.backward()
    sign_grad = img.grad.sign()
    img.data = img + epsilon * sign_grad
    
    attack = model(img).cuda()
    res = (label.cpu().numpy() != (torch.argmax(attack)).cpu().numpy())
    return res, img

def deprocess(img):
    img[0][0] = img[0][0] * 0.229 + 0.485
    img[0][1] = img[0][1] * 0.224 + 0.456
    img[0][2] = img[0][2] * 0.225 + 0.406
    return img

for num, file in enumerate(files):
    res, img = fgsm(model = model, num = num, file = file, epsilon = 0.3)

    deprocess(img)    
    img = img.squeeze(0)
    torchvision.utils.save_image(img, newdir + file)
