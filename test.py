import time

import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import cv2
import torch.nn.functional as F

from torch import nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms


img = cv2.imread("1.png")  # 读取要预测的图片

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图

img = np.array(img).astype(np.float32)

img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]

img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]

img = Variable(torch.from_numpy(img))


class Cnn(nn.Module):

    def __init__(self, in_dim, n_class):

        super(Cnn, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),

            nn.ReLU(True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5, stride=1, padding=0),

            nn.ReLU(True), nn.MaxPool2d(2, 2))



        self.fc = nn.Sequential(

            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))



    def forward(self, x):

        out = self.conv(x)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


net = Cnn(1, 10)

net.load_state_dict(torch.load('./cnn.pth'))

net.eval() #测试模式

outputs = net(img)

print(outputs)

outputs=torch.abs(outputs)

_, pred = torch.max(outputs, 1)  #获取最大概率类别

print(pred)