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


img = cv2.imread("7.png")  # 读取要预测的图片

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图

img = np.array(img).astype(np.float32)


img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]

img = Variable(torch.from_numpy(img))

img = img.view(img.size(0), -1)




class logsticRegression(nn.Module):

    def __init__(self, in_dim, n_class):

        super(logsticRegression, self).__init__()

        self.logstic = nn.Linear(in_dim, n_class)



    def forward(self, x):

        out = self.logstic(x)

        return out

net = logsticRegression(28 * 28, 10)

net.load_state_dict(torch.load('./logstic.pth'))
net.eval()

torch.no_grad()


outputs = net(img)

print(outputs)

outputs=torch.abs(outputs)

_, pred = torch.max(outputs, 1)

print(pred)