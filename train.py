import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms

from torchvision import datasets

import matplotlib.pyplot as plt



# 定义超参数

batch_size = 64

learning_rate = 0.01

num_epoches = 20



def to_np(x):

    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集

train_dataset = datasets.MNIST(

    root='./data', train=True, transform=transforms.ToTensor(), download=True)



test_dataset = datasets.MNIST(

    root='./data', train=False, transform=transforms.ToTensor())



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# 定义 Convolution Network 模型

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



model = Cnn(1, 10)

use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速

if use_gpu:

    model = model.cuda()

criterion = nn.CrossEntropyLoss()   #交叉熵

optimizer = optim.SGD(model.parameters(), lr=learning_rate) #随机梯度下降



# 开始训练
loss_count = []
for epoch in range(num_epoches):
    for i,(x,y) in enumerate(train_loader):

        batch_x = Variable(x)
        batch_y = Variable(y)

        # 获取最后输出
        out = model(batch_x)

        # 获取损失
        loss = criterion(out,batch_y)
        # 使用优化器优化损失

        optimizer.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        optimizer.step() # 将参数更新值施加到net的parmeters上

        if i%20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
            torch.save(model.state_dict(), './cnn.pth')

        if i % 100 == 0:
            for a,b in test_loader:

                test_x = Variable(a)
                test_y = Variable(b)
                out = model(test_x)

                #验证准确率
                accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                print('accuracy:\t',accuracy.mean())
                break

plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count,label='Loss')
plt.legend()
plt.show()


