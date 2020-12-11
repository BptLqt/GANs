import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn

batch_size = 256
trans = transforms.Compose([transforms.ToTensor()])

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=True)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=True)

idx = train_set.train_labels==4
train_set.data = train_set.train_data[idx]

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)


net_D = nn.Sequential(nn.Linear(784,256), nn.ReLU(),
                     nn.Linear(256,128), nn.ReLU(),
                     nn.Linear(128, 1), nn.Sigmoid())
net_G = nn.Sequential(nn.Linear(256, 512), nn.ReLU(),
                     nn.Linear(512,1024), nn.ReLU(),
                     nn.Linear(1024,784), nn.Tanh())


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
net_G.apply(init_weights)
net_D.apply(init_weights)

optim_D = torch.optim.Adam(net_D.parameters(), lr=0.0002)
optim_G = torch.optim.Adam(net_G.parameters(), lr=0.0002)
criterion = nn.BCELoss()

epochs = 100
fixed_z = torch.randn(5,256)
imgs = []
for epoch in range(epochs):
    av_loss_D, av_loss_G = 0,0
    for i, (X,_) in enumerate(train_loader):
        z = torch.randn(X.shape[0],256)
        y_ones = torch.ones_like(_).float()
        y_zeros = torch.zeros_like(_).float()
        y_ones_hat = net_D(X.view(-1,784))
        
        real_loss_D = criterion(y_ones_hat, y_ones.view(y_ones_hat.shape))
        
        fake_X = net_G(z).detach()
        y_zeros_hat = net_D(fake_X)
        fake_loss_D = criterion(y_zeros_hat, y_zeros.view(y_zeros_hat.shape))
        
        loss_D = real_loss_D + fake_loss_D
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        
        
        new_X = net_G(z)
        D_new_X = net_D(new_X)
        loss_G = criterion(D_new_X,y_ones.reshape(D_new_X.shape))
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()
        
        av_loss_D += loss_D
        av_loss_G += loss_G
    imgs.append(net_G(fixed_z))
    print("epoch : {}, av loss D : {}, av loss G : {}".format(epoch+1,av_loss_D/i,av_loss_G/i))


for i in imgs:
    plt.imshow(i[0].detach().numpy().reshape(28,28),cmap="gray")
    plt.show()

import matplotlib.pyplot as plt
imgs2 = net_G(torch.randn(5,256))
net_D(img)
for i in img:
  plt.imshow(i.detach().numpy().reshape(28,28),cmap='gray')
  plt.show()
