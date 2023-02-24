"""
This code belongs to the paper:
-- A wavelet neural operator based elastography for localization and
   quantification of tumors.
-- Authored by: Tapas Tripura, Abhilash Awasthi, Sitikantha Roy, 
   Souvik Chakraborty.
   
This code is for the BLOCK CNN implementation for Inclusion dataset.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)

# %%
class CNN2d(nn.Module):
    def __init__(self, width):
        super(CNN2d, self).__init__()

        self.width = width
        
        # forward layers,
        self.conv0 = nn.Conv2d(self.width, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 16, 3)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2)
        self.conv3 = nn.Conv2d(8, 4, 3)
        
        # transpose layers,
        self.conv4 = nn.ConvTranspose2d(4, 8, 3)
        self.conv5 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.conv6 = nn.ConvTranspose2d(16, 32, 3)
        self.conv7 = nn.ConvTranspose2d(32, self.width, 3, stride=2)

    def forward(self, x):
            
        # forward layers,
        x = self.conv0(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        
        # transpose layers,
        x = self.conv4(x)
        x = F.gelu(x)
        x = self.conv5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = F.gelu(x)
        x = self.conv7(x)

        return x

class Block_CNN(nn.Module):
    def __init__(self, width):
        super(Block_CNN, self).__init__()

        self.width = width
        self.padding = 0 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width) 

        self.conv0 = CNN2d(self.width)
        self.conv1 = CNN2d(self.width)
        self.conv2 = CNN2d(self.width)
        self.conv3 = CNN2d(self.width)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # do padding, if required
        
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)  
        x = self.conv3(x)

        # x = x[..., :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x
    

# %%
""" Model configurations """

PATH = 'data/DATA_inclusion_tension.mat'
ntrain = 440
ntest = 60

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75

width = 64

h = 51
s = 51

# %%
""" Read data """
reader = MatReader(PATH)

x_grid = reader.read_field('DATA1')[:,:,0,:]
y_grid = reader.read_field('DATA1')[:,:,1,:]
u_x = reader.read_field('DATA1')[:,:,2,:]
u_y = reader.read_field('DATA1')[:,:,3,:]
E_final = reader.read_field('DATA1')[:,:,-1,:]

x_grid = x_grid.permute(2,0,1)
y_grid = y_grid.permute(2,0,1)
u_x = u_x.permute(2,0,1)
u_y = u_y.permute(2,0,1)
E_final = E_final.permute(2,0,1)

ux_normalizer = UnitGaussianNormalizer(u_x)
u_x = ux_normalizer.encode(u_x)

uy_normalizer = UnitGaussianNormalizer(u_y)
u_y = uy_normalizer.encode(u_y)

x_grid = x_grid.unsqueeze(-1)
y_grid = y_grid.unsqueeze(-1)
u_x = u_x.unsqueeze(-1)
u_y = u_y.unsqueeze(-1)

data = torch.cat( (u_x,u_y,x_grid,y_grid), dim=-1).to(device)

y_normalizer = UnitGaussianNormalizer(E_final)
E_final = y_normalizer.encode(E_final)

x_train = data[:ntrain,:,:,:]
y_train = E_final[:ntrain,:,:]

x_test = data[-ntest:,:,:,:]
y_test = E_final[-ntest:,:,:]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = Block_CNN(width).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

epoch_loss = torch.zeros(epochs)
epoch_pred = torch.zeros(epochs,h,s)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
    
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    epoch_loss[ep] = train_l2
    epoch_pred[ep] = out[-7,:,:].detach().cpu()
    test_l2 /= ntest
    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
    
# %%
""" Prediction """
pred = torch.zeros(y_test.shape)
actual = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        pred[index] = out
        actual[index] = y

        test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
        test_e[index] = test_l2
        print(index, test_l2)
        index = index + 1

print('Mean Error:', 100*torch.mean(test_e))
    
# %%
""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
figure7 = plt.figure(figsize = (10, 5))
plt.subplots_adjust(hspace=0.01)
index = 0
for value in range(y_test.shape[0]):
    if value % 16 == 1:
        plt.subplot(2,5, index+1)
        plt.imshow(actual[value,:,:], label='True', cmap='seismic', interpolation='Gaussian')
        plt.title('Actual')
        plt.subplot(2,5, index+1+5)
        plt.imshow(pred.cpu().detach().numpy()[value,:,:], cmap='seismic', interpolation='Gaussian')
        # plt.imshow(pred[value,:,:], cmap='seismic')
        plt.title('Identified')
        plt.margins(0)
        index = index + 1
