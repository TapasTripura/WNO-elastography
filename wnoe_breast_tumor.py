"""
This code belongs to the paper:
-- A wavelet neural operator based elastography for localization and
   quantification of tumors.
-- Authored by: Tapas Tripura, Abhilash Awasthi, Sitikantha Roy, 
   Souvik Chakraborty.
   
This code is for the Wavelet Neural Operator - elastography for Gaussian with location and width change dataset.
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
""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        dwt_ = DWT(J=self.level, mode='symmetric', wave='db6').to(dummy.device)
        self.mode_data, _ = dwt_(dummy)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave='db6').to(x.device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft[:, :, :, :] = self.mul2d(x_ft[:, :, :, :], self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)

        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db6').to(x.device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO2d, self).__init__()

        """
        The WNO-elastography architecture.
        
        input: the solution of the coefficient function and locations (u(x), u(y), x, y)
        input shape: (batchsize, x=s, y=s, c=4)
        ~~~
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# %%
""" Model configurations """

PATH = 'data/DATA_Tumor.mat'
ntrain = 440
ntest = 60

batch_size = 20
learning_rate = 0.001

epochs = 1000
step_size = 50
gamma = 0.75

level = 3
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
model = WNO2d(width, level, x_train.permute(0,3,1,2)).cuda()
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
        x, y = x.cuda(), y.cuda()

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
            x, y = x.cuda(), y.cuda()

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
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).reshape(s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        pred[index] = out
        actual[index] = y

        test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
        print(index, test_l2)
        index = index + 1
        
    
# %%
""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
figure7 = plt.figure(figsize = (10, 5))
plt.subplots_adjust(hspace=0.01)
index = 0
for value in range(y_test.shape[0]):
    if value % 18 == 1:
        plt.subplot(2,5, index+1)
        plt.imshow(actual[value,:,:], label='True', cmap='seismic', interpolation='Gaussian')
        plt.title('Actual')
        plt.subplot(2,5, index+1+5)
        plt.imshow(pred.cpu().detach().numpy()[value,:,:], cmap='seismic', interpolation='Gaussian')
        # plt.imshow(pred[value,:,:], cmap='seismic')
        plt.title('Identified')
        plt.margins(0)
        index = index + 1

# %%
"""
For saving the trained model and prediction data
"""
# torch.save(model, 'model/data_tumor_500samples')
# scipy.io.savemat('pred/data_tumor_500samples.mat', mdict={'pred': pred.cpu().numpy()})

# scipy.io.savemat('epoch_loss/epoch_loss_tumor_500samples.mat', mdict={'epoch_loss': epoch_loss.cpu().numpy()})
# scipy.io.savemat('epoch_prediction/epoch_pred_tumor_500samples.mat', mdict={'epoch_pred': epoch_pred.cpu().numpy()})

