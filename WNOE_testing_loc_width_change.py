"""
This code belongs to the paper:
-- A wavelet neural operator based elastography for localization and
   quantification of tumors.
-- Authored by: Tapas Tripura, Abhilash Awasthi, Sitikantha Roy, 
   Souvik Chakraborty.
   
This code is for the Wavelet Neural Operator - elastography for Gaussian with location and width change dataset.
"""

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

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

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

PATH = 'data/DATA_Gaussian_loc_width_change.mat'

ntrain = 10
ntest = 40

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75

level = 3
width = 48

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

model = torch.load('model/model_Gaussian_loc_width')
print(count_params(model))

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

# %%
""" Prediction """
pred = torch.zeros(y_test.shape)
actual = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape)
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
        test_e[index] = test_l2
        print(index, test_l2)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

u_x = ux_normalizer.decode(u_x.squeeze(-1))
u_y = uy_normalizer.decode(u_y.squeeze(-1))

figure1 = plt.figure(figsize = (22, 24))
plt.subplots_adjust(hspace=0.1, wspace=0.4)
index = 1
for value in range(y_test.shape[0]):
    if value % 10 == 1:
        print(value)
        plt.subplot(5,4, index)
        plt.imshow(u_x.cpu()[value,:,:], cmap='jet', extent=[0,20,0,20],
                   interpolation='Gaussian')
        plt.title('Case-{}'.format(index))
        plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(fraction=0.046)
        if index == 1:
            plt.text(-9,8,'$u_{xx}$', rotation=90, fontsize=30, color='r', fontweight='bold')
        
        plt.subplot(5,4, index+4)
        plt.imshow(u_y.cpu()[value,:,:], cmap='jet', extent=[0,20,0,20],
                    interpolation='Gaussian')
        plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(fraction=0.046)
        if index == 1:
            plt.text(-9,8,'$u_{yy}$', rotation=90, fontsize=30, color='c', fontweight='bold')
        
        plt.subplot(5,4, index+8)
        plt.imshow(actual.cpu()[value,:,:], cmap='jet',
                    extent=[0,20,0,20], interpolation='Gaussian', vmin=0.01, vmax=0.015)
        plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(fraction=0.046)
        if index == 1:
            plt.text(-9,5,'E (Actual)', rotation=90, fontsize=30, color='m', fontweight='bold')
        
        plt.subplot(5,4, index+12)
        plt.imshow(pred.cpu()[value,:,:], cmap='jet',
                    extent=[0,20,0,20], interpolation='Gaussian', vmin=0.01, vmax=0.015)
        plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(fraction=0.046)
        if index == 1:
            plt.text(-9,3,'E (Predicted)', rotation=90, fontsize=30, color='g', fontweight='bold')
        
        plt.subplot(5,4, index+16)
        plt.imshow(torch.abs(pred[value,:,:]-actual[value,:,:])/torch.abs(actual[value,:,:]).cpu(),
                    cmap='jet', extent=[0,20,0,20], interpolation='Gaussian',
                    vmin=0, vmax=1e-2)
        plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(fraction=0.046)
        if index == 1:
            plt.text(-9,7,'Error', rotation=90, fontsize=30, color='b', fontweight='bold')
        
        plt.margins(0)
        index = index + 1

# figure1.savefig('center_width_51.pdf', format='pdf', dpi=500, bbox_inches='tight')
