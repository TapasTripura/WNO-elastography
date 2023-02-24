# Wavelet-Neural-Operator-for-pdes
This repository contains the python codes of the paper 
  > + "Wavelet  Neural  Operator  for  solving  parametric  partialdifferential  equations  in  computational  mechanics  problems", authored by Tapas Tripura, Abhilash Awasthi, Sitikantha Roy, and Souvik Chakraborty.

# Architecture of the wavelet neural operator elastography (WNO-elastography). 
![WNO](WNO_elastography.png)

# Files
A short despcription on the files are provided below for ease of readers.
  + `wno_1d_Advection_time_III.py`: This code is for 1-D wave advection equation (time-dependent problem).
  + `wno_1d_Burger_discontinuous.py`: This code is for 1-D Burgers' equation with discontinuous field (time-dependent problem).
  + `wno_1d_Burgers.py`: This code is for 1-D Burger's equation (time-independent problem).
  + `Example_4_boucwen.py`: This code is for 2-D Allen-Cahn equation (time-independent problem).
  + `wno_2d_AC.py`: This code is for 2-D Darcy equation (time-independent problem).
  + `wno_2d_Darcy.py` contains useful functions, like, library construction, data-normalization.
  + `wno_2d_Darcy_notch.py`: This code is for 2-D Darcy equation in triangular domain with notch (time-independent problem).
  + `wno_2d_ERA5.py`: This code is for forecast of monthly averaged 2m air temperature (time-independent problem).
  + `wno_2d_ERA5_time.py`: This code is for weekly forecast of 2m air temperature (time-dependent problem).
  + `wno_2d_time_NS.py`: This code is for 2-D Navier-Stokes equation (2D time-dependent problem).
  + `utilities3.py` contains some useful functions (taken from [FNO paper](https://github.com/zongyi-li/fourier_neural_operator)).

# Library support
Following packages are required to be installed to run the above codes:
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + Xarray-Grib reader (To read ERA5 data in section 5) [Link-1](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html?highlight=install), [Link-2](https://anaconda.org/anaconda/xarray) 

Copy all the data in the folder 'data' and place the folder 'data' inside the same mother folder where the codes are present.	Incase, the location of the data are changed, the correct path should be given.

# Testing
For performing predictions on new inputs, one can use the 'WNO_testing_(.).py' codes given in the `Testing` folder. The trained models, that were used to produce results for the WNO paper can be found in the following link:
  > [Models](https://drive.google.com/drive/folders/1scfrpChQ1wqFu8VAyieoSrdgHYCbrT6T?usp=sharing)

# Dataset
  + The training and testing datasets for the (i) Burgers equation with discontinuity in the solution field (section 4.1), (ii) 2-D Allen-Cahn equation (section 4.5), and (iii) Weakly-monthly mean 2m air temperature (section 5) are available in the following link:
    > [Dataset](https://drive.google.com/drive/folders/1AnH7l9oeOgoLdZiIl5YDmyomZX-0_QPA?usp=sharing)
