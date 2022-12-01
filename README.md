# PINN-and-thermography-processing
Adding physics information for an improved prediction (numerical simulation) of the IR thermography detection. 

For IR thermography, the tricky part of it is the explanation of the detection results. For most cases, there are many outside effects that could deter the detection. 
Such as backgroumd temperature, IR camera resoulution, measurment noises, and non-homogeniety of surface temperature etc. 

Here, we try to address the problem using the powerful neural network with the assist of physics information (i.e. the governing PDE) to propagate a solution from known thermograms data (2D data form). 

However, this arises a new problem. 
For practical heat diffusion condition, which is described by Fourier's heat equation:

$f(u(x,y,z,t)) = ∂u/∂t - lambda(∂^2u/∂x^2+∂^2u/∂y^2+∂^2u/∂z^2) $

is a 4 component PDE (time, x,y,z). 

Interestingly, thermography detection is only capable of surface temperature detection. In other words, it fails to provide information underneath.  

In this work, we proposed a physics-informed neural network (PINN) for thermographic data processing.
PINN not only inherits the prediction capabilities of deep neural networls but also condsider physical laws presented in the form of PDE. 
The methodology also perform the Fourier's Law PDE parameter discovery, in term realize the numerical simulation of the IR defect detection. 


![Project Scheme](https://github.com/dalenhsiao/PINN-and-thermography-processing/blob/main/Doc/project%20scheme.png)

# Requirements
- The code is built for Tensorflow 2.0 and above versions. 






# Assumptions 
- We assume the sound regions are much larger than defect regions, that is, the sound regions consist the majority of the training data. By that, the model predicted image shall represent the general background of the thermography data. 

- The sample is assumed to be a heat insulator, meaning there will be no heat exhange at the boundaries.

- The evaluated data is collected during the cooldown phase where there is no external heat source and only focuses on the heat diffusion inside the sample.



# Files
- ```PINN_model.py``` is the physics informed model for the cooldown phase of IR thermography, given the Fourier's Law of heat conduction. 

- ```data_loader.py``` loads and format the input training data for PINN model training. 

- ```utilities.py``` utilities. 

- ```specimen1_main.py``` is the main code for the PINN evaluation for specimen 1 (the ddescription for specimen 1 is in section [Specimen 1](#specimen-1)


# Specimen 1
The Original thermogram data is 425 pixels in width and 617 pixels in length and was sampled at a freqency 1 frame/sec.
IR experiment setup:

![Exp setup](https://github.com/dalenhsiao/PINN-and-thermography-processing/blob/main/Doc/specimen1.png)



![Raw thermograms](https://github.com/dalenhsiao/PINN-and-thermography-processing/blob/main/Doc/Raw_thermograms.gif)
### Data
Use the following code to load the .mat data:
```python
from data_loader import *
# data_loader(filepath, MAT_dict_label, #of sample pixels, sample real dimensions, inspect tspan)
dl = data_loader("./data/defect1.mat", "matrix",sample_pixels, sample_dims, tspan)
```
- Thermography Data: Data captured from Active thermography and translated into matrix
- Collocation Data: Data generated using Latin Hypercube Sampling (LHS) which is uniformly distributed in the (x,y,z,t) data space
- Boundary Condition: BC at each timestep





# Reminder
The numpy reshape causes bug in reshaping the image data into (pixels, time_step)

while our reshape function in the following could fix the unpleasant bug. 
It can be find in ```utilities.py```
```python
# The processed matrix_store.shape() = (width*length, time_step)
def reshape(matrix_store, matrix, x, y, T):
    count = 0
    idx = 0
    enum = 0
    for t in range(0, T):
        while True:
            if idx * x >= x * y:
                idx = 0
                count += 1
            if idx * x < x * y:
                matrix_store[enum, t] = matrix[t, idx * x + count]
                idx += 1
                enum += 1
            if count == x or enum == x * y:
                idx = 0
                count = 0
                enum = 0
                break;
```
