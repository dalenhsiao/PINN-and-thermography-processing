# PINN-and-thermography-processing
Adding physics information for an improved prediction (simulation) of the IR thermography detection. 

For IR thermography, the tricky part of it is the explanation of the detection results. For most cases, there are many outside effects that could deter the detection. 
Such as backgroumd temperature, IR camera resoulution, measurment noises, and non-homogeniety of surface temperature etc. 

Here, we try to address the problem using the powerful neural network with the assist of physics information (i.e. the governing PDE) to propagate a solution from known thermograms data (2D data form). 

However, this arises a new problem. 
For practical heat diffusion condition, which is described by Fourier's heat equation, is a 4 components PDE (time, x,y,z). 
Interestingly, thermography detection is only capable of surface temperature detection. In other words, it fails to provide information underneath.  

In this work, we proposed a physics-informed neural network (PINN) for thermographic data processing.
PINN not only inherits the prediction capabilities of deep neural networls but also condsider physical laws presented in the form of PDE. 
The methodology also perform the Fourier's Law PDE parameter discovery, in term realize the numerical simulation of the IR defect detection. 


# Usage
The Original thermogram data is 425 pixels in width and 617 pixels in length.

![Raw thermograms](https://github.com/dalenhsiao/PINN-and-thermography-processing/blob/main/Doc/ezgif.com-gif-maker%20(1).gif)
### Data
Use the following code to load the .mat data:
```python
import scipy.io
import numpy
data = scipy.io.loadmat(r'./defect1.mat')
original = np.array(data["matrix"])
```
- Thermography Data: Data captured from Active thermography and translated into matrix
- Collocation Data: Data generated using Latin Hypercube Sampling (LHS) which is uniformly distributed in the (x,y,z,t) data space
- Boundary Condition: BC at each timestep

### Model Structure
The physics information input as PDE combine with a deep neural netwrok structure




# Reminder
The numpy reshape causes bug in reshaping the image data into (pixels, time_step)
Please use our reshape function in the following for image reshaping, you can also find it in the Utilities
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
