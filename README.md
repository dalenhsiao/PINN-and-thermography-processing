# PINN-and-thermography-processing
Adding physics information for an improved prediction (simulation) of the IR thermography detection. 

For IR thermography, the tricky part of it is the explanation of the detection results. For most cases, there are many outside effects that could deter the detection. 
Such as backgroumd temperature, IR camera resoulution, measurment noises, and non-homogeniety of surface temperature etc. 

Here, we try to address the problem using the powerful neural network with the assist of physics information (i.e. the governing PDE) to propagate a solution from known thermograms data (2D data form). 

However, this arises a new problem. 
For practical heat diffusion condition, which is described by Fourier's heat equation, is a 4 components PDE (time, x,y,z). 
Interestingly, thermography detection is only capable of surface temperature detection. In other words, it fails to provide information underneath.  

