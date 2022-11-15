# Usage
The Original thermogram data is 425 pixels in width and 617 pixels in length. 

Use the following code to load the .mat data:
```
import scipy.io
import numpy
data = scipy.io.loadmat(r'./defect1.mat')
original = np.array(data["matrix"])
```

# Reminder
The numpy reshape causes bug in reshaping the image data into (pixels, time_step)
Please use our reshape function in the following for image reshaping, you can also find it in the Utilities
```
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
