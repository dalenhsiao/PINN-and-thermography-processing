"""
@author: Tung-Yu Hsiao
"""
import numpy as np
from data_loader import *
from PINN_model import *
import os

def cf(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


sample_pixels = (617,425, 100)
sample_dims = (0.18, 0.18, 0.052)
tspan = (9, 82)
pnts_no = sample_pixels[0]*sample_pixels[1]
total_t = tspan[1]-tspan[0]

# data loading
n_train = 100000
n_f = 10000
n_bc = 10000
dl = data_loader("./data/defect1.mat", sample_pixels, sample_dims, tspan)
X_train = dl.training_X(n_train)
X_f = dl.collocation_data(n_f)
x_bc, y_bc, z_bc = dl.bc_data(n_bc)
lb, ub = dl.lb, dl.ub
# real
X_star = dl.X_star()
U = dl.real_u()


# Training

N_iter = 100000
lambda_array = np.zeros(N_iter+2, dtype=float)
loss_array = np.zeros(N_iter+2,dtype=float)

# Model default stopped at early stopping 1000 steps amd save best paramters only
layers = [4, 30, 30, 30, 30, 30, 30, 1]
model = PhysicsInformedNN(X_train, X_f, lb, ub,x_bc, y_bc, z_bc,layers)
model.summary()

Dir = "./Results/"
cf(Dir)
model.train(N_iter, loss_array, lambda_array, fileName=Dir+"training")
model.save(Dir+"model.pkl")

# Prediction
u_pred = model.predict(X_star)
u_diff = U-u_pred

u_pred = u_pred.reshape(pnts_no, total_t).T
u_diff = u_diff.reshape(pnts_no, total_t).T

# Plot result
for i in range(0, total_t):
    plt.subplot(10, 8, i + 1)
    plt.imshow(np.reshape(u_pred[i, :], (425, 617)), vmin=u_pred[i, :].min(),
               vmax=u_pred[i, :].max(), cmap='jet')
    plt.colorbar()
plt.suptitle("U prediction")
plt.savefig(Dir+"background.png", bbox_inches='tight', dpi=300)
plt.show()

for i in range(0, total_t):
    plt.subplot(10, 8, i + 1)
    plt.imshow(np.reshape(u_diff[i, :], (425, 617)), vmin=u_diff[i, :].min(),
               vmax=u_diff[i, :].max(), cmap='jet')
    plt.colorbar()
plt.suptitle("U difference")
# plt.savefig(Dir+"diff.png", bbox_inches='tight', dpi=300)
plt.show()





