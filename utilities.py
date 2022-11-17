"""
@author: Tung-Yu Hsiao
"""


import numpy as np
import matplotlib.pyplot as plt


# reshape input image to (ponit_no, timestep)
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


def plot_rar(argmax_arr, x_size):
    argmax_plot = np.zeros((argmax_arr.shape[0], 2))
    for i in range(0, argmax_arr.shape[0]):
        y = 0
        while (True):
            if (argmax_arr[i] + 1 - y * x_size > x_size):
                y += 1
                continue
            else:
                x = argmax_arr[i] - y * x_size
                break
        argmax_plot[i, 0] = x
        argmax_plot[i, 1] = y
    return argmax_plot


# for data flattening
def data_flatten_and_concat(X_location, t_star):
    T = t_star.shape[0]
    N = X_location.shape[0]
    # data rearranging
    X = np.tile(X_location[:, 0:1], (1, T)) # N x T
    Y = np.tile(X_location[:, 1:2], (1, T))
    Z = np.tile(X_location[:, 2:3], (1, T))
    T = np.tile(t_star[:,0:1], (1,N)).T

    xx = X.flatten()[:, None] # NT x 1
    yy = Y.flatten()[:, None]
    zz = Z.flatten()[:, None]
    tt = T.flatten()[:, None]
    out = np.concatenate([xx,yy,zz, tt], axis = 1)
    return out



# data sampling 
def data_prep(X, N_train = 10000):
    N_sample = X.shape[0]
    idx = np.random.choice(N_sample, N_train, replace=False)
    X_out = np.empty((N_train, X.shape[1]))
    for i, id in enumerate(idx):
        X_out[i, :] = X[id, :]
    return X_out


# graph = [t_step, width x length ]
# frames -> "all" for plotting all subplots; int for single plot; ()for a range
def plot_reslts(graph, shape, title, n_frames = "all"):
    n = graph.shape[0]
    (width, height) = shape
    if n_frames == "all":
        for i, num in enumerate(range(0, n), start=1):
            num = str(num + 1)
            ax = plt.subplot(8, int(n / 8) + 1, i)
            plt.imshow(np.reshape(graph[i-1, :], (-1, height)), vmin=graph[i-1, :].min(),
                       vmax=graph[i-1, :].max(), cmap='jet')
            plt.title('Frame' + num, loc='center')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=cax)
        plt.suptitle(title)
        plt.savefig(title +  ".png", bbox_inches='tight', dpi=300)
        plt.show()

    if isinstance(n_frames, int):

        plt.imshow(np.reshape(graph[n_frames-1, :], (-1, height)), vmin=graph[n_frames-1, :].min(),
                   vmax=graph[n_frames-1, :].max(), cmap='jet')
        plt.colorbar()
        plt.title(title)
        plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
        plt.show()

    if len(n_frames) == 2:
        n = n_frames[1]-n_frames[0]+1
        for i, num in enumerate(range(0, n), start = 1):
            num = str(num+1)
            ax = plt.subplot(8, int(n/8)+1, i)
            plt.imshow(np.reshape(graph[i-1, :], (-1, height)), vmin=graph[i-1, :].min(),
                       vmax=graph[i-1, :].max(), cmap='jet')
            plt.title('Frame' + num, loc='center')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax = cax)
        plt.suptitle(title)
        plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
        plt.show()
