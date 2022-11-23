"""
@author: Tung-Yu Hsiao
"""
import scipy.io
import copy
import numpy as np
from utilities import *
from pyDOE import lhs
from sklearn import preprocessing


class data_loader(object):
    # data = numpy array; (time_step, image_pixels)
    # sample_pixels = [num_height_pixels,num_width_pixels,num_depth_pixels]
    # sample_dims (actual sample size)= [height, width, depth]
    # t_span = (lb, ub)
    def __init__(self,data, sample_pixels,sample_dims,t_span):
        # pixel no
        self.height = sample_pixels[0]
        self.width = sample_pixels[1]
        self.depth = sample_pixels[2]
        # actual length
        len_h = sample_dims[0]
        len_w = sample_dims[1]
        len_d = sample_dims[2]

        # load data
        # data = (t, points)
        self.data = data
        
        # Normalization
        matrix = copy.deepcopy(self.normalize(self.data)[t_span[0]:t_span[1], :])
        total_t = t_span[1]-t_span[0]

        # Reshape matrix into (points, t) and store in U_star
        self.U_star = np.empty((matrix.shape[1], matrix.shape[0]))
        reshape(self.U_star, matrix, self.width, self.height, total_t)

        t_star = np.linspace(0, 1, total_t)
        self.t_star = t_star.reshape(-1, 1)  # T x 1
        self.sample_pixels = sample_pixels

        # grid interval
        self.x_interval = len_h / self.height
        self.y_interval = len_w / self.width
        self.z_interval = len_d / self.depth

        # ub, lb
        self.lb = np.array([0, 0, 0, 0])
        self.ub = np.array([((self.height - 1) * len_h / self.height),
                       ((self.width - 1) * len_w / self.width),
                       ((self.depth - 1) * len_d / self.depth), 1])


        # self.U = (pixel value)
        UU = self.U_star[:, :]  # N x T
        self.U = UU.flatten()[:, None]  # NT x 1
    def normalize(self, matrix):
        self.min = np.min(matrix)
        self.max = np.max(matrix)
        normalized = (matrix-self.min)/(self.max-self.min)
        return normalized
    def reverse_normalize(self, matrix):
        denormalized = matrix*(self.max-self.min)+self.min
        return denormalized
    # IR thermography only capture the surface temperature (z=0)
    def training_X(self, n_train):
        X_star = np.empty([self.U_star.shape[0], 3])
        index = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                X_star[index][0] = j * self.x_interval
                X_star[index][1] = i * self.y_interval
                X_star[index][2] = 0  # Z
                index += 1
        out = data_flatten_and_concat(X_star, self.t_star)
        self.X = np.concatenate([out, self.U], axis = 1)
        X_train = np.array(data_prep(self.X, n_train), dtype=np.float32)

        return X_train

    # thermograms location data
    def X_star(self):
        return self.X

    def real_u(self):
        return self.U
    # Latin Hypercube Sampling generated data
    def collocation_data(self, n_f):

        # Collocation data
        X_f = self.lb + (self.ub - self.lb) * lhs(4, n_f)  # lhs(shape)

        return np.array(X_f, dtype=np.float32)

    # 3 dimensional location data
    def generate_data(self):
        X_generated = np.empty([self.U_star.shape[0] * self.depth, 3])
        index = 0
        for i in range(0, self.depth):
            for j in range(0, self.width):
                for k in range(0, self.height):
                    X_generated[index][0] = k * self.x_interval
                    X_generated[index][1] = j * self.y_interval
                    X_generated[index][2] = i * self.z_interval
                    index += 1
        return X_generated

    def bc_data(self, n_bc):
        location = self.generate_data()
        xbc_idx = self.sample_bc(location, axis="x")
        ybc_idx = self.sample_bc(location, axis="y")
        zbc_idx = self.sample_bc(location, axis="z")
        x_bc = data_flatten_and_concat(location[xbc_idx], self.t_star)
        y_bc = data_flatten_and_concat(location[ybc_idx], self.t_star)
        z_bc = data_flatten_and_concat(location[zbc_idx], self.t_star)
        xbc_train = np.array(data_prep(x_bc, n_bc), dtype=np.float32)
        ybc_train = np.array(data_prep(y_bc, n_bc), dtype=np.float32)
        zbc_train = np.array(data_prep(z_bc, n_bc), dtype=np.float32)
        return xbc_train, ybc_train, zbc_train


    def sample_bc(self, location,axis):
        if axis == "x":
            bc_lb_idx = [idx for idx, output in enumerate(location) if location[idx, 0] == 0]
            bc_ub_idx = [idx for idx, output in enumerate(location) if location[idx, 0] == self.ub[0]]
            bc = bc_ub_idx+bc_lb_idx

        if axis == "y":
            bc_lb_idx = [idx for idx, output in enumerate(location) if location[idx, 1] == 0]
            bc_ub_idx = [idx for idx, output in enumerate(location) if location[idx, 1] == self.ub[1]]
            bc = bc_ub_idx + bc_lb_idx
        if axis == "z":
            bc_lb_idx = [idx for idx, output in enumerate(location) if location[idx, 2] == 0]
            bc_ub_idx = [idx for idx, output in enumerate(location) if location[idx, 2] == self.ub[2]]
            bc = bc_ub_idx + bc_lb_idx
        return bc


