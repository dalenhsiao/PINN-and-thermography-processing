import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from pyDOE import lhs
import copy
import pickle
import scipy.optimize as sopt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import mean_squared_error
import math


def cf(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

cf("./training_data")
tf.compat.v1.set_random_seed(1234)


class PhysicsInformedNN:
    # INPUT X = (x,y,z,t,u)
    # INPUT BC = (x,y,z,t,u)
    # Initialize the class
    # t step for gru
    def __init__(self, X, X_f,lb,ub, xbc, ybc, zbc, layers,load_model = 0, ModelDir = ""):

        tf.compat.v1.disable_eager_execution()


        self.lb= lb
        self.ub = ub

        self.X = X



        self.count = 0

        self.X_f = X_f

        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]
        self.z_f = X_f[:,2:3]
        self.t_f = X_f[:,3:4]

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.z = X[:, 2:3]
        self.t = X[:, 3:4]
        self.u = X[:, 4:5]

        self.x_x_bc = xbc[:, 0:1]
        self.x_y_bc = xbc[:, 1:2]
        self.x_z_bc = xbc[:, 2:3]
        self.x_t_bc = xbc[:, 3:4]

        self.y_x_bc = ybc[:, 0:1]
        self.y_y_bc = ybc[:,1:2]
        self.y_z_bc = ybc[:, 2:3]
        self.y_t_bc = ybc[:, 3:4]


        self.z_x_bc = zbc[:, 0:1]
        self.z_y_bc = zbc[:, 1:2]
        self.z_z_bc = zbc[:, 2:3]
        self.z_t_bc = zbc[:, 3:4]


        # Initiate layers
        # load old model or create a new one
        #　for old model the structure must match
        if load_model:
            self.layers, self.weights, self.biases, self.lambda_1 = self.load_NN(ModelDir)
        #　create new model
        else:
            self.weights, self.biases = self.initialize_NN(layers)
            # Initialize parameters
            self.lambda_1 = tf.Variable([0.000001], dtype=tf.float32)
            self.layers = layers



        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=True))
        # training data
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])


        # Collocation data
        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.z_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])





        # BC data
        self.x_x_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_x_bc.shape[1]])
        self.x_y_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_y_bc.shape[1]])
        self.x_z_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_z_bc.shape[1]])
        self.x_t_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_t_bc.shape[1]])

        self.y_x_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_x_bc.shape[1]])
        self.y_y_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_y_bc.shape[1]])
        self.y_z_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_z_bc.shape[1]])
        self.y_t_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_t_bc.shape[1]])

        self.z_x_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_x_bc.shape[1]])
        self.z_y_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_y_bc.shape[1]])
        self.z_z_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_z_bc.shape[1]])
        self.z_t_bc_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_t_bc.shape[1]])




        # My Code
        # u prediction
        self.u_pred, _, _, _ = self.net_u(self.x_tf, self.y_tf, self.z_tf, self.t_tf)
        # bc loss 1st derivative
        _, self.x_bc_loss, _, _ = self.net_u(self.x_x_bc_tf, self.x_y_bc_tf, self.x_z_bc_tf, self.x_t_bc_tf)
        _, _, self.y_bc_loss, _ = self.net_u(self.y_x_bc_tf, self.y_y_bc_tf, self.y_z_bc_tf, self.y_t_bc_tf)
        _,_,_,self.z_bc_loss = self.net_u(self.z_x_bc_tf, self.z_y_bc_tf, self.z_z_bc_tf, self.z_t_bc_tf)
        # PDE loss
        self.f_u_pred = self.net_u(self.x_f_tf, self.y_f_tf, self.z_f_tf, self.t_f_tf, collocation=True)




        # loss function
        self.loss = (tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.x_bc_loss)) +\
                    tf.reduce_mean(tf.square(self.y_bc_loss)) +\
                    tf.reduce_mean(tf.square(self.z_bc_loss)))



        # original optimizer

        # Optimizer (Using Adam here)
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        self.should_stop = 0
        self.best_loss = 1000000000  # arbitrary large number
        self.stopping_step = 0
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        self.best_weights = 0
        self.best_biases = 0
        self.best_lambda_1 = 0





    # initailize NN weights and biases
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # A way to give weights and biases initial values
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.dtypes.float32),
                           dtype=tf.float32)




    # Create NN structure
    def neural_net(self,X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            print(f"layer:{l}, {H}")
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # mine
    # triaining data & bc
    def net_u(self, x,y,z,t, collocation = False):
        X = tf.concat([x,y,z,t], 1)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(X)
            with tf.GradientTape(persistent=True) as g:  # persistence = True to watch multiple variables
                g.watch(X)
                u = self.neural_net(X, self.weights, self.biases)
            du = g.gradient(u, X)
            u_x = du[:, 0:1]
            u_y = du[:, 1:2]
            u_z = du[:, 2:3]
            u_t = du[:, 3:4]
        u_xx = gg.gradient(u_x, X)[:, 0:1]
        u_yy = gg.gradient(u_y, X)[:, 1:2]
        u_zz = gg.gradient(u_z, X)[:, 2:3]

        if not collocation:
            return u, u_x, u_y, u_z
        elif collocation:
            lambda_1 = self.lambda_1
            f_u = u_t - lambda_1 * (u_xx + u_yy + u_zz)
            return f_u



    def callback(self, loss, lambda_1):
        if (self.count < 100000):
            self.loss_array2[self.count] = loss
            # self.lambda_array2[self.count] = lambda_1
            self.count += 1

        print('Loss: %.3e, l1: %.3f' % (loss, lambda_1))
        return loss, lambda_1

    # use for non-defect

    def refine(self, x_rar, y_rar, t_rar):
        tf_dict = {self.x_f_tf: x_rar, self.y_f_tf: y_rar, self.t_f_tf: t_rar}
        f_v = self.sess.run(self.f_v_pred, tf_dict)

        argmax_idx = np.argmin(f_v)

        return argmax_idx



    # lambda also trained here
    def train(self, nIter, loss_array, lambda_array, fileName,early_stopping = 1000):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,self.t_tf: self.t, self.u_tf: self.u,
                   self.x_x_bc_tf: self.x_x_bc, self.x_y_bc_tf: self.x_y_bc, self.x_z_bc_tf: self.x_z_bc,self.x_t_bc_tf: self.x_t_bc,
                   self.y_x_bc_tf: self.y_x_bc, self.y_y_bc_tf: self.y_y_bc, self.y_z_bc_tf:self.y_z_bc, self.y_t_bc_tf:self.y_t_bc,
                   self.z_x_bc_tf:self.z_x_bc, self.z_y_bc_tf:self.z_y_bc, self.z_z_bc_tf:self.z_z_bc, self.z_t_bc_tf:self.z_t_bc,
                   self.x_f_tf: self.x_f, self.y_f_tf:self.y_f, self.z_f_tf: self.z_f, self.t_f_tf: self.t_f
                   # self.init_state: np.zeros([len(self.layers), self.x.shape[0], 20]),
                   # self.init_state_xbc: np.zeros([len(self.layers), self.x_xbc.shape[0], 20]),
                   # self.init_state_ybc: np.zeros([len(self.layers), self.x_ybc.shape[0], 20])
                   # , self.x_f_tf: self.x_f, self.y_f_tf:self.y_f, self.t_f_tf: self.t_f,
                   # self.x_ic_tf: self.x_ic, self.y_ic_tf: self.y_ic, self.t_ic_tf: self.t_ic, self.u_ic_tf: self.u_ic,
                   }
        # X_pool = X_pool
        # argmax_arr = argmax_arr
        print(f"x_f:{self.x_f_tf}")

        initial_time = time.time()
        start_time = time.time()




        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)


            lambda_1_value = self.sess.run(self.lambda_1)

            loss_array[it] = loss_value
            # lambda_array[it] = lambda_1_value
            if (self.should_stop == 1):
                break;

            if (loss_value < self.best_loss):
                self.stopping_step = 0
                self.best_loss = loss_value

                # save best only
                self.best_weights = copy.copy(self.weights)
                self.best_biases = copy.copy(self.biases)
                self.best_lambda_1 = copy.copy(self.lambda_1)
                best_iter = it

            else:
                self.stopping_step += 1
            if (self.stopping_step >= early_stopping):
                self.should_stop = 1
                print('Early stopping at step It: %d, loss: %.3e' % (it, loss_value))
                continue;
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.5e, l1: %.12f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, elapsed))
                start_time = time.time()

        final_time = time.time()
        training_time = final_time - initial_time
        print('Training took %.3f s' % training_time)
        print(f"Best_loss = {self.best_loss}; Iter {best_iter}")
        self.plot_training(loss_array[:it], fileName, self.best_loss)



    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:,0:1], self.y_tf: X_star[:, 1:2], self.z_tf: X_star[:,2:3], self.t_tf: X_star[:,3:4]}
        self.weights = self.best_weights
        self.biases = self.best_biases
        self.lambda_1 = self.best_lambda_1
        u_star = self.sess.run(self.u_pred, tf_dict)
        # print(u_star.shape())
        return u_star


    def save(self, fileDir):
        # saved file .pkl
        weights =self.sess.run(self.best_weights)
        biases = self.sess.run(self.best_biases)
        lambda_1 = self.sess.run(self.best_lambda_1)
        structure = self.layers


        with open(fileDir, "wb") as f:
            # pickle.dump([weightsh, biasesh, weightsr, biasesr,weightsz, biasesz ,\
            # weightso, biaseso,Uweightsh, Uweightsr ,Uweightsz, lambda1] , f)
            pickle.dump([structure, weights, biases, lambda_1], f)
            print("Save Successful !")


    def load_NN(self, filedir):
        weights_all = []
        biases_all =[]
        # num_layers1 = len(layers1)
        with open(filedir, 'rb') as f:
            # w1, b1, w2, b2, w3, b3, w4, b4, uw1,uw2, uw3, lambda1 = pickle.load(f)
            layers, weights, biases, lambda1 = pickle.load(f)


        for num in range(0, len(layers)-1):

            W = tf.Variable(weights[num], dtype=tf.float32)
            b = tf.Variable(biases[num], dtype= tf.float32)
            weights_all.append(W)
            biases_all.append(b)

        print("Parameters  import completed !")
        # return structure weights, biases, lambda
        return layers, weights_all, biases_all, lambda1

    def get_weights(self):
        return self.sess.run([self.best_weights, self.best_biases])
    def get_param(self):
        return self.sess.run(self.lambda_1)
    def summary(self):
        numweights = 0
        for weights in self.layers:
            numweights += weights
        print(f'NumLayers: {len(self.layers)}, total weights: {numweights}')


    def plot_training(self, loss_array,fileName, loss_value):
        epoch = np.linspace(0, len(loss_array), num = len(loss_array))
        plt.plot(epoch, loss_array)
        plt.title(f"training loss \n loss = {loss_value}")
        plt.xlabel(r'epoch')
        plt.ylabel(r'loss')
        plt.savefig(f"./" + fileName)
        plt.show()
