"""
Created on Friday, March 16, 2018

@author: rupreetg

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
mpl.use("Agg")
import sys
from os import path

libpath = "../tflab/tflab"

if(libpath not in sys.path):
    sys.path.append(libpath)

from network import FeedForwardLogistic
from optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer

def Prepare_data():
    import sklearn
    from sklearn.datasets import fetch_mldata
    custom_data_home = "./data"
    mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
    train_x = mnist.data
    train_x = train_x/255  # normalize the data so that the value is between 0 & 1
    train_y = mnist.target

    n_samples = mnist.data.shape[0]
    n_features = mnist.data.shape[1]
    n_classes = len(np.unique(mnist.target))

    train_y = train_y.astype(np.int16)
    train_y = np.eye(n_classes)[train_y]
    return train_x, train_y, n_samples, n_features, n_classes

#Setting the parameters...
learning_rate = 0.001
steps = 10000
train_x, train_y, n_samples, n_features, n_classes = Prepare_data()

rng = np.random
rng.seed(1234)

opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate,scale=1.001),
    tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    ASRMSPropOptimizer(base_learning_rate=learning_rate,scale=1.001),
    tf.train.AdamOptimizer(learning_rate=learning_rate),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
    tf.train.AdagradOptimizer(learning_rate=learning_rate)
]
opt_names = [
            'SGD', 
            'SGD+AS', 
            'RMSProp', 
            'RMSProp+AS', 
            'ADAM', 
            'SGD+M', 
            'SGD+NM', 
            'Adagrad'
            ]

losses = []
with tf.Session() as sess:
    for i, opt in enumerate(opts):
        print(opt_names[i])
        reg  = FeedForwardLogistic([784, 10], nonlinearities = lambda x: tf.exp(x)/ tf.reduce_sum(tf.exp(x)))
        loss = reg.train(sess, train_x, train_y, minibatch_size = 20, steps= steps, optimizer = opts[i])
        losses.append(loss)

plt.clf()
for loss, opt_name in zip(losses, opt_names):
    plt.plot(loss[::5], '+-', alpha=.5, label=opt_name)
plt.legend()
plt.savefig("./plots/lg_comparison.png")