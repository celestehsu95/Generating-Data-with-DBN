#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
from DBN import DBN
from random import sample
import numpy as np
# from aux.updateClass import readClass
import pickle

def plotSamples(namePickle,nameFile,dim,mf,indices=None):
    if mf == 1:
        mf = True
    else:
        mf = False
    with open('/content/pickles/2020_11_13_07_50_30_3072_500_3072_cifar_dbn/dbn.pickle','rb') as picklefile:
      dbn = pickle.load(picklefile)
    # dbn = readClass(namePickle)
    
    cifar = tf.keras.datasets.cifar10
    (x_train, _), (_,_) = cifar.load_data()
    x_train = x_train/255.0
    x_train = [tf.cast(tf.reshape(x,shape=(3072,1)),"float32") for x in x_train]
    # if indices is None:
    #     indices = sample(range(len(x_train)), number_samples)
    # x_train = [x_train[ind] for ind in indices]
    number_samples=100
    samples = dbn.generate_visible_samples(mean_field = mf)
    indices = sample(range(len(x_train)), number_samples)
    x_train = [x_train[i] for i in indices]
    plot_samples_vs_orig(samples,x_train)
    plotSamples_DBN(samples, nameFile, dim)

def plot_samples_vs_orig(samples, orig):
  #msre = tf.reduce_mean(tf.square(orig - samples))
  #self.train_msre[batch_idx] = msre
    orig = tf.squeeze(tf.stack(orig))
    samples = tf.squeeze(tf.stack(samples))
    msre = tf.reduce_mean(tf.square(orig - samples), axis=1)
    plt.scatter(range(len(orig)), msre, marker='o')
    plt.xlabel('sample')
    plt.ylabel('msre')
    plt.title('cifar_DBN_Original vs Generated Image')
    plt.savefig('cifar-dbn-msre1.png')



def plotSamples_DBN(obj, name, dim = 28, nrows = 10, ncols = 10):
    import matplotlib.pyplot as plt
    
    obj=[tf.reshape(x,shape=(32,32,3)) for x in obj]
    # obj=(obj+1)/2.0
    obj=np.asarray(obj)
    print(obj.shape)
    img = 64
    for i in range(100):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(obj[i,:,:,:])
        # plt.show()
    plt.savefig(name+".png", dpi=400)
# def plotSamples_DBN(obj, name, dim = 28, nrows = 10, ncols = 10):
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
#     i = 0
#     for row in ax:
#         for col in row:
#             col.imshow(tf.reshape(obj[i],(dim,dim)),cmap='Greys_r')
#             col.axis('off')
#             i += 1
#     fig.savefig(os.path.abspath(os.path.dirname(os.getcwd()))+"/img/"+name+".png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test",
                        help="file name of output png <default: 'test'>")
    parser.add_argument("--dim", type=int, default=28,
                        help="square dimensions on which to remap images <default: 28>")
    parser.add_argument("--mean-field", type=int, default=1,
                        help="draw actual samples (0) or mean-field samples (1) <default: 1>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str, 
                               help="name of directory where dbn.pickle is stored",
                               required=True)
    args = parser.parse_args()
    plotSamples(args.pickle,args.out,args.dim,args.mean_field)
