# Contents
dbn_train.py is for training the dbn.
dbn.py is the models we built.
dbn_generate.py is for generating the images we got.


# Generating Data with DBN
The goal of this project was to construct and train a Deep Bayesian Network on an insufficient/imbalanced dataset, generate new samples to augment the dataset, and observe how the augmented dataset improves the performance of classification models.

# Data

LFW Face Database
The data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. These are some examples.
<img width="304" alt="Screen Shot 2022-02-10 at 5 39 28 PM" src="https://user-images.githubusercontent.com/98995087/153525195-bb65954b-1b76-42cd-b0b5-20ad6b7f7bb2.png">


# Restricted Boltzmann Machines (RBM)
Boltzmann machines (BMs) are bidirectionally connected networks of stochastic processing units, which can be interpreted as neural network models. A BM can be used to learn important aspects of an unknown probability distribution based on samples from this distribution.
This learning process is generally difficult and time-consuming, it can be simplified by imposing restrictions on the network topology - Restricted Boltzmann Machines.

A Restricted Boltzmann Machine (RBM) is a parameterized generative model representing a probability distribution. 

Given some training data, the model learns parameters such that the probability distribution represented by the model fits the training data as well as possible.
It consists of m visible units V = (V1, ..., Vm) to represent observable data and n hidden units H = (H1, ..., Hn) to capture dependencies between observed variables.

<img width="412" alt="Screen Shot 2022-02-10 at 5 35 35 PM" src="https://user-images.githubusercontent.com/98995087/153524877-3dce95c4-168b-43bc-be38-5a3216ede3e5.png">

# Deep Bayesian Network
A DBN consists of a stacked Restricted Boltzmann Machines. Each RBM’s hidden layer feeds the next RBM’s visible layer.
Each RBM is trained one at a time, after one has completed training it initializes the next RBM with its learned weights and biases
The training process consists of two techniques known as Gibbs Sampling and Persistent Contrastive Divergence.
Gibbs Sampling is the process of traversing the Markov chain that the RBM consists of. 

# Result

<img width="1083" alt="Screen Shot 2022-02-10 at 10 20 26 PM" src="https://user-images.githubusercontent.com/98995087/153545931-9351ca98-cae7-4da1-a34e-c34eabe23e45.png">


