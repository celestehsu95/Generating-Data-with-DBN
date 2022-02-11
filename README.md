# Generating Data withRBM
The goal of this project was to construct and train a Restricted Boltzmann Machine on an insufficient/imbalanced dataset, generate new samples to augment the dataset, and observe how the augmented dataset improves the performance of classification models.

# Data
Mnist
The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
The digits are size-normalized and centered in a fixed-size image.
 <img width="1233" alt="Screen Shot 2022-02-10 at 5 07 36 PM" src="https://user-images.githubusercontent.com/98995087/153525070-73d91ef9-80bd-4313-9a9b-efe3f3d08be0.png">

CIFAR10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.
<img width="609" alt="Screen Shot 2022-02-10 at 5 38 45 PM" src="https://user-images.githubusercontent.com/98995087/153525147-d49ab029-6344-4a4e-95ab-09d4488b4a22.png">

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


# Result
RBM with MNIST

<img width="961" alt="Screen Shot 2022-02-10 at 5 36 55 PM" src="https://user-images.githubusercontent.com/98995087/153524996-131aa4a9-99eb-40b0-9e28-e0ef44f97d1d.png">

RBM with CIFAR10
<img width="1122" alt="Screen Shot 2022-02-10 at 5 40 26 PM" src="https://user-images.githubusercontent.com/98995087/153525283-75ac85f0-7009-4e0f-9ab1-279d36ca5b0a.png">


RBM with LFW Face Database
<img width="1110" alt="Screen Shot 2022-02-10 at 5 41 22 PM" src="https://user-images.githubusercontent.com/98995087/153525363-c0cb0dd5-fdab-46c7-8e65-ccb291c0c724.png">


