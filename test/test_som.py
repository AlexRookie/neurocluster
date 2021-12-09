# SOM network test
# Ref: https://github.com/FlorentF9/DESOM/
# Ref2: https://towardsdatascience.com/self-organizing-map-layer-in-tensroflow-with-interactive-code-manual-back-prop-with-tf-580e0b60a1cc

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from som import SOMLayer
from metrics import *

from tensorflow.keras.datasets import mnist

#=======================================================================================#

# Parameters
epochs = 8000
batch_size = 256
map_size = [10,10]

#=======================================================================================#

# Load data
(x_train, _), (x_val, _) = mnist.load_data()

# Normalize values between 0 and 1
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
# Flatten to 784-dimensional vector
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

#=======================================================================================#

# Define model

inputs = tf.keras.layers.Input(shape=(x_train.shape[-1], ), name='input') #X_train.shape[-1]
som_layer = SOMLayer(map_size=map_size, name='SOM')(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=som_layer)

print(model.summary())

def som_loss(weights, distances):
    """
    Calculate SOM reconstruction loss
    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))

model.compile(optimizer='adam', loss=som_loss)

#=======================================================================================#

def kmeans_loss(y_pred, distances):
    """
    Calculate k-means reconstruction loss
    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])

def map_dist(y_pred, map_size=map_size):
    """
    Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
    
    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
    # Return
        pairwise distance matrix (map_dist[i,k] is the distance on the map between assigned cell of data point i and cell k)
    """
    n_prototypes = map_size[0]*map_size[1]
    labels = np.arange(n_prototypes)
    tmp = np.expand_dims(y_pred, axis=1)
    d_row = np.abs(tmp-labels)//map_size[1]
    d_col = np.abs(tmp%map_size[1]-labels%map_size[1])
    return d_row + d_col

def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function (gaussian neighborhood)
        # Arguments
            d: distance on the map
            T: temperature parameter
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d**2)/(T**2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)

def som_fit(x_train, y_train=None, x_val=None, y_val=None, epochs=100, batch_size=256, decay='exponential'):
    """
    Training procedure
    # Arguments
       X_train: training set
       y_train: (optional) training labels
       X_val: (optional) validation set
       y_val: (optional) validation labels
       epochs: number of training epochs
       batch_size: training batch size
       decay: type of temperature decay ('exponential' or 'linear')
    """

    Tmax = 10  # initial temperature parameter
    Tmin = 0.1 # final temperature parameter

    # Number of epochs where SOM neighborhood is decreased
    som_epochs = epochs

    # Evaluate metrics on training/validation batch every eval_interval epochs
    eval_interval = 10

    # Set and compute some initial values
    index = 0
    if x_val is not None:
        index_val = 0
    T = Tmax

    for ite in range(epochs):
        # Get training and validation batches
        if (index + 1) * batch_size > x_train.shape[0]:
            x_batch = x_train[index * batch_size::]
            if y_train is not None:
                y_batch = y_train[index * batch_size::]
            index = 0
        else:
            x_batch = x_train[index * batch_size:(index + 1) * batch_size]
            if y_train is not None:
                y_batch = y_train[index * batch_size:(index + 1) * batch_size]
            index += 1
        if x_val is not None:
            if (index_val + 1) * batch_size > x_val.shape[0]:
                x_val_batch = x_val[index_val * batch_size::]
                if y_val is not None:
                    y_val_batch = y_val[index_val * batch_size::]
                index_val = 0
            else:
                x_val_batch = x_val[index_val * batch_size:(index_val + 1) * batch_size]
                if y_val is not None:
                    y_val_batch = y_val[index_val * batch_size:(index_val + 1) * batch_size]
                index_val += 1

        # Compute cluster assignments for batches
        d = model.predict([x_batch])
        y_pred = d.argmin(axis=1)
        if x_val is not None:
            d_val = model.predict([x_val_batch])
            y_val_pred = d_val.argmin(axis=1)

        # Update temperature parameter
        if ite < som_epochs:
            if decay == 'exponential':
                T = Tmax*(Tmin/Tmax)**(ite/(som_epochs-1))
            elif decay == 'linear':
                T = Tmax - (Tmax-Tmin)*(ite/(som_epochs-1))

        # Compute topographic weights batches
        w_batch = neighborhood_function(map_dist(y_pred, map_size), T, neighborhood='gaussian')
        if x_val is not None:
            w_val_batch = neighborhood_function(map_dist(y_val_pred, map_size), T, neighborhood='gaussian')

        # Train on batch
        loss = model.train_on_batch(x_batch, w_batch)

        if ite % eval_interval == 0:
            # Evaluate losses and metrics
            Lsom = loss
            Lkm  = kmeans_loss(y_pred, d)
            Ltop = loss - kmeans_loss(y_pred, d)
            quantization_err = quantization_error(d)
            topographic_err  = topographic_error(d, map_size)

            print('iteration {} - T={}'.format(ite, T))
            print('[Train] - Lsom={:f} (Lkm={:f}/Ltop={:f})'.format(Lsom, Lkm, Ltop))
            print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(quantization_err, topographic_err))

    return model

# Train model
model = som_fit(x_train, epochs=epochs, batch_size=batch_size)

#=======================================================================================#

# Plot
som_weights = model.get_layer(name='SOM').get_weights()[0]
img_size = int(np.sqrt(x_train.shape[1]))
fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))
for k in range(map_size[0] * map_size[1]):
   ax[k // map_size[1]][k % map_size[1]].imshow(som_weights[k].reshape(img_size, img_size), cmap='gray')
   ax[k // map_size[1]][k % map_size[1]].axis('off')
plt.subplots_adjust(hspace=0.05, wspace=0.05)

plt.draw() # non-blocking plot
plt.pause(0.1)

#  # Save the final model
# logfile.close()
# print('saving model to:', save_dir + '/kerasom_model_final.h5')
# self.model.save_weights(save_dir + '/kerasom_model_final.h5')

# Predict
d = model.predict(np.expand_dims(x_val[0], axis=0), verbose=0)
d = d.argmin(axis=1)

plt.figure(10)
plt.imshow(255.*x_val[0].reshape(img_size, img_size))

plt.draw() # non-blocking plot
plt.pause(0.1)