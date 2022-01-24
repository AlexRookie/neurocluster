import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, map_size=(10,10), epochs=50, batch=32, learn_rate=0.0001):
        self.map_size = np.asarray(map_size).astype(int)
        self.epochs = int(epochs)
        self.batch_size = int(batch)
        self.l_rate = learn_rate
        self.model = []

    def rmse(self, y_true, y_pred):
        # Root mean squared error (rmse) for regression
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def define_model(self, units):
        self.units = np.asarray(units).astype(int)

        inputs = tf.keras.layers.Input(shape=(2, self.units, ), name='input') #X_train.shape[-1]
        flatten = tf.keras.layers.Flatten()(inputs)
        som_layer = SOMLayer(map_size=self.map_size, name='SOM')(flatten)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=som_layer)

        return self.model

    def prepare_data(self, data, training_percentage=70):
        data = np.asarray(data)

        num_of_samples = data.shape[0]
        train = int(training_percentage*num_of_samples/100)
        valid = num_of_samples-train

        if train < self.batch_size or valid < self.batch_size:
            self.batch_size = 1
        else:
            # Samples must be multiplier of batch
            train = int(train/self.batch_size) * self.batch_size
            valid = num_of_samples-train
            valid = int(valid/self.batch_size) * self.batch_size

        x_train = data[0:train, :, :]
        #y_train = data[0:train, :, -10:]
        x_valid = data[train:train+valid, :, :]
        #y_valid = data[train:train+valid, :, -10:]

        self.x_train = np.array(x_train)
        #self.y_train = np.array(y_train)
        self.x_valid = np.array(x_valid)
        #self.y_valid = np.array(y_valid)

        return self.x_train, self.x_valid

    def kmeans_loss(self, y_pred, distances):
        """
        Calculate k-means reconstruction loss
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
            distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
        # Return
            k-means reconstruction loss
        """
        return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])

    def map_dist(self, y_pred, map_size):
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

    def neighborhood_function(self, d, T, neighborhood='gaussian'):
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

    def som_fit(self, x_train, y_train=None, x_val=None, y_val=None, decay='exponential'):
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
        
        x_train = np.asarray(x_train)

        Tmax = 10  # initial temperature parameter
        Tmin = 0.1 # final temperature parameter
    
        # Number of epochs where SOM neighborhood is decreased
        som_epochs = self.epochs
    
        # Evaluate metrics on training/validation batch every eval_interval epochs
        eval_interval = 10
    
        # Set and compute some initial values
        index = 0
        if x_val is not None:
            index_val = 0
        T = Tmax
    
        for ite in range(self.epochs):
            # Get training and validation batches
            x_batch = np.expand_dims(x_train[index], axis=0)
            if (index + 1) * self.batch_size >= x_train.shape[0]:
                #x_batch = x_train[index * self.batch_size::]
                #if y_train is not None:
                #    y_batch = y_train[index * batch_size::]
                index = 0
            else:
                #x_batch = x_train[index * self.batch_size:(index + 1) * self.batch_size]
                #if y_train is not None:
                #    y_batch = y_train[index * batch_size:(index + 1) * batch_size]
                index += 1
            #if x_val is not None:
            #    if (index_val + 1) * batch_size > x_val.shape[0]:
            #        x_val_batch = x_val[index_val * batch_size::]
            #        if y_val is not None:
            #            y_val_batch = y_val[index_val * batch_size::]
            #        index_val = 0
            #    else:
            #        x_val_batch = x_val[index_val * batch_size:(index_val + 1) * batch_size]
            #        if y_val is not None:
            #            y_val_batch = y_val[index_val * batch_size:(index_val + 1) * batch_size]
            #        index_val += 1
    
            # Compute cluster assignments for batches
            d = self.model.predict(x_batch)
            y_pred = d.argmin(axis=1)
            #if x_val is not None:
            #    d_val = self.model.predict([x_val_batch])
            #    y_val_pred = d_val.argmin(axis=1)
    
            # Update temperature parameter
            if ite < som_epochs:
                if decay == 'exponential':
                    T = Tmax*(Tmin/Tmax)**(ite/(som_epochs-1))
                elif decay == 'linear':
                    T = Tmax - (Tmax-Tmin)*(ite/(som_epochs-1))
    
            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred, self.map_size), T, neighborhood='gaussian')
            #if x_val is not None:
            #    w_val_batch = neighborhood_function(map_dist(y_val_pred, map_size), T, neighborhood='gaussian')
    
            # Train on batch
            loss = self.model.train_on_batch(x_batch, w_batch)
    
            if ite % eval_interval == 0:
                # Evaluate losses and metrics
                Lsom = loss
                Lkm  = self.kmeans_loss(y_pred, d)
                Ltop = loss - self.kmeans_loss(y_pred, d)
                #quantization_err = quantization_error(d)
                #topographic_err  = topographic_error(d, map_size)
    
                print('iteration {} - T={}'.format(ite, T))
                print('[Train] - Lsom={:f} (Lkm={:f}/Ltop={:f})'.format(Lsom, Lkm, Ltop))
                #print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(quantization_err, topographic_err))
    
        return self.model

    def som_loss(self, weights, distances):
        """
        Calculate SOM reconstruction loss
        # Arguments
            weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
            distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
        # Return
            SOM reconstruction loss
        """
        return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))

    def train_model(self, inputs):
        inputs = np.asarray(inputs)
        # Compile model 
        self.model.compile(optimizer='adam', loss='mse')
        # Train model
        self.model = self.som_fit(inputs)

        return self.model

    def predict(self, model, inputs):
        inputs = np.asarray(inputs)
        pred = model(inputs) # == model.predict([inputs])
        pred = np.argmin(pred)
        return pred

    def plot(self):
        # Plot
        som_weights = self.model.get_layer(name='SOM').get_weights()[0]
        
        fig1, axes = plt.subplots(nrows=self.map_size[0], ncols=self.map_size[1], figsize=(10, 10))
        for k in range(self.map_size[0] * self.map_size[1]):
           axes[k // self.map_size[1]][k % self.map_size[1]].imshow(som_weights[k].reshape(2, self.units), cmap='gray')
           axes[k // self.map_size[1]][k % self.map_size[1]].axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        
        plt.draw() # non-blocking plot
        plt.pause(0.1)

#=======================================================================================#

class SOMLayer(tf.keras.layers.Layer):
    """
    Self-Organizing Map layer class with rectangular topology
    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` witch represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shape
        2D tensor with shape: `(n_samples, n_prototypes)`
    """

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        input_dim = input_shape[1]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.prototypes = self.add_weight(shape=(self.n_prototypes, input_dim), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Calculate pairwise squared euclidean distances between inputs and prototype vectors
        Arguments:
            inputs: the variable containing data, Tensor with shape `(n_samples, latent_dim)`
        Return:
            d: distances between inputs and prototypes, Tensor with shape `(n_samples, n_prototypes)`
        """
        # Note: (tf.expand_dims(inputs, axis=1) - self.prototypes) has shape (n_samples, n_prototypes, latent_dim)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.n_prototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

