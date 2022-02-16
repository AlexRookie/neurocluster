import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as K
import numpy as np
#import matplotlib.pyplot as plt
from time import sleep

class Network:
    def __init__(self):
        self.model = []
        self.map_size = None
        self.units    = None
        self.classes  = None
        self.batch_size = None
        self.epochs     = None
        self.l_rate     = None

    def define_model(self, map_size=(10,10), units=(1,10), classes=0):
        self.map_size = np.asarray(map_size).astype(int)
        self.units = np.asarray(units).astype(int)
        self.classes = np.asarray(classes).astype(int)

        inputs = tf.keras.layers.Input(shape=(self.units[0], self.units[1]), name='input') #X_train.shape[-1]
        flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
        som_layer = SOMLayer(map_size=self.map_size, name='SOM')(flatten)
        #self.model = tf.keras.models.Model(inputs=inputs, outputs=som_layer)
        outputs = tf.keras.layers.Dense(units=self.classes, activation='softmax', name='classifier')(som_layer)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=[som_layer, outputs])

        return self.model

    def prepare_data(self, X, y=None, training_percentage=70, batch=32, randomize=True):
        X = np.asarray(X)
        y = np.asarray(y)
        self.batch_size = int(batch)

        # One hot encode
        y = tf.keras.utils.to_categorical(y)

        if randomize:
            m = X.shape[1]
            n = X.shape[2]
            X = X.reshape(X.shape[0], m*n)
            perm = np.random.permutation(X.shape[0])
            X = X[perm, :]
            y = y[perm, :]
            X = X.reshape(X.shape[0], m, n)

        # Normalize
        #x = np.copy(X)
        #for i in range(x.shape[0]):
        #    x[i, 0, :] =  (x[i, 0, :] - np.min(x[i, 0, :])) / (np.max(x[i, 0, :]) - np.min(x[i, 0, :]))
        #    x[i, 1, :] =  (x[i, 1, :] - np.min(x[i, 1, :])) / (np.max(x[i, 1, :]) - np.min(x[i, 1, :]))

        # Shift
        x = np.copy(X)
        for i in range(x.shape[0]):
            x[i, 0, :] = x[i, 0, :] - x[i, 0, 0]
            x[i, 1, :] = x[i, 1, :] - x[i, 1, 0]

        num_of_samples = x.shape[0]
        train = int(training_percentage*num_of_samples/100)
        valid = num_of_samples-train

        if train < self.batch_size or valid < self.batch_size:
            self.batch_size = 1
        else:
            # Samples must be multiplier of batch
            train = int(train/self.batch_size) * self.batch_size
            valid = num_of_samples-train
            valid = int(valid/self.batch_size) * self.batch_size

        x_train = x[0:train, :, :]
        x_valid = x[train:train+valid, :, :]
        y_train = y[0:train, :]
        y_valid = y[train:train+valid, :]

        self.x_train = np.array(x_train)
        self.x_valid = np.array(x_valid)
        self.y_train = np.array(y_train)
        self.y_valid = np.array(y_valid)

        return self.x_train, self.x_valid, self.y_train, self.y_valid

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

    def som_fit(self, x_train, y_train=None, decay='exponential'):
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
        y_train = np.asarray(y_train)

        Tmax = 10  # initial temperature parameter
        Tmin = 0.1 # final temperature parameter
        som_epochs = self.epochs # Number of epochs where SOM neighborhood is decreased

        eval_interval = 10 # Evaluate metrics on training/validation batch every eval_interval epochs
    
        # Set and compute some initial values
        index = 0
        T = Tmax
    
        for it in range(self.epochs):
            # Get training and validation batches
            #x_batch = np.expand_dims(x_train[index], axis=0)
            #y_batch = np.expand_dims(y_train[index], axis=0)
            if (index + 1) * self.batch_size >= x_train.shape[0]:
                x_batch = x_train[index * self.batch_size::]
                if y_train is not None:
                    y_batch = y_train[index * self.batch_size::]
                index = 0
            else:
                x_batch = x_train[index * self.batch_size:(index + 1) * self.batch_size]
                if y_train is not None:
                    y_batch = y_train[index * self.batch_size:(index + 1) * self.batch_size]
                index += 1
    
            # Compute cluster assignments for batches
            d, _ = self.model.predict(x_batch)
            y_pred = d.argmin(axis=1)
    
            # Update temperature parameter
            if it < som_epochs:
                if decay == 'exponential':
                    T = Tmax*(Tmin/Tmax)**(it/(som_epochs-1))
                elif decay == 'linear':
                    T = Tmax - (Tmax-Tmin)*(it/(som_epochs-1))
    
            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred, self.map_size), T, neighborhood='gaussian')

            # Train on batch
            loss = self.model.train_on_batch(x_batch, [w_batch, y_batch]) # loss: ['loss', 'SOM_loss', 'classifier_loss', 'SOM_accuracy', 'classifier_accuracy']

            if it % eval_interval == 0:
                # Evaluate losses and metrics
                Lsom = loss[1]
                Lkm  = self.kmeans_loss(y_pred, d)
                Ltop = loss[1] - self.kmeans_loss(y_pred, d)
                #quantization_err = quantization_error(d)
                #topographic_err  = topographic_error(d, map_size)
    
                print('iteration {} - T={}'.format(it, T))
                print('[Train] - Lsom={:f} (Lkm={:f}/Ltop={:f})'.format(Lsom, Lkm, Ltop))
                #print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(quantization_err, topographic_err))
                sleep(0.2)
    
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

    def rmse(self, y_true, y_pred):
        # Root mean squared error (rmse) for regression
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

    def train_model(self, x_train, y_train=None, epochs=50, learn_rate=0.01):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).astype(np.int8)
        self.epochs = int(epochs)
        self.l_rate = learn_rate

        # Compile model
        #self.model.compile(optimizer='adam', loss=self.som_loss)    #, metrics=[self.rmse])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.l_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy']) #keras.optimizers.Adam(1e-3)
        
        # Train model
        self.model = self.som_fit(x_train, y_train)
        return self.model

    def predict(self, inputs):
        inputs = np.asarray(inputs)
        if len(inputs.shape)==1:
            inputs = np.expand_dims(inputs, axis=0)
        # Predict
        som_pred, pred = self.model(inputs) # == model.predict([inputs])
        #pred = np.argmin(pred)
        #y_classes = pred.argmax(axis=-1)
        return som_pred, pred

    def save(self, file):
        self.model.save_weights(file+'_weights.h5')
        self.model.save(file+'.h5')

    def load_model(self, file):
        self.model.load(file+'.h5')
        return self.model

    def load_weights(self, file):
        self.model.load_weights(file+'_weights.h5')
        return self.model

    #def plot(self):
    #    # Plot
    #    som_weights = self.model.get_layer(name='SOM').get_weights()[0]
    #    
    #    fig1, axes = plt.subplots(nrows=self.map_size[0], ncols=self.map_size[1], figsize=(10, 10))
    #    for k in range(self.map_size[0] * self.map_size[1]):
    #       axes[k // self.map_size[1]][k % self.map_size[1]].imshow(som_weights[k].reshape(2, self.units), cmap='gray')
    #       axes[k // self.map_size[1]][k % self.map_size[1]].axis('off')
    #    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #    
    #    plt.draw() # non-blocking plot
    #    plt.pause(0.1)

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

