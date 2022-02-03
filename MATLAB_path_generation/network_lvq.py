import sys
sys.path.append("./libraries/Modified-SOM")

import numpy as np
import matplotlib.pyplot as plt
from detection.competitive_learning import SOM, CombineSomLvq

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

class Network:
    def __init__(self):
        self.model = []
        self.batch_size = 32
        # Label encoder
        self.encoder = LabelEncoder()

    def define_model(self, map_size=(10,10)):
        map_size = np.asarray(map_size).astype(int)
        # Setting the random state
        random_state = 17
        self.model = SOM(n_rows=map_size[0], n_cols=map_size[1], random_state=random_state) # CombineSomLvq(n_rows=map_size[0], n_cols=map_size[1], random_state=random_state)
        return self.model

    def prepare_data(self, X, y, training_percentage=70, batch=32, randomize=True):
        X = np.asarray(X)
        y = np.asarray(y)
        self.batch_size = int(batch)

        if randomize:
            perm = np.random.permutation(X.shape[0])
            if len(X.shape)==2:
                X = X[perm,:]
            if len(X.shape)==3:
                X = X[perm,:,:]
            y = y[perm]

        num_of_samples = X.shape[0]
        train = int(training_percentage*num_of_samples/100)
        valid = num_of_samples-train

        if train < self.batch_size or valid < self.batch_size:
            self.batch_size = 1
        else:
            # Samples must be multiplier of batch
            train = int(train/self.batch_size) * self.batch_size
            valid = num_of_samples-train
            valid = int(valid/self.batch_size) * self.batch_size

        if len(X.shape)==2:
            X_train = X[0:train,:]
            X_valid = X[train:train+valid,:]
        if len(X.shape)==3:
            X_train = X[0:train,:,:]
            X_valid = X[train:train+valid,:,:]
        y_train = y[0:train]
        y_valid = y[train:train+valid]

        y_train = self.encoder.fit_transform(y_train)

        self.X_train = np.ascontiguousarray(X_train)
        self.y_train = np.ascontiguousarray(y_train)
        self.X_valid = np.ascontiguousarray(X_valid)
        self.y_valid = np.ascontiguousarray(y_valid)
        return self.X_train, self.y_train, self.X_valid, self.y_valid

    def train_model(self, X_train, y_train, epochs_unsup=50, epochs_sup=50, learn_rate=0.01):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).astype(np.int8)
        epochs_unsup = int(epochs_unsup)
        epochs_sup = int(epochs_sup)

        # Train the SOM
        self.model.fit(X_train, None, weights_init= "pca",
            num_iters = epochs_unsup, batch_size = self.batch_size, 
            neighborhood = "gaussian",
            learning_rate = 0.75, learning_decay_rate = 1, learning_rate_decay_function = None,
            sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None, num_clusters = 3, 
            conscience = False, verbose = 1)

        # Train the LVQ
        #self.model.fit(X_train, y_train, weights_init = "pca", labels_init = None,
        #    unsup_num_iters = epochs_unsup, unsup_batch_size = self.batch_size,
        #    sup_num_iters = epochs_sup, sup_batch_size = self.batch_size,
        #    neighborhood = "gaussian",
        #    learning_rate = learn_rate, learning_decay_rate = 1, learning_rate_decay_function = None,
        #    sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
        #    conscience = False, verbose = 1)
        return self.model

    def predict(self, X_valid, y_valid=None):
        X_valid = np.asarray(X_valid)

        # Predict the result
        y_pred = self.model.predict(X_valid)
        #y_pred = self.encoder.inverse_transform(y_pred)

        #if y_valid is not None:
        #    y_valid = np.asarray(y_valid)
        #    # Make confusion matrix
        #    cm = confusion_matrix(y_valid, y_pred)
        #    # Print the confusion matrix
        #    print(cm)
        #    print('Accuracy:', accuracy_score(y_valid, y_pred))

        y_pred = np.asarray(y_pred)
        return y_pred

    def predict_som(self, X_valid, y_valid=None):
        X_valid = np.asarray(X_valid)
        # Predict the result
        y_pred = self.model.predict_som(X_valid)
        y_pred = np.asarray(y_pred)
        return y_pred

    def get_data(self):
        weights, labels, bias, clus_labels = [], [], [], []
        weights = self.model._competitive_layer_weights
        #labels = self.model._nodes_label
        bias = self.model._bias
        clus_labels = self.model.cluster_label

        weights = np.asarray(weights)
        labels = np.asarray(labels)
        bias = np.asarray(bias)
        clus_labels = np.asarray(clus_labels)
        return weights, labels, bias, clus_labels

    def load_data(self, weights=None, labels=None, bias=None, clus_labels=None):
        weights = np.asarray(weights)
        labels = np.asarray(labels)
        bias = np.asarray(bias)
        clus_labels = np.asarray(clus_labels)

        self.model._competitive_layer_weights = weights
        self.model._nodes_label = labels
        self.model._bias = bias
        self.model.clus_labels = clus_labels
        return

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
