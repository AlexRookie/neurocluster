import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.encoder     = []
        self.decoder     = []
        self.decoder_theta = []
        self.decoder_omega = []
        self.autoencoder = []
        self.classifier  = []
        self.units          = None
        self.latent_neurons = None
        self.classes        = None
        self.batch_size = None
        self.epochs_unsup = None
        self.epochs_sup   = None
        self.l_rate_unsup = None
        self.l_rate_sup   = None

    def define_model(self, units=(1,10), latent_neurons=8, classes=0):
        self.units = np.asarray(units).astype(int)
        self.latent_neurons =  np.asarray(latent_neurons).astype(int)
        self.classes = np.asarray(classes).astype(int)

        encoder_input = tf.keras.layers.Input(shape=(self.units[0], self.units[1]), batch_size=None, name='EncoderInput')
        x = tf.keras.layers.Permute((2,1), input_shape=(self.units[0], self.units[1]), name='Permute-1')(encoder_input)
        x = tf.keras.layers.Conv1D(filters=6, kernel_size=5, padding='valid', name='Conv1D-1')(x)
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding='valid', name='Conv1D-2')(x)
        x = tf.keras.layers.Conv1D(filters=10, kernel_size=3, padding='valid', name='Conv1D-3')(x)
        x = tf.keras.layers.Flatten(name='Flatten')(x)
        x = tf.keras.layers.Dense(units=20, name='Dense-1')(x)
        x = tf.keras.layers.Dense(units=10, name='Dense-2')(x)
        encoded = tf.keras.layers.Dense(units=self.latent_neurons, name='Dense-3')(x)

        decoder_input = tf.keras.layers.Input(shape=(encoded.shape[1], ), batch_size=None, name='DecoderInput')
        x = tf.keras.layers.Dense(units=10, name='Dense-4')(decoder_input)
        x = tf.keras.layers.Dense(units=20, name='Dense-5')(x)
        x = tf.keras.layers.Dense(units=40, name='Dense-6')(x)
        x = tf.keras.layers.Reshape((4,10), name='Reshape-1')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, padding='valid', name='Deconv1D-1')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=6, kernel_size=3, padding='valid', name='Deconv1D-2')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=5, padding='valid', name='Deconv1D-3')(x)
        decoded = tf.keras.layers.Permute((2,1), input_shape=(self.units[1], self.units[0]), name='Permute-2')(x)

        #decoder_theta_input = tf.keras.layers.Input(shape=(self.latent_neurons, ), batch_size=None, name='DecoderThetaInput')
        #x = tf.keras.layers.Dense(units=40, name='Dense-3')(decoder_theta_input)
        #x = tf.keras.layers.Reshape((4,10), name='Reshape-2')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, padding='valid', name='Deconv1D-4')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=6, kernel_size=3, padding='valid', name='Deconv1D-5')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=2, kernel_size=5, padding='valid', name='Deconv1D-6')(x)
        #decoded_theta = tf.keras.layers.Permute((2,1), input_shape=(self.units[1], self.units[0]), name='Permute-3')(x)
        #
        #decoder_omega_input = tf.keras.layers.Input(shape=(self.latent_neurons, ), batch_size=None, name='DecoderOmegaInput')
        #x = tf.keras.layers.Dense(units=40, name='Dense-4')(decoder_omega_input)
        #x = tf.keras.layers.Reshape((4,10), name='Reshape-3')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, padding='valid', name='Deconv1D-7')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=6, kernel_size=3, padding='valid', name='Deconv1D-8')(x)
        #x = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, padding='valid', name='Deconv1D-9')(x)
        #decoded_omega = tf.keras.layers.Permute((2,1), input_shape=(self.units[1], 1), name='Permute-4')(x)

        #x = tf.keras.layers.Dense(units=16, name='Dense-7')(encoded)
        classified = tf.keras.layers.Dense(units=self.classes, activation='softmax', name='Classifier')(encoded)

        # Encoder model
        self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoded)
        # Decoder model
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoded)
        #self.decoder_theta = tf.keras.models.Model(inputs=decoder_theta_input, outputs=decoded_theta)
        #self.decoder_omega = tf.keras.models.Model(inputs=decoder_omega_input, outputs=decoded_omega)
        # Autoencoder model
        self.autoencoder = tf.keras.models.Model(inputs=encoder_input, outputs=self.decoder(encoded));
        #                                        outputs=[self.decoder(encoded), self.decoder_theta(encoded[:,0:self.latent_neurons]), self.decoder_omega(encoded[:,-(self.latent_neurons+1):-1])])
        # Classifier model
        self.classifier = tf.keras.models.Model(inputs=encoder_input, outputs=classified)

        return self.encoder, self.decoder, self.autoencoder, self.classifier

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

        # Shift x & y coordinates
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

    def rmse(self, y_true, y_pred):
        # Root mean squared error (rmse) for regression
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

    def train_model_unsup(self, x_train, y_train=None, epochs=50, learn_rate=0.0001): 
        x_train = np.asarray(x_train)
        self.epochs_unsup = int(epochs)
        self.l_rate_unsup = learn_rate

        # Unsupervised training
        self.autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate_unsup),
                                 loss = 'mean_squared_error',
                                 metrics = [self.rmse])

        fit_unsup = self.autoencoder.fit(x_train, x_train,
                                         epochs = self.epochs_unsup,
                                         batch_size = self.batch_size,
                                         shuffle = True,
                                         verbose = 1)

        return self.encoder, self.decoder, self.autoencoder, self.classifier, fit_unsup

    def train_model_sup(self, x_train, y_train, epochs=50, learn_rate=0.0001): 
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).astype(np.int8)
        self.epochs_sup = int(epochs)
        self.l_rate_sup = learn_rate

        # Copy weights
        for i in range(len(self.encoder.layers)):
            self.classifier.layers[i].set_weights(self.encoder.layers[i].get_weights())
        # Set non-trainable layers
        for i in range(len(self.encoder.layers)): #for layer in self.classifier.layers[0:7]:
            self.classifier.layers[i].trainable = False

        # Supervised training
        self.classifier.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate_sup),
                                loss = 'categorical_crossentropy',
                                metrics = ['accuracy'])
        
        fit_sup = self.classifier.fit(x_train, y_train,
                                       epochs = self.epochs_sup,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       verbose = 1)

        ## Set trainable layers
        #for i in range(len(self.encoder.layers)):
        #    self.classifier.layers[i].trainable = True
        #
        ## Over training
        #self.classifier.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate_sup),
        #                        loss = 'categorical_crossentropy',
        #                        metrics = ['accuracy'])
        #
        #fit_over = self.classifier.fit(x_train, y_train,
        #                               epochs = self.epochs_sup,
        #                               batch_size = self.batch_size,
        #                               shuffle = True,
        #                               verbose = 1)

        return self.encoder, self.decoder, self.autoencoder, self.classifier, fit_sup#, fit_over

    def predict(self, model, inputs):
        inputs = np.asarray(inputs)
        # Predict
        pred = model(inputs) # == model.predict([inputs])
        return pred

    def save(self, file):
        self.encoder.save_weights(file+'_encoder_weights.h5')
        self.decoder.save_weights(file+'_decoder_weights.h5')
        self.autoencoder.save_weights(file+'_autoencoder_weights.h5')
        self.classifier.save_weights(file+'_classifier_weights.h5')
        self.classifier.save(file+'_classifier.h5')

    def load_model(self, file):
        self.classifier.load(file+'.h5')
        return self.classifier

    def load_weights(self, file):
        self.encoder.load_weights(file+'_encoder_weights.h5')
        self.decoder.load_weights(file+'_decoder_weights.h5')
        self.autoencoder.load_weights(file+'_autoencoder_weights.h5')
        self.classifier.load_weights(file+'_classifier_weights.h5')
        return self.encoder, self.decoder, self.autoencoder, self.classifier

"""
# Test per vedere se si e' rotto qualcosa

data = np.random.random((1000, 2, 500))
[x_train, y_train, x_valid, y_valid] = prepare_data(data, training_percentage=80)
model = define_model()
print(model.summary())
model = train_model(model, x_train, y_train)
y_pred = model([x_valid])
y_pred = np.asarray(y_pred)
RMSE = rmse(y_valid, y_pred)
print(RMSE)
plt.plot(y_pred.flatten())
plt.plot(y_valid.flatten())
plt.legend(['predicted', 'validation'])
plt.show()
"""
