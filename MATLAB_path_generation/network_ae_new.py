import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.encoder     = []
        self.decoder     = []
        self.autoencoder = []
        self.classifier  = []
        self.units   = None
        self.classes = None
        self.batch_size = None
        self.epochs_unsup = None
        self.epochs_sup   = None
        self.l_rate       = None

    def define_model(self, units=(1,10), classes=0):
        self.units = np.asarray(units).astype(int)
        self.classes = np.asarray(classes).astype(int)
        
        encoder_input = tf.keras.layers.Input(shape=(self.units[0], self.units[1]), batch_size=None, name='EncoderInput')
        x = tf.keras.layers.Permute((2,1), input_shape=(self.units[0], self.units[1]), name='Permute-1')(encoder_input)
        x = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='valid', name='Conv1D-1')(x)
        x = tf.keras.layers.Conv1D(filters=15, kernel_size=3, padding='valid', name='Conv1D-2')(x)
        x = tf.keras.layers.Conv1D(filters=20, kernel_size=3, padding='valid', name='Conv1D-3')(x)
        x = tf.keras.layers.Flatten(name='Flatten')(x)
        encoded = tf.keras.layers.Dense(units=32, activation='tanh', name='Dense-1')(x)

        #encoder_input = tf.keras.layers.Input(shape=(2, int(points), ), batch_size=None, name='EncoderInput')
        #x = tf.keras.layers.Reshape((2,int(points),1), name='Reshape-1')(encoder_input)
        #x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,11), activation='relu', padding='valid', name='Conv2D-1')(x)
        #x = tf.keras.layers.Conv2D(filters=7, kernel_size=(1,11), activation='relu', padding='valid', name='Conv2D-2')(x)
        #x = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,11), activation='relu', padding='valid', name='Conv2D-3')(x)
        #x = tf.keras.layers.Flatten(name='Flatten')(x)
        #x = tf.keras.layers.Dense(units=20, activation='relu', use_bias=False, name='Dense-1')(x)
        #encoded = tf.keras.layers.Dense(units=10, activation='relu', use_bias=False, name='Dense-2')(x)
        ##x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv1D-1')(encoder_input)
        ##x = tf.keras.layers.MaxPooling1D(2, padding='same', name='MaxPooling1D-1')(x)
        ##x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same', name='Conv1D-2')(x)
        ##encoded = tf.keras.layers.MaxPooling1D(2, padding='same', name='MaxPooling1D-2')(x)
        ## at this point the representation is (1, 8) i.e. 8-dimensional
        
        decoder_input = tf.keras.layers.Input(shape=(encoded.shape[1], ), batch_size=None, name='DecoderInput')
        x = tf.keras.layers.Dense(units=80, name='Dense-2')(decoder_input)
        x = tf.keras.layers.Reshape((4,20), name='Reshape')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=15, kernel_size=3, padding='valid', name='Deconv1D-1')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=10, kernel_size=3, padding='valid', name='Deconv1D-2')(x)
        x = tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=5, padding='valid', name='Deconv1D-3')(x)
        decoded = tf.keras.layers.Permute((2,1), input_shape=(self.units[1], self.units[0]), name='Permute-2')(x)
        
        #decoder_input = tf.keras.layers.Input(shape=(encoded.shape[1], ), batch_size=None, name='DecoderInput')
        #x = tf.keras.layers.Dense(units=20, activation='relu', use_bias=False, name='Dense-3')(decoder_input)
        #x = tf.keras.layers.Dense(units=400, activation='relu', use_bias=False, name='Dense-4')(x)
        #x = tf.keras.layers.Reshape((2,20,10), name='Reshape-2')(x)
        #x = tf.keras.layers.Conv2DTranspose(filters=7, kernel_size=(1,11), activation='relu', padding='valid', name='Deconv2D-1')(x)
        #x = tf.keras.layers.Conv2DTranspose(filters=5, kernel_size=(1,11), activation='relu', padding='valid', name='Deconv2D-2')(x)
        #x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,11), activation='relu', padding='valid', name='Deconv2D-3')(x)
        #decoded = tf.keras.layers.Reshape((2, int(points),), name='Reshape-3')(x)
        ##x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same', name='Conv1D-3')(decoder_input)
        ##x = tf.keras.layers.UpSampling1D(2, name='UpSampling1D-1')(x)
        ##x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same', name='Conv1D-4')(x)
        ##x = tf.keras.layers.UpSampling1D(2, name='UpSampling1D-2')(x)
        ##decoded = tf.keras.layers.Conv1D(50, 3, activation='sigmoid', name='Conv1D-5')(x)

        x = tf.keras.layers.Dense(units=16, name='Dense-3')(encoded)
        classified = tf.keras.layers.Dense(units=self.classes, activation='softmax', name='Classifier')(x)
        
        # Encoder model
        self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoded)
        # Decoder model
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoded)
        # Autoencoder model
        self.autoencoder = tf.keras.models.Model(inputs=encoder_input, outputs=self.decoder(encoded))
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

    def rmse(self, y_true, y_pred):
        # Root mean squared error (rmse) for regression
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

    def train_model(self, x_train, y_train, epochs_unsup=50, epochs_sup=10, learn_rate=0.0001): 
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).astype(np.int8)
        self.epochs_unsup = int(epochs_unsup)
        self.epochs_sup = int(epochs_sup)
        self.l_rate = learn_rate

        # Unsupervised training
        self.autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate),
                                 loss = 'mean_squared_error',
                                 metrics = [self.rmse])

        fit_unsup = self.autoencoder.fit(x_train, x_train,
                                         epochs = self.epochs_unsup,
                                         batch_size = self.batch_size,
                                         shuffle = True,
                                         verbose = 1)

        # Set weights and non-trainable layers
        for i in range(2,7):
            self.classifier.layers[i].set_weights(self.encoder.layers[i].get_weights())
        #for layer in self.classifier.layers[0:7]:
        #    layer.trainable = False

        # Supervised training
        self.classifier.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate),
                                loss = 'categorical_crossentropy',
                                metrics = ['accuracy'])
        
        fit_sup1 = self.classifier.fit(x_train, y_train,
                                       epochs = self.epochs_sup,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       verbose = 1)

        # Set trainable layers
        #for layer in self.classifier.layers:
        #    layer.trainable = True
        #
        ## Over training
        #self.classifier.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate/100),
        #                        loss = 'categorical_crossentropy',
        #                        metrics = ['accuracy'])
        #
        #fit_sup2 = self.classifier.fit(x_train, y_train,
        #                               epochs = self.epochs_sup,
        #                               batch_size = self.batch_size,
        #                               shuffle = True,
        #                               verbose = 1)

        return self.encoder, self.decoder, self.autoencoder, self.classifier, fit_unsup, fit_sup1#, fit_sup2

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

"""
#=======================================================================================#

# Define model

input_img = keras.Input(shape=(28, 28, 1))

x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x4 = MaxPooling2D((2, 2), padding='same')(x3)
x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
encoded = MaxPooling2D((2, 2), padding='same')(x5)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

decoder_input = Input(shape=(4,4,8,))

x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(8, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu')(x9)
x11 = UpSampling2D((2, 2))(x10)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x11)

# Encoder model
encoder = Model(inputs=input_img, outputs=encoded)
print(encoder.summary())
# Decoder model
decoder = Model(inputs=decoder_input, outputs=decoded)
print(decoder.summary())

# Autoencoder model
autoencoder = Model(inputs=input_img, outputs=decoder(encoded))
print(autoencoder.summary())

# Compile model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#=======================================================================================#

# Load data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize values between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print("Train shape, test shape:")
print(x_train.shape)
print(x_test.shape)

# Train model
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

#=======================================================================================#

# Predict autoencoder
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.draw()
plt.pause(0.1)

#=======================================================================================#

pred = autoencoder.predict(np.expand_dims(x_test[0], axis=0))
plt.subplot(2, 1, 1)
plt.imshow(x_test[0].reshape(28, 28))
plt.subplot(2, 1, 2)
plt.imshow(pred.reshape(28, 28))
plt.show()

features = encoder.predict(np.expand_dims(x_test[0],axis=0))
plt.imshow(features.reshape((4, 4 * 8)).T)
plt.show()

synt = decoder.predict(features)
plt.imshow(synt.reshape(28, 28))
plt.show()

#=======================================================================================#

features2 = np.random.random(features.shape)

plt.subplot(1, 2, 1)
plt.imshow(features.reshape((4, 4 * 8)).T)
plt.subplot(1, 2, 2)
plt.imshow(features2.reshape((4, 4 * 8)).T)
plt.show()

synt2 = decoder.predict(features2)

plt.subplot(1, 2, 1)
plt.imshow(synt.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.imshow(synt2.reshape(28, 28))
plt.show()
"""
