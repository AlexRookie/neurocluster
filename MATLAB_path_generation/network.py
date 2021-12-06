import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, epochs=50, batch=32, learn_rate=0.0001):
        self.epochs = int(epochs)
        self.batch_size = int(batch)
        self.l_rate = learn_rate
        self.model = []

    def rmse(self, y_true, y_pred):
        # Root mean squared error (rmse) for regression
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def define_model(self, points):
        
        encoder_input = tf.keras.layers.Input(shape=(2, int(points), ), batch_size=None, name='EncoderInput')
        x = tf.keras.layers.Reshape((2,int(points),1), name='Reshape-1')(encoder_input)
        x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,11), activation=None, padding='valid', name='Conv2D-1')(x)
        x = tf.keras.layers.Conv2D(filters=7, kernel_size=(1,11), activation=None, padding='valid', name='Conv2D-2')(x)
        x = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,11), activation=None, padding='valid', name='Conv2D-3')(x)
        x = tf.keras.layers.Flatten(name='Flatten')(x)
        x = tf.keras.layers.Dense(units=20, activation='relu', use_bias=False, name='Dense-1')(x)
        encoded = tf.keras.layers.Dense(units=10, activation='relu', use_bias=False, name='Dense-2')(x)
        #x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv1D-1')(encoder_input)
        #x = tf.keras.layers.MaxPooling1D(2, padding='same', name='MaxPooling1D-1')(x)
        #x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same', name='Conv1D-2')(x)
        #encoded = tf.keras.layers.MaxPooling1D(2, padding='same', name='MaxPooling1D-2')(x)
        # at this point the representation is (1, 8) i.e. 8-dimensional
        
        decoder_input = tf.keras.layers.Input(shape=(encoded.shape[1], ), batch_size=None, name='DecoderInput')
        x = tf.keras.layers.Dense(units=20, activation='relu', use_bias=False, name='Dense-3')(decoder_input)
        x = tf.keras.layers.Dense(units=400, activation='relu', use_bias=False, name='Dense-4')(x)
        x = tf.keras.layers.Reshape((2,20,10), name='Reshape-2')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=7, kernel_size=(1,11), activation=None, padding='valid', name='Deconv2D-1')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=5, kernel_size=(1,11), activation=None, padding='valid', name='Deconv2D-2')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,11), activation=None, padding='valid', name='Deconv2D-3')(x)
        decoded = tf.keras.layers.Reshape((2, int(points),), name='Reshape-3')(x)
        #x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same', name='Conv1D-3')(decoder_input)
        #x = tf.keras.layers.UpSampling1D(2, name='UpSampling1D-1')(x)
        #x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same', name='Conv1D-4')(x)
        #x = tf.keras.layers.UpSampling1D(2, name='UpSampling1D-2')(x)
        #decoded = tf.keras.layers.Conv1D(50, 3, activation='sigmoid', name='Conv1D-5')(x)
        
        # Encoder model
        self.encoder = tf.keras.models.Model(inputs = encoder_input, outputs = encoded)
        # Decoder model
        self.decoder = tf.keras.models.Model(inputs = decoder_input, outputs = decoded)
        # Autoencoder model
        self.autoencoder = tf.keras.models.Model(inputs = encoder_input, outputs = self.decoder(encoded))
        
        return self.encoder, self.decoder, self.autoencoder

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

    def train_model(self, inputs):
        inputs = np.asarray(inputs)
        #if len(self.y_train.shape) < len(self.x_train.shape):
        #    self.y_train = np.expand_dims(self.y_train, axis=2)

        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.l_rate),
                                 loss = 'mean_squared_error',
                                 metrics = [self.rmse])

        fit = self.autoencoder.fit(inputs, inputs,
                                   epochs = self.epochs,
                                   batch_size = self.batch_size,
                                   shuffle = True,
                                   verbose = 1)

        return self.encoder, self.decoder, self.autoencoder, fit

    def predict(self, model, inputs):
        inputs = np.asarray(inputs)
        pred = model(inputs) # == model.predict([inputs])
        pred = np.array(pred)
        return pred

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
