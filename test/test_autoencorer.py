# Simple autoencoder test
# Ref: https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

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
