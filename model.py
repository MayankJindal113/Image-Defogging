import numpy as np
import pickle
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Load dataset
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize and reduce dataset size for testing
x_train = x_train[:1000].astype('float32') / 255.0
x_test = x_test[:200].astype('float32') / 255.0

# Define a simple autoencoder
autoencoder = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=2, batch_size=32, validation_data=(x_test, x_test))

# Save the trained model using pickle
with open('autoencoder_model.pkl', 'wb') as file:
    pickle.dump(autoencoder, file)
