import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np

# Load in the data
cifar100 = tf.keras.datasets.cifar100

# Distribute it to train and test set
(x_train, y_train), (x_val, y_val) = cifar100.load_data()
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

y_train = tf.one_hot(y_train,
                     depth=y_train.max() + 1,
                     dtype=tf.float64)
y_val = tf.one_hot(y_val,
                   depth=y_val.max() + 1,
                   dtype=tf.float64)

y_train = tf.squeeze(y_train)
y_val = tf.squeeze(y_val)

model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), activation='gelu',
                  input_shape=(32, 32, 3), padding='same'),
    layers.Conv2D(32, (3, 3),
                  activation='gelu',
                  padding='same'),
    layers.Conv2D(64, (3, 3),
                  activation='gelu',
                  padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3),
                  activation='gelu',
                  padding='same'),

    layers.Flatten(),
    layers.Dense(256, activation='gelu'),
    layers.BatchNormalization(),
    layers.Dense(256, activation='gelu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(100, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['AUC', 'accuracy']
)

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=64,
                 verbose=1,
                 validation_data=(x_val, y_val))