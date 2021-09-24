import os

import numpy as np
import tensorflow as tf
import random as rn
import matplotlib.pyplot as plt

from scipy.stats import beta
from quantileNetwork import QuantileNet, make_dataset

# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


def main():
    # Declare network
    network = QuantileNet()

    # Two hidden layers with 10 perceptrons each, sigmoid activation
    network.add(tf.keras.layers.Dense(10))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(10))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(2))

    # We want 100,000 training examples
    samples = 100000

    # training data, 2 dimensional examples, each dimension is a different
    # beta distribution with alphas 0.5 and 2 and betas 0.5 and 5
    x_vals_t = np.zeros([samples, 1])
    noise_1_t = beta.rvs(0.5, 0.5, size=x_vals_t.shape)
    noise_2_t = beta.rvs(2, 5, size=x_vals_t.shape)
    np.random.shuffle(noise_2_t)

    y_vals_t = np.squeeze(np.array([noise_1_t, noise_2_t])).T

    # Turn the input data into a dataset better for the quantile network.
    x_vals_t, y_vals_t = make_dataset(x_vals_t,   # input x
                                      y_vals_t,   # input y
                                      1,          # x dims
                                      2,          # y dims
                                      samples)    # examples

    # Repeat for validation data, use only 10% the amount of data used for
    # training
    x_vals_v = np.zeros([samples//10, 1])
    noise_1_t = beta.rvs(0.5, 0.5, size=x_vals_v.shape)
    noise_2_t = beta.rvs(2, 5, size=x_vals_v.shape)
    y_vals_v = np.squeeze(np.array([noise_1_t, noise_2_t])).T

    # Turn validation data into useful dataset
    x_vals_v, y_vals_v = make_dataset(x_vals_v, y_vals_v, 1, 2, samples//10)

    # Train for 100 epochs, restore best weights, patience of 10
    epochs = 100
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                  restore_best_weights=True)]

    # Train 4 times with decreasing learning rates, restor best weights
    # each time, best weights are determined by lowest loss, use AMS grad
    # optimizer
    for x in range(4):
        network.compile(optimizer=tf.keras.optimizers.Adam(0.01*(10**(-x)),
                                                           amsgrad=True),
                        loss=network.loss)

        network.fit(x_vals_t, y_vals_t, validation_data=(x_vals_v, y_vals_v),
                    epochs=epochs, callbacks=callbacks, batch_size=512)

        # Save the network
        network.save("Beta", save_traces=False)

if(__name__ == "__main__"):
    main()
