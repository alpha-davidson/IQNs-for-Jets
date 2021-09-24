import os

import numpy as np
import tensorflow as tf
import random as rn

from quantileNetwork import QuantileNet, make_dataset

# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


# Trig function to model
def func(x):
    return(np.cos(4*x+1)/3+x*np.sin(x)/3)


def main():
    # Declare network
    network = QuantileNet()

    # 5 hidden layers with 50 perceptrons each, swish activation
    network.add(tf.keras.layers.Dense(50))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(50))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(50))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(50))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(50))
    network.add(tf.keras.layers.Activation("swish"))
    network.add(tf.keras.layers.Dense(2))

    # We want 100,000 training examples
    samples = 100000

    # training data, 1 dimensional input, Gaussian noise with sd 0.1*x
    x_vals_t = np.array([np.linspace(-3, 3, samples)]).T
    noise_t = np.random.normal(0, 0.1, size=x_vals_t.shape)*x_vals_t
    y_vals_t = func(x_vals_t)+noise_t

    # Turn the normal data into a dataset better for the quantile network.
    x_vals_t, y_vals_t = make_dataset(x_vals_t,   # input x
                                      y_vals_t,   # input y
                                      1,          # x dims
                                      1,          # y dims
                                      samples)    # examples

    # Repeat for validation data, 10,000 samples this time
    validation_samples = 10000
    x_vals_v = np.array([np.linspace(-3, 3, validation_samples)]).T
    noise_t = np.random.normal(0, 0.1, size=x_vals_v.shape)*x_vals_v
    y_vals_v = func(x_vals_v)+noise_t

    # Make the quantile net training set
    x_vals_v, y_vals_v = make_dataset(x_vals_v, y_vals_v, 1, 1,
                                      validation_samples)

    # Train for 100 epochs, restore best weights
    epochs = 100
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  restore_best_weights=True)]

    for x in range(4):
        network.compile(optimizer=tf.keras.optimizers.Adam(0.01 * (10**(-x)),
                                                           amsgrad=True),
                        loss=network.loss,
                        run_eagerly=False)

        network.fit(
            x_vals_t,
            y_vals_t,
            validation_data=(
                x_vals_v,
                y_vals_v),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=512)

        # Save the network
        network.save("Trig_Gaussian", save_traces=False)

if(__name__ == "__main__"):
    main()
