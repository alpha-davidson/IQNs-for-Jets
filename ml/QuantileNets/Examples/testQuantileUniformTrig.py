import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from quantileNetwork import QuantileNet, sample_net


# Trig function to model
def func(x):
    return(np.cos(4*x+1)/3+x*np.sin(x)/3)


def main():
    model_name = "Trig_Uniform"  # Network name
    model = QuantileNet()        # Used to load network
    counts = 100                   # Test spots

    # Test input, does not need modified for predictions
    model_input = np.array([np.linspace(-3, 3, counts)])
    model_input = tf.cast(model_input, tf.float32)

    # Load model
    custom_objects = model.custom_objects()
    new_model = tf.keras.models.load_model(model_name,
                                           custom_objects=custom_objects)

    # 1000 samples per input
    sample_num = 10000

    # Call the network
    out = sample_net(new_model,             # Network
                     sample_num,             # Number of samples
                     model_input,           # Input
                     model_input.shape[1],  # Number of different inputs
                     model_input.shape[0],  # 1d input
                     1)                     # 1d output

    # Plot some output slices
    model_input = np.array(model_input[0, :])
    for x in range(0, len(out[:, 0, 0]), 20):
        noise_t = np.random.uniform(-1, 1, size=sample_num)
        noise_t = noise_t*model_input[x]*np.sqrt(3)*0.1

        plot_min = func(model_input[x]) - abs(model_input[x])*np.sqrt(3)*0.11
        plot_max = func(model_input[x]) + abs(model_input[x])*np.sqrt(3)*0.11

        plt.figure(figsize=[7, 7])
        plt.hist(np.squeeze(out[x, :, 0]), bins=50, label="predicted",
                 histtype="step", range=(plot_min, plot_max))
        plt.hist(func(model_input[x])+noise_t, bins=50, label="true",
                 histtype="step", range=(plot_min, plot_max))
        plt.legend()
        plt.title("Sample output slices")
        plt.show()

    # Calulate mean/std
    out = np.array(out[:, :, 0])
    output_mean = tf.reduce_mean(out, 1)
    output_std = tf.math.reduce_std(out, 1)

    true_output = func(model_input)

    # Plot mean, confidence interval over all inputs
    plt.figure(figsize=[7, 7])
    plt.plot(model_input, output_mean, color="r", label="output mean")
    plt.fill_between(model_input, output_mean - output_std,
                     output_mean + output_std, alpha=0.5, color="r",
                     label="output mean +/- 1 sd")
    plt.plot(model_input, true_output, color="k", label="true mean")
    plt.fill_between(model_input, true_output - 0.1*model_input,
                     true_output + 0.1*model_input, alpha=0.5, color="k",
                     label="true mean +/- 1 sd")
    plt.legend()
    plt.show()

    # Plot std over all inputs
    plt.figure(figsize=[7, 7])
    plt.plot(model_input, output_std, label="predicted")
    plt.plot(model_input, np.absolute(model_input*0.1), label="true")
    plt.title("standard deviation")
    plt.legend()
    plt.show()

if(__name__ == "__main__"):
    main()
