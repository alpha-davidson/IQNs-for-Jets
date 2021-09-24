
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from math import pi
from scipy.special import erf
from quantileNetwork import QuantileNet, sample_net, predict_dist


def pdf_func(x):
    # pdf function for double Gaussian
    val = np.exp(-0.5*((x-0.5)/0.1)**2)
    val += np.exp(-0.5*((x+0.5)/0.1)**2)
    val = val/(0.2*np.sqrt(2*pi))
    return(val)


def cdf_func(x):
    # cdf function for double Gaussian
    val = 0.25*(1+erf((x-0.5)/(0.1*np.sqrt(2))))
    val += 0.25*(1+erf((x+0.5)/(0.1*np.sqrt(2))))
    return(val)


def main():
    model_name = "Multi_Modal"  # Network name
    model = QuantileNet()  # Used to load network
    sample_num = 100000  # Number of random samples to draw

    # Need to supply custom objects in order to load model
    custom_objects = model.custom_objects()
    new_model = tf.keras.models.load_model(model_name,
                                           custom_objects=custom_objects)

    # Input is just a single 0
    model_input = tf.cast(np.zeros([1, 1]), tf.float32)

    out = sample_net(new_model,              # Network
                     sample_num,             # Number of samples
                     model_input,            # Input
                     model_input.shape[1],   # Number of different inputs
                     model_input.shape[0],   # 1d input
                     2)                      # 2d output

    dim_1 = np.array(out[0, :, 0])  # 1st predicted dimension
    dim_2 = np.array(out[0, :, 1])  # 2nd predicted dimension

    # true samples
    x_vals_v = np.zeros([sample_num//2, 1])
    noise_1_t = np.concatenate([np.random.normal(-0.5, 0.1,
                                                 size=x_vals_v.shape),
                                np.random.normal(0.5, 0.1,
                                                 size=x_vals_v.shape)],
                               axis=0)
    np.random.shuffle(noise_1_t)
    noise_2_t = np.concatenate([np.random.normal(-0.5, 0.1,
                                                 size=x_vals_v.shape),
                                np.random.normal(0.5, 0.1,
                                                 size=x_vals_v.shape)],
                               axis=0)
    np.random.shuffle(noise_2_t)
    true_samples = np.squeeze(np.array([noise_1_t, noise_2_t])).T

    dim_1_true = np.array(true_samples[:, 0])
    dim_2_true = np.array(true_samples[:, 1])

    # Plot data
    plt.figure(figsize=[7, 7])
    plt.hist2d(dim_1, dim_2, bins=50, range=[[-0.8, 0.8], [-0.8, 0.8]])
    plt.title("2d hist of sampled data")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist2d(dim_1_true, dim_2_true, bins=50, range=[[-0.8, 0.8],
                                                       [-0.8, 0.8]])
    plt.title("2d hist of true data")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist(dim_1, bins=100, range=(-0.8, 0.8), label="predicted",
             histtype="step")
    plt.hist(dim_1_true, bins=100, range=(-0.8, 0.8), label="true",
             histtype="step")
    plt.title("1st dimension samples")
    plt.legend()
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist(dim_2, bins=100, range=(-0.8, 0.8), label="predicted",
             histtype="step")
    plt.hist(dim_2_true, bins=100, range=(-0.8, 0.8), label="true",
             histtype="step")
    plt.title("2nd dimension samples")
    plt.legend()
    plt.show()

    # Generate pdf and cdf for 10,000 quantiles between 0 and 1
    quants = tf.cast(np.linspace(0, 1, 10000), tf.float32)
    cdf, pdf, quantiles = predict_dist(new_model,  # Model object
                                       quants,    # Quantiles to sample at
                                       model_input,  # Input examples
                                       model_input.shape[1],  # Number of
                                                              # input examples
                                       0,  # Current sampling dim
                                       model_input.shape[0],  # Input dim
                                       2)  # Output dim

    # pdf, cdf, and quantile function for 1st dimension
    plt.figure(figsize=[7, 7])
    plt.plot(quantiles, cdf[0, :], label="pred")
    plt.plot(cdf_func(np.linspace(-0.8, 0.8, 1000)),
             np.linspace(0.00-0.8, 0.8, 1000), label="true")
    plt.legend()

    plt.title("quantile")
    plt.show()
    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], quantiles, label="pred")
    plt.plot(np.linspace(0.00-0.8, 0.8, 1000),
             cdf_func(np.linspace(-0.8, 0.8, 1000)), label="true")
    plt.legend()
    plt.title("cdf")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], pdf[0, :], label="pred")
    plt.plot(np.linspace(-0.8, 0.8, 1000),
             pdf_func(np.linspace(-0.8, 0.8, 1000)), label="true")
    plt.legend()
    plt.title("pdf")
    plt.show()

    cdf, pdf, quantiles = predict_dist(new_model, quants,
                                       tf.cast(model_input, tf.float32),
                                       model_input.shape[1], 1,
                                       model_input.shape[0], 2,
                                       previous_samples=model_input)

    # pdf, cdf, and quantile function for 1st dimension
    plt.figure(figsize=[7, 7])
    plt.plot(quantiles, cdf[0, :], label="pred")
    plt.plot(cdf_func(np.linspace(-0.8, 0.8, 1000)),
             np.linspace(0.00-0.8, 0.8, 1000), label="true")
    plt.legend()
    plt.title("quantile")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], quantiles, label="pred")
    plt.plot(np.linspace(0.00-0.8, 0.8, 1000),
             cdf_func(np.linspace(-0.8, 0.8, 1000)), label="true")
    plt.legend()
    plt.title("cdf")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], pdf[0, :], label="pred")
    plt.plot(np.linspace(-0.8, 0.8, 1000),
             pdf_func(np.linspace(-0.8, 0.8, 1000)), label="true")
    plt.legend()
    plt.title("pdf")
    plt.show()

if(__name__ == "__main__"):
    main()
