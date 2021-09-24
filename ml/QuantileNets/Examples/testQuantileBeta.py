
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import beta
from quantileNetwork import QuantileNet, sample_net, predict_dist


def main():
    model_name = "Beta"  # Network name
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
                     model_input,           # Input
                     model_input.shape[1],  # Number of different inputs
                     model_input.shape[0],  # 1d input
                     2)                     # 2d output

    dim_1 = np.array(out[0, :, 0])  # 1st predicted dimension
    dim_2 = np.array(out[0, :, 1])  # 2nd predicted dimension

    # true samples
    x_vals_t = np.zeros([sample_num, 1])
    noise_1_t = beta.rvs(0.5, 0.5, size=x_vals_t.shape)
    noise_2_t = beta.rvs(2, 5, size=x_vals_t.shape)
    np.random.shuffle(noise_2_t)

    true_samples = np.squeeze(np.array([noise_1_t, noise_2_t])).T

    dim_1_true = np.array(true_samples[:, 0])
    dim_2_true = np.array(true_samples[:, 1])

    # Plot data
    plt.figure(figsize=[7, 7])
    plt.hist2d(dim_1, dim_2, bins=50, range=[[0, 1], [0, 1]])
    plt.title("2d hist of sampled data")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist2d(dim_1_true, dim_2_true, bins=50, range=[[0, 1], [0, 1]])
    plt.title("2d hist of true data")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist(dim_1, bins=100, range=(0, 1), label="predicted", histtype="step")
    plt.hist(dim_1_true, bins=100, range=(0, 1), label="true", histtype="step")
    plt.title("1st dimension samples")
    plt.legend()
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.hist(dim_2, bins=100, range=(0, 1), label="predicted", histtype="step")
    plt.hist(dim_2_true, bins=100, range=(0, 1), label="true", histtype="step")
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
    rv = beta(0.5, 0.5)  # beta distribution, alpha=0.5, beta=0.5
    plt.figure(figsize=[7, 7])
    plt.plot(quantiles, cdf[0, :], label="pred")
    plt.plot(rv.cdf(np.linspace(0, 1, 1000)),
             np.linspace(0.000, 1, 1000), label="true")
    plt.legend()

    plt.title("quantile")
    plt.show()
    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], quantiles, label="pred")
    plt.plot(np.linspace(0.000, 1, 1000),
             rv.cdf(np.linspace(0, 1, 1000)), label="true")
    plt.legend()
    plt.title("cdf")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], pdf[0, :], label="pred")
    plt.plot(np.linspace(0.001, 0.999, 1000),
             rv.pdf(np.linspace(0.001, 0.999, 1000)), label="true")
    plt.legend()
    plt.title("pdf")
    plt.show()

    cdf, pdf, quantiles = predict_dist(new_model, quants,
                                       tf.cast(model_input, tf.float32),
                                       model_input.shape[1], 1,
                                       model_input.shape[0], 2,
                                       previous_samples=model_input)

    # pdf, cdf, and quantile function for 1st dimension
    rv = beta(2, 5)  # beta distribution, alpha=0.2, beta=5
    plt.figure(figsize=[7, 7])
    plt.plot(quantiles, cdf[0, :], label="pred")
    plt.plot(rv.cdf(np.linspace(0, 1, 1000)),
             np.linspace(0.000, 1, 1000), label="true")
    plt.legend()
    plt.title("quantile")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], quantiles, label="pred")
    plt.plot(np.linspace(0.000, 1, 1000),
             rv.cdf(np.linspace(0, 1, 1000)), label="true")
    plt.legend()
    plt.title("cdf")
    plt.show()

    plt.figure(figsize=[7, 7])
    plt.plot(cdf[0, :], pdf[0, :], label="pred")
    plt.plot(np.linspace(0.001, 0.999, 1000),
             rv.pdf(np.linspace(0.001, 0.999, 1000)), label="true")
    plt.legend()
    plt.title("pdf")
    plt.show()

if(__name__ == "__main__"):
    main()
