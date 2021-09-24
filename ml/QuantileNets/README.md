# Quantile Nets
This folder contains example code of how to use the quantile networks. The general principal behind them is to learn the quantile function of a set of data, which is the inverse cumulative distribtuion funcion. Using this quantile function, we can draw random samples from a distribution, as uniform samples of the range (0,1) when input to the quantile function give random samples from the full distribution. Additionally, using the autodifferentiation in TensorFlow, it is possible to take the derivative of the quantile functions and from it obtain the probability distribution function which gives rise to the quantile function.

As the quantile of a dataset only really makes sense with respect to one dimension we must turn a higher dimensional space into a set of 1-dimensional spaces. Thus, when tasked with learning multiple dimensions x, y, z, given data D, the network learns p(x|D), p(y|x,D), and p(z|x,y,D), as p(x,y,z|D) = p(x|D)p(y|x,D)p(z|x,y,D). This requires the network to make n calls to sample from n dimensional data.

For an interesting paper dealing with a version of quantile networks for generating distributions, see  https://arxiv.org/pdf/1806.05575.pdf.

A novel feature of this implementation of the quantile networks is an adaptive normalization method which can be applied during training. This method is applied by default, but can be shut off by passing a value other than 'normalizing' to the keyword argument network_type in the QuantileNet object and the prediction code. What it does is learn the quantile function two ways. First, it learns the default way where it directly predicts the quantile function. While it is doing this, it also learns the quantile function with the following activation function applied to the network output
```
arctanh((x+3)/6)*A + B
```
Here, x is the network output, B is the median output of the first method, and A is the distance between the 0.16 and 0.84 quantiles divided by 2 from the first method. What this method does is map a quantile function with asympototes at 0 and 1 to a bounded function by means of the arctanh, and uses A and B to transform to make the quantile function centered and scaled so that the arctanh function linearizes it as much as possible. This significantly improves the ability of the quantile net to learn the tails of distributions and its use is highly recomended. 

At the same level as this README in the repository is a file quantileNetwork.py. This contains all of the code necesary to train a quantile net and make predictions from it. Located with in the Examples folder is the code for training four different example networks and using them to make predictions. The quantileNetwork.py file is also located there for convnience. 


## QuantileNetwork File

Inside of quantileNet.py is the code for the actual QuantileNet object, as well as three additional functions. When training a model, the first of these to use is the helper function make_dataset. This function accepts the following:
- input_vals: inputs to the network
- output_vals: outputs of the network
- input_dims: dimension of input data
- output_dims: dimension of output data
- samples: number of training examples


It returns:
- dset_in: the new input dataset
- dset_out: the new output dataset
    
This function turns typical machine learning training sets into a quantile one. This function turns a typical input, output dataset into a dataset of the following input form:
(input, one hot encoding of output dim to predict, output dims already predicted, zero padding)

After this comes the actual QuantileNet object. The constructor contains three inputs:
	
- grad_loss_scale: value to scale the graident loss scle. The gradient
		loss comes from negative slopes from the quantile function, 
		default is 100
- network_type: if this is 'normalizing' the network learns two approximations to the quanilte function, one with no normalization to set the scale for the other approxiamtion, default is 'normalizing'
- clip: distance from 1 and -1 beyond which the prediction value is fixed, defautl is 1e-7

After initializing the network, simply call the add function, which merely adds the given layer to the network. An example is given below.
```
network = QuantileNet()
network.add(tf.keras.layers.Dense(20))
```

From here, the network can be trained like a typical Keras Sequential model. Once training is done, the network can be saved as normal, however save_traces must be set to false. See below for an example.
```
network.save("Beta", save_traces=False)
```

When loading a saved model from which to make predictions a dictionary with some of the special quantile functions must be suppled. This can be accomplished through using a function in the QuantileNet object designed just for this. Simply use code like the following:
```
model_name = "Beta"  # Network name
model = QuantileNet()  # Used to load network

# Need to supply custom objects in order to load model
custom_objects = model.custom_objects()
new_model = tf.keras.models.load_model(model_name,
									   custom_objects=custom_objects)
``` 

Once a network has been loaded there are two options with which predictions can be made. The first simply accepts a set of inputs and generates the requested number of samples that correspond to the output distribution. Note that the make_dataset function does not have to be used for either of these. This is the sample_net function. It accepts the following inputs:
- quantile_object : The quantile network object
- quantile_samples : The number of samples to take for each input
- inputs : Input data
- input_count : Number of examples
- input_dims : Dimension of input data
- output_dims : Dimension of output distribution
- network_type: if this is 'normalizing' the network learns two approximations to the quanilte function, one with no normalization to set the scale for the other approxiamtion, defualt is normalizing'
- clip: distance from 1 and -1 beyond which the prediction value is fixed, default is 1e-7

And it returns the follwoing output:
- output : The output of the quantile net. It has shape (input_count, quantile_samples, output_dims)

The other option for predictions, predict_dist, accepts as input a list of quantiles, a set of inputs, the current dimension of the output being predicted, and any previously sampled output values. It then returns data which can be used to construction the quantile function, cdf, and pdf for the output distribution. The inputs it accepts are:
- quantile_object : The quantile network object
- quantiles : quantiles at which to calculate the pdf and cdf
- inputs : Input data
- input_count : Number of examples
- current_dim : The output dimension at which to calculate the pdf and cdf
- input_dims : Dimension of input data
- output_dims : Dimension of output distribution
- network_type : if this is 'normalizing' the network learns two approximations to the quanilte function, one with no normalization to set the scale for the other approxiamtion, default is 'normalizing'
- clip : distance from 1 and -1 beyond which the prediction value is fixed, default is 1e-7
- previous_samples : values for previously sampled dimensions, default is None

It returns:
- cdf : The values of the distribution associated with the quantiles
- pdf : The probabilities of the quantile values
- quantiles : Same as the input quantiles
	
An example use case is the following:
```
cdf, pdf, quantiles = predict_dist(new_model, quants,
								   tf.cast(model_input, tf.float32),
								   model_input.shape[1], 1,
								   model_input.shape[0], 2,
								   previous_samples=model_input)

# pdf, cdf, and quantile function for 1st dimension
rv = beta(2, 5)  # beta distribution, alpha=0.2, beta=5
plt.figure()
plt.plot(quantiles, cdf[0, :], label="pred")
plt.plot(rv.cdf(np.linspace(0, 1, 1000)),
		 np.linspace(0.000, 1, 1000), label="true")
plt.legend()
plt.title("quantile")
plt.show()

plt.figure()
plt.plot(cdf[0, :], quantiles, label="pred")
plt.plot(np.linspace(0.000, 1, 1000),
		 rv.cdf(np.linspace(0, 1, 1000)), label="true")
plt.legend()
plt.title("cdf")
plt.show()

plt.figure()
plt.plot(cdf[0, :], pdf[0, :], label="pred")
plt.plot(np.linspace(0.001, 0.999, 1000),
		 rv.pdf(np.linspace(0.001, 0.999, 1000)), label="true")
plt.legend()
plt.title("pdf")
plt.show()
```

## Examples Folder

In the examples folder are 4 training files and 4 prediction files. Two correspond to learning two dimensional distributions with a single input, and two correspond to learning one dimensional distributions with multiple outputs.


The two dimensional problems are trainQuantileBeta.py and trainQuantileMultiModal.py. For the beta distribution problem, the first dimension is a beta distribution with alpha=0.5 and beta=0.5. The second distribution has alpha=2 and beta=5. For the multi-modal problem, both dimensions consist of two Gaussian distributions with standard deviation 0.1 and with means at -0.5 and 0.5. Running the prediction scripts for both results in plots of the sampled joint and marginal distributions, as well as calculated pdfs, cdfs, and quantile functions.

The 1 dimensional problems are trainQuantileGaussianTrig.py and trainQuantileUniformTrig.py. In both of these cases the function
```
f(x) = cos(4*x+1)/3+x*sin(x)/3
```
is learned. For the Gaussian problem, there is Gaussian noise in the data with mean determined by f, and standard deviation x/10. For the uniform problem the spread is adjusted so that the mean and standard deviation are the same as in the Gaussian case. In the prediction code a few graphs of the distributions predicted for a given x are presented, a plot of the mean and a 1 sd confidence interval over the training range, and a plot of the standard deviation over the training range.
