import numpy as np
import tensorflow as tf


def make_dataset(input_vals, output_vals, input_dims, output_dims, samples):
    """
    A function for turning typical machine learning training sets into
    quantile one. This function turns a typical input, output dataset into
    a dataset of the following input form:
    input, one hot encoding of output dim to predict, output dims already
    predicted, zero padding

    ----------
    input_vals: inputs to the network
    output_vals: outputs of the network
    input_dims: dimension of input data
    output_dims: dimension of output data
    samples: number of training examples

    Returns
    -------
    dset_in: the new input dataset
    dset_out: the new output dataset
    """

    dset_in = []
    dset_out = []
    for x in range(output_dims):
        temp = [input_vals]
        for y in range(x):
            temp.append(np.zeros((samples, 1)))

        temp.append(np.ones((samples, 1)))

        for y in range(x+1, output_dims):
            temp.append(np.zeros((samples, 1)))

        for y in range(0, x):
            temp.append(output_vals[:, y:y+1])

        for y in range(x, output_dims-1):
            temp.append(np.zeros((samples, 1)))

        dset_in.append(np.concatenate(temp, axis=1))
        dset_out.append(output_vals[:, x:x+1])

    dset_in = np.concatenate(dset_in, axis=0)
    dset_out = np.concatenate(dset_out, axis=0)
    return(dset_in, dset_out)


class QuantileNet(tf.keras.Model):
    """
    An implementation of a quantile network. For intput data D and output dims
    x, y, z, ..., this network learns p(x), p(y|x), p(z|x, y), ... and can be
    used to sample from p(x, y, z, ...)
    """
    def __init__(self, grad_loss_scale=100, network_type="normalizing",
                 clip=1e-7):
        """
        ----------
        grad_loss_scale: value to scale the graident loss scle. The gradient
            loss comes from negative slopes from the quantile function
        network_type: if this is 'normalizing' the network learns two
            approximations to the quanilte function, one with no normalization
            to set the scale for the other approxiamtion
        clip: distance from 1 and -1 beyond which the prediction value is
            fixed

        Returns
        -------
        None
        """
        super(QuantileNet, self).__init__()

        if(network_type == "normalizing"):
            self.loss = self.normalizing_loss
            self.inner_call = self.normalizing_call
        else:
            self.loss = self.no_normalizing_loss
            self.inner_call = self.no_normalizing_call

        self.grad_loss_scale = grad_loss_scale
        self.net_layers = []
        self.clip = clip

    def custom_objects(self):
        custom = {"QuantileNet": self,
                  "no_normalizing_loss": self.no_normalizing_loss,
                  "normalizing_loss": self.normalizing_loss,
                  "loss": self.loss}
        return(custom)

    def add(self, layer):
        """
        Add a layer to the network
        ----------
        layer: a layer to add to the network

        Returns
        -------
        None
        """
        self.net_layers.append(layer)

    def normalizing_call(self, inputs):
        """
        This function accepts inputs to the network and creates output. It
        also randomly picks and assigns quantiles to predict. As this is the
        normalizing call, it makes both a prediction from the base network and
        from the normalized network with information from the median and
        median +/- 1sd from the base network

        Parameters
        ----------
        inputs : The inputs to the network

        Returns
        -------
        grad_loss : The loss associated with negative quantile slopes
        output : The output of the quantile network
        """

        # Find number of examples
        count = 0
        for input_val in tf.transpose(inputs):
            count += 1
        grad_loss = np.float32(0.0)

        # Sample median - 1 sd, median, and median + 1 sd
        quantiles_low = -2.04*tf.ones(shape=[1, count],
                                      dtype=tf.float32)**(1)
        quantiles_mid = 0*tf.ones(shape=[1, count],
                                  dtype=tf.float32)**(1)
        quantiles_high = 2.04*tf.ones(shape=[1, count],
                                      dtype=tf.float32)**(1)

        inputs_low = tf.transpose(tf.concat([inputs, quantiles_low], axis=0))
        inputs_mid = tf.transpose(tf.concat([inputs, quantiles_mid], axis=0))
        inputs_high = tf.transpose(tf.concat([inputs, quantiles_high], axis=0))

        for x in range(len(self.net_layers)):
            layer = self.net_layers[x]
            inputs_low = layer(inputs_low)
            inputs_mid = layer(inputs_mid)
            inputs_high = layer(inputs_high)

        out_low = inputs_low[:, 1:2]
        out_mid = inputs_mid[:, 1:2]
        out_high = inputs_high[:, 1:2]

        shift = out_mid
        scale = (out_high-out_low)/2

        # Randomly sample quantiles
        quantiles = tf.random.uniform(shape=[1, count], minval=0, maxval=1,
                                      dtype=tf.float32)
        quantiles = quantiles*6 - 3

        # Full inputs
        inputs = tf.transpose(tf.concat([inputs, quantiles], axis=0))

        # Make predictions and get gradients
        out_norm = None
        out_base = None
        with tf.GradientTape(persistent=True) as g:
            g.watch(inputs)
            val = inputs
            for x in range(len(self.net_layers)):
                layer = self.net_layers[x]
                val = layer(val)
            out_norm = val[:, 0:1]
            out_base = val[:, 1:2]

        # Calculate gradient losses
        grads = g.gradient(out_norm, inputs)[:, -1]
        loss = self.grad_loss_scale*tf.math.square(tf.where(grads < 0, grads,
                                                            0))
        grad_loss += tf.reduce_mean(loss)

        grads = g.gradient(out_base, inputs)[:, -1]
        loss = self.grad_loss_scale*tf.math.square(tf.where(grads < 0, grads,
                                                            0))
        grad_loss += tf.reduce_mean(loss)

        # Transform edges of normalized network prediction
        out_norm = tf.clip_by_value((out_norm+3)/6, -1+self.clip, 1-self.clip)

        # Unnormalize output prediction
        out_norm = tf.math.atanh(out_norm)*scale+shift
        output = tf.concat([out_norm, out_base, tf.transpose(quantiles)],
                           axis=1)
        return(grad_loss, output)

    def no_normalizing_call(self, inputs):
        """
        This function accepts inputs to the network and creates output. It
        also randomly picks and assigns quantiles to predict. As this is the
        normalizing call, it makes both a prediction from the base network and
        from the normalized network with information from the median and
        median +/- 1sd from the base network

        Parameters
        ----------
        inputs : The inputs to the network

        Returns
        -------
        grad_loss : The loss associated with negative quantile slopes
        output : The output of the quantile network
        """

        count = 0
        # Find number of examples
        for input_val in tf.transpose(inputs):
            count += 1
        grad_loss = np.float32(0.0)

        # Randomly sample quantiles
        quantiles = tf.random.uniform(shape=[1, count], minval=0, maxval=1,
                                      dtype=tf.float32)
        quantiles = quantiles*6 - 3

        # Full inputs
        inputs = tf.transpose(tf.concat([inputs, quantiles], axis=0))

        # Make predictions and get gradients
        out_base = None
        with tf.GradientTape(persistent=False) as g:
            g.watch(inputs)
            val = inputs
            for x in range(len(self.net_layers)):
                layer = self.net_layers[x]
                val = layer(val)
            out_base = val

        # Calculate gradient losses
        grads = g.gradient(out_base, inputs)[:, -1]
        loss = self.grad_loss_scale*tf.math.square(tf.where(grads < 0, grads,
                                                            0))
        grad_loss += tf.reduce_mean(loss)

        output = tf.concat([out_base, tf.transpose(quantiles)],
                           axis=1)
        return(grad_loss, output)

    def call(self, inputs):
        """
        This function registers the gradLoss and calls the network outside of
        the tf.function
        Parameters
        ----------
        inputs : The inputs to the network
        Returns
        -------
        outputVal : The output of the network
        """

        inputs = tf.transpose(inputs)
        grad_loss, output_val = self.inner_call(inputs)
        self.add_loss(grad_loss)
        return(output_val)

    def normalizing_loss(self, y_actual, y_pred):
        """
        This function calcualtes the quantile loss. Note that the quantiles
        give to the network are actually between -3 and 3 to maximize the
        range they cover. This must be undone by the loss function. As
        this is for the normalizing network it accepts two predictions,
        one for each network type

        Parameters
        ----------
        y_actual : The true outputs
        y_pred : The predicted outputs, along with the quantiles
        Returns
        -------
        outputVal : The output of the network
        """

        quants = tf.expand_dims(y_pred[:, -1], 1)
        quants = (quants+3)/6

        y_pred_1 = tf.expand_dims(y_pred[:, 0], 1)
        val = y_actual - y_pred_1
        loss_val = tf.where(val < 0.0, tf.abs((-1+quants)), quants)
        loss_val = loss_val*tf.abs(val)
        loss_val = tf.reduce_mean(loss_val)
        true_loss = loss_val

        y_pred_2 = tf.expand_dims(y_pred[:, 1], 1)
        val = y_actual-y_pred_2
        loss_val = tf.where(val < 0.0, tf.abs((-1+quants)), quants)
        loss_val = loss_val*tf.abs(val)
        loss_val = tf.reduce_mean(loss_val)

        true_loss += loss_val
        return(true_loss)

    def no_normalizing_loss(self, y_actual, y_pred):
        """
        This function calcualtes the quantile loss. Note that the quantiles
        give to the network are actually between -3 and 3 to maximize the
        range they cover. This must be undone by the loss function.

        Parameters
        ----------
        y_actual : The true outputs
        y_pred : The predicted outputs, along with the quantiles
        Returns
        -------
        true_loss : The loss of the network predictions
        """
        quants = (tf.expand_dims(y_pred[:, -1], 1)+3)/6

        y_pred = tf.expand_dims(y_pred[:, 0], 1)
        val = y_actual-y_pred
        loss_val = tf.where(val < 0.0, tf.abs((-1+quants)), quants)
        loss_val = loss_val*tf.abs(val)
        loss_val = tf.reduce_mean(loss_val)
        true_loss = loss_val
        return(true_loss)


def sample_net(quantile_object, quantile_samples, inputs, input_count,
               input_dims, output_dims, network_type="normalizing",
               clip=1e-7):
    """
    This function is used for making predictions from quantile nets.

    Parameters
    ----------
    quantile_object : The quantile network object
    quantile_samples : The number of samples to take for each input
    inputs : Input data
    input_count : Number of examples
    input_dims : Dimension of input data
    output_dims : Dimension of output distribution
    network_type: if this is 'normalizing' the network learns two
        approximations to the quanilte function, one with no normalization
        to set the scale for the other approxiamtion
    clip: distance from 1 and -1 beyond which the prediction value is
        fixed
    Returns
    -------
    output : The output of the quantile net. It has shape
        (input_count, quantile_samples, output_dims)
    """

    clip = 1 - clip

    # Declare two possible prediction functions
    @tf.function(jit_compile=True)
    def predict_normalizing(val):
        """
        Run interior predictions within graph
        """
        # median minus 1 sd
        val_low = tf.identity(val[:, :-1])
        quantiles_low = -2.04*tf.ones(shape=val[:, -2:-1].shape,
                                      dtype=tf.float32)
        val_low = tf.concat([val_low, quantiles_low], axis=1)

        # median
        val_mid = tf.identity(val[:, :-1])
        quantiles_mid = tf.zeros(shape=val[:, -2:-1].shape, dtype=tf.float32)
        val_mid = tf.concat([val_mid, quantiles_mid], axis=1)

        # median plus 1 sd
        val_high = tf.identity(val[:, :-1])
        quantiles_high = 2.04*tf.ones(shape=val[:, -2:-1].shape,
                                      dtype=tf.float32)
        val_high = tf.concat([val_high, quantiles_high], axis=1)

        # make predictions
        for x in range(len(quantile_object.net_layers)):
            layer = quantile_object.net_layers[x]
            val = layer(val)
            val_low = layer(val_low)
            val_mid = layer(val_mid)
            val_high = layer(val_high)

        # get actual value, plus normalizing info
        val = val[:, 0:1]
        val_low = val_low[:, 1:2]
        val_mid = val_mid[:, 1:2]
        val_high = val_high[:, 1:2]

        # transform prediction
        val = tf.clip_by_value((val+3)/6, -clip, clip)
        val = tf.math.atanh(val)*(val_high-val_low)/2+val_mid
        return(val)

    @tf.function(jit_compile=True)
    def predict_no_normalizing(val):
        """
        Run interior predictions within graph
        """
        for x in range(len(quantile_object.net_layers)):
            layer = quantile_object.net_layers[x]
            val = layer(val)
        val = val[:, 0:1]
        return(val)

    if(network_type == "normalizing"):
        predict_inner = predict_normalizing
    else:
        predict_inner = predict_no_normalizing

    # Initial input, sampling the first dimension
    final_inputs = []
    sampling_locs = [[1.0]]
    for x in range(1, output_dims):
        sampling_locs.append([0.0])

    sampling_locs = tf.cast(sampling_locs, dtype=tf.float32)
    sampling_locs = tf.repeat(sampling_locs, quantile_samples, axis=1)
    numExamples = len(tf.transpose(inputs))

    # Random quantile value
    randomQuant = (tf.random.uniform([1, quantile_samples*numExamples],
                                     minval=-3, maxval=3,
                                     dtype=tf.dtypes.float32))

    # Input repeated according to number of samples
    extendedInput = tf.repeat(inputs, quantile_samples, axis=1)

    # No data for these values yet
    zeroInputs = tf.zeros([output_dims-1, quantile_samples*numExamples],
                          dtype=tf.float32)

    # Combine the input components
    sampling_locs = tf.repeat(sampling_locs, numExamples, axis=1)
    final_inputs.append(tf.squeeze(tf.concat([extendedInput,
                                              sampling_locs, zeroInputs,
                                              randomQuant], axis=0)))

    # Combine all the inputs
    final_inputs = tf.transpose(tf.concat(final_inputs, axis=1))

    # Declare two prediction functions
    def predict_main(val):
        """
        Main predict loop. Accepts as input the input data, returns
        the output data,
        """
        # Input already setup
        output = predict_inner(val)

        # Setup while loop to make general predictions
        i = tf.constant(1)

        def condition(i, current_state, current_prob):
            return(tf.less(i, output_dims))
        i, val, output = tf.while_loop(condition, predict_loop,
                                       [i, val, output])
        location = (i-1) % output_dims

        start_coord = input_dims+output_dims
        end_coord = input_dims+output_dims+location
        final_output = tf.concat([val[:, start_coord:end_coord],
                                  output,
                                  val[:, end_coord+1:-1]],
                                 axis=1)

        return(final_output)

    def predict_loop(x, val, output):
        """
        The main prediction driver. This reorganizes the input to the network
        as dimensions are samples. It accepts a counter variables to stop
        sampling when the required dimensions have been sampled.
        """
        location = x % output_dims
        old_location = (x-1) % output_dims

        # Raw input
        vectors = [val[:, 0:input_dims]]
        # Not sampling here
        for y in range(location):
            vectors.append(val[:, 0:1]*0)

        # Sampling here
        vectors.append(val[:, 0:1]*0+1)

        # Not sampling here
        for y in range(location+1, output_dims):
            vectors.append(val[:, 0:1]*0)
        # Previous coordinates
        start_coord = input_dims+output_dims
        end_coord = input_dims+output_dims+old_location
        vectors.append(val[:, start_coord:end_coord])
        # Just sampled coordinate
        vectors.append(output)

        # Previous coordinates
        vectors.append(val[:, input_dims+output_dims+old_location+1:-1])

        # New quantiles
        vectors.append(tf.random.uniform(output.shape, -3, 3, tf.float32))
        val = tf.concat(vectors, axis=1)
        output = predict_inner(val)
        return(tf.add(x, 1), val, output)

    # Generate the predictions
    output = predict_main(final_inputs)
    output = tf.reshape(output, (input_count, quantile_samples, output_dims))
    return(output)


def predict_dist(quantile_object, quantiles, inputs, input_count, current_dim,
                 input_dims, output_dims,
                 network_type="normalizing", clip=1e-7,
                 previous_samples=None):
    """
    This function is used for predicting pdf and cdfs using a quantile neural
    network

    Parameters
    ----------
    quantile_object : The quantile network object
    quantiles : quantiles at which to calculate the pdf and cdf
    inputs : Input data
    input_count : Number of examples
    current_dim : The output dimension at which to calculate the pdf and cdf
    input_dims : Dimension of input data
    output_dims : Dimension of output distribution
    network_type : if this is 'normalizing' the network learns two
        approximations to the quanilte function, one with no normalization
        to set the scale for the other approxiamtion
    clip : distance from 1 and -1 beyond which the prediction value is
        fixed
    previous_samples : values for previously sampled dimensions
    Returns
    -------
    cdf : The values of the distribution associated with the quantiles
    pdf : The probabilities of the quantile values
    quantiles : Same as the input quantiles
    """

    clip = 1 - clip

    # Declare two possible prediction functions
    @tf.function(jit_compile=True)
    def predict_normalizing(inputs):
        """
        Run interior predictions within graph
        """
        with tf.GradientTape(persistent=False) as g:
            g.watch(inputs)
            val = inputs
            # median minus 1 sd
            val_low = tf.identity(val[:, :-1])
            quantiles_low = -2.04*tf.ones(shape=val[:, -2:-1].shape,
                                          dtype=tf.float32)
            val_low = tf.concat([val_low, quantiles_low], axis=1)

            # median
            val_mid = tf.identity(val[:, :-1])
            quantiles_mid = tf.zeros(shape=val[:, -2:-1].shape,
                                     dtype=tf.float32)
            val_mid = tf.concat([val_mid, quantiles_mid], axis=1)

            # median plus 1 sd
            val_high = tf.identity(val[:, :-1])
            quantiles_high = 2.04*tf.ones(shape=val[:, -2:-1].shape,
                                          dtype=tf.float32)
            val_high = tf.concat([val_high, quantiles_high], axis=1)

            # make predictions
            for x in range(len(quantile_object.net_layers)):
                layer = quantile_object.net_layers[x]
                val = layer(val)
                val_low = layer(val_low)
                val_mid = layer(val_mid)
                val_high = layer(val_high)

            # get actual value, plus normalizing info
            val = val[:, 0:1]
            val_low = val_low[:, 1:2]
            val_mid = val_mid[:, 1:2]
            val_high = val_high[:, 1:2]

            # transform prediction
            val = tf.clip_by_value((val+3)/6, -clip, clip)
            val = tf.math.atanh(val)*(val_high-val_low)/2+val_mid
            cdf = val
        pdf = 1/(6*g.gradient(cdf, inputs))
        pdf = pdf[:, -1]

        return(cdf, pdf)

    @tf.function(jit_compile=True)
    def predict_no_normalizing(inputs):
        """
        Run interior predictions within graph
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch(inputs)
            val = inputs
            for x in range(len(quantile_object.net_layers)):
                layer = quantile_object.net_layers[x]
                val = layer(val)
            cdf = val[:, 0:1]
        pdf = 1/g.gradient(cdf, inputs)[:, -1]
        return(cdf, pdf)

    if(network_type == "normalizing"):
        predict_inner = predict_normalizing
    else:
        predict_inner = predict_no_normalizing

    # Initial input, sampling the first dimension
    final_inputs = []
    sampling_locs = []
    for x in range(0, current_dim):
        sampling_locs.append([0.0])
    sampling_locs.append([1.0])
    for x in range(current_dim+1, output_dims):
        sampling_locs.append([0.0])

    quantile_samples = tf.size(quantiles)
    sampling_locs = tf.cast(sampling_locs, dtype=tf.float32)
    sampling_locs = tf.repeat(sampling_locs, quantile_samples, axis=1)
    numExamples = len(tf.transpose(inputs))

    # Random quantile value
    randomQuant = tf.expand_dims(quantiles*6 - 3, 0)

    # Input repeated according to number of samples
    extendedInput = tf.repeat(inputs, quantile_samples, axis=1)

    # No data for these values yet
    zeroInputs = tf.zeros([output_dims-1-current_dim,
                           quantile_samples*numExamples],
                          dtype=tf.float32)

    # Combine the input components
    sampling_locs = tf.repeat(sampling_locs, numExamples, axis=1)

    if(previous_samples is not None):
        extendedPreviousSamples = tf.repeat(previous_samples, quantile_samples,
                                            axis=1)

        final_inputs.append(tf.squeeze(tf.concat([extendedInput,
                                                  sampling_locs,
                                                  extendedPreviousSamples,
                                                  zeroInputs,
                                                  randomQuant], axis=0)))
    else:
        final_inputs.append(tf.squeeze(tf.concat([extendedInput,
                                                  sampling_locs,
                                                  zeroInputs,
                                                  randomQuant], axis=0)))

    # Combine all the inputs
    final_inputs = tf.transpose(tf.concat(final_inputs, axis=1))

    # Generate the predictions
    cdf, pdf = predict_inner(final_inputs)
    cdf = tf.reshape(cdf, (input_count, quantile_samples))
    pdf = tf.reshape(pdf, (input_count, quantile_samples))
    return(cdf, pdf, quantiles)
