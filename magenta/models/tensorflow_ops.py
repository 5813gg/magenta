"""Various Tensorflow Ops used in models. All are class objects.

Note that this is being deprecated in favor of tf.Slim.
"""
import math

import numpy as np
import tensorflow.google as tf

from tensorflow.python.training import moving_averages


class Conv(object):
  """Convolutional object.

  We use Xavier weight initialization for the filters.

  Args:
    filter_size: A four-tuple representing the size of the filter map. Has the
      form [B, H, W, C].
    strides: Convolutional strides, defaults to [1, 1, 1, 1].
    name: Name of the scope.
    padding: Type of padding. See tf.nn.conv2d.
    constants: Dict of values that we can feed to replace the weights and bias.
    use_bias: Boolean determining if we add a bias term.
  """

  def __init__(self,
               filter_size,
               strides=None,
               name='Conv',
               padding='SAME',
               constants=None,
               use_bias=True):
    self.filter_size = filter_size
    self.name = name
    self.strides = strides or [1, 1, 1, 1]
    self.padding = padding
    self.use_bias = use_bias
    with tf.name_scope(self.name + '/vars'):
      if not constants:
        fan_in = np.float32(filter_size[0] * filter_size[1] * filter_size[2])
        stddev = 1.0 / np.sqrt(fan_in)
        self.weights = tf.Variable(
            tf.truncated_normal(
                filter_size, stddev=stddev), name='W')
        if use_bias:
          self.bias = tf.Variable(tf.zeros([filter_size[-1]]), name='b')
      else:
        self.weights = tf.constant(constants.weights[name])
        if use_bias:
          self.bias = tf.constant(constants.biases[name])

  def __call__(self, input_):
    with tf.name_scope(self.name):
      conv = tf.nn.conv2d(
          input_, self.weights, strides=self.strides, padding=self.padding)
      if self.use_bias:
        return tf.nn.bias_add(conv, self.bias)
      else:
        return conv


class DeConv(object):
  """DeConvolutional object.

  We use Xavier weight initialization for the filters.

  Args:
    filter_size: A four-tuple representing the size of the filter map. Has the
      form [B, H, W, C].
    name: Name of the scope.
    strides: Convolutional strides, defaults to [1, 1, 1, 1].
    padding: Type of padding. See tf.nn.conv2d.
    use_bias: Boolean determining if we add a bias term.
  """

  def __init__(self,
               filter_size,
               name='DeConv',
               strides=None,
               padding='SAME',
               use_bias=False):
    self.filter_size = filter_size
    self.name = name
    self.strides = strides or [1, 1, 1, 1]
    self.padding = padding
    self.use_bias = use_bias
    with tf.name_scope(name + '/vars'):
      fan_in = np.float32(filter_size[0] * filter_size[1] * filter_size[3])
      stddev = 1.0 / np.sqrt(fan_in)
      self.weights = tf.Variable(
          tf.truncated_normal(
              filter_size, stddev=stddev), name='W')
      if use_bias:
        self.bias = tf.Variable(tf.zeros([filter_size[-2]]), name='b')

  def __call__(self, input_):
    assert input_.get_shape().is_fully_defined(), 'Shape not fully defined'
    batch_size, height, width, _ = input_.get_shape().as_list()

    out_height = height * self.strides[1]
    out_width = width * self.strides[2]

    if self.padding == 'VALID':
      out_height += int(2 * math.floor(self.filter_size[0] / 2.0))
      out_width += int(2 * math.floor(self.filter_size[1] / 2.0))

    output_shape = [batch_size, out_height, out_width, self.filter_size[2]]

    with tf.name_scope(self.name):
      conv = tf.nn.conv2d_transpose(
          input_,
          self.weights,
          output_shape,
          strides=self.strides,
          padding=self.padding)
      if self.use_bias:
        return tf.nn.bias_add(conv, self.bias)
      else:
        return conv


class FC(object):
  """Fully connected operation.

  Uses Xavier weight initialization. Assumes first dimension is batch size.

  Args:
    output nodes: Number of fully connected nodes to calculate.
    name: Name of the scope.
    stddev: Float standard deviation of normal distribution used to initialize
      weights. Defaults to setting this using the input_tensor size and Xavier
      initialization.
    bias_init: Initial value added to each node.
  """

  def __init__(self, output_nodes, name='FC', stddev=None, bias_init=0.1):
    self.output_nodes = output_nodes
    self.name = name
    self.stddev = stddev
    self.bias_init = bias_init

  def __call__(self, input_):
    with tf.name_scope(self.name + '/vars'):
      input_shape = input_.get_shape().as_list()
      if len(input_shape) > 2:
        input_shape = (-1, np.prod(input_shape[1:]))  # Flatten each example
      shape = (input_shape[-1], self.output_nodes)

      if self.stddev is None:
        self.stddev = np.sqrt(1.0 / shape[0])

      self.weights = tf.Variable(
          tf.truncated_normal(
              shape, stddev=self.stddev), name='W')
      self.biases = tf.Variable(tf.ones(shape[-1]) * self.bias_init, name='b')
    with tf.name_scope(self.name):
      flattened_input = tf.reshape(input_, input_shape)
      product = tf.matmul(flattened_input, self.weights)
      tf.histogram_summary(self.name, product)
      return tf.nn.bias_add(product, self.biases)


class LogRelu(object):
  """This is a Relu with a log. We add an epsilon to prevent 0.

  Args:
    name: The name of the op.
    epsilon: A small float value to make the log non-zero.

  Returns:
    logrelu: The resulting tensorflow op.
  """

  def __init__(self, name='LogRelu', epsilon=1e-4):
    self.name = name
    self.epsilon = epsilon
    self.relu = Relu(name + '_relu')

  def __call__(self, input_):
    with tf.name_scope(self.name):
      logrelu = tf.log(self.relu(input_) + self.epsilon)
    return logrelu


class Relu(object):
  """Normal relu op.

  Args:
    name: The name of the op.

  Returns:
    relu: The resulting tensorflow op.
  """

  def __init__(self, name='Relu'):
    self.name = name

  def __call__(self, input_):
    with tf.name_scope(self.name):
      return tf.nn.relu(input_)


class Sigmoid(object):
  """Normal sigmoid op.

  Args:
    name: The name of the op.

  Returns:
    sigmoid: The resulting tensorflow op.
  """

  def __init__(self, name):
    self.name = name

  def __call__(self, input_):
    with tf.name_scope(self.name):
      return tf.nn.sigmoid(input_)


class Pool(object):
  """A pooling op.

  Args:
    ksize: See the tensorflow docs.
    strides: See the tensorflow docs.
    name: See the tensorflow docs.
    pooling: The type of pooling, either 'max' or 'avg'.

  Returns:
    pool: Either max_pool or avg_pool depending on the pooling input.
  """

  def __init__(self, ksize=None, strides=None, name='Pool', pooling='max'):
    self.ksize = ksize or [1, 1, 4, 1]
    self.strides = strides or [1, 1, 4, 1]
    self.name = name

    if pooling not in ['max', 'avg']:
      raise ValueError('Please submit a valid pooling.')
    self.pooling = pooling

  def __call__(self, input_):
    with tf.name_scope(self.name):
      if self.pooling == 'max':
        return tf.nn.max_pool(
            input_,
            ksize=self.ksize,
            strides=self.strides,
            padding='VALID')
      elif self.pooling == 'avg':
        return tf.nn.avg_pool(
            input_,
            ksize=self.ksize,
            strides=self.strides,
            padding='VALID')


class BatchNorm(object):
  """BatchNorm object for doing batch normalization.

  Args:
    name: Name of the scope.
    is_training: If training, calculate moving averages, otherwise initialize
      from checkpoint.
    decay: A float Tensor or float value. The moving average decay.
    epsilon: A small float number to avoid dividing by 0.
  """

  def __init__(self, name='BatchNorm', is_training=True, decay=0.999,
               epsilon=0.001):
    self.name = name
    self.decay = decay
    self.epsilon = epsilon
    self.is_training = is_training

  def __call__(self, input_):
    shape = input_.shape[-1]
    with tf.name_scope(self.name + '/vars'):
      self.beta = tf.Variable(tf.zeros(shape), name='beta')
      self.moving_mean = tf.Variable(
          tf.zeros(shape), name='moving_mean', trainable=False)
      self.moving_variance = tf.Variable(
          tf.ones(shape), name='moving_variance', trainable=False)
    with tf.name_scope(self.name) as scope:
      control_inputs = []
      if self.is_training:
        axis = list(range(len(input_.shape) - 1))
        mean, variance = tf.nn.moments(input_, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay)
        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.decay)
        control_inputs = [update_moving_mean, update_moving_variance]
      else:
        mean = self.moving_mean
        variance = self.moving_variance
      with tf.control_dependencies(control_inputs):
        return tf.nn.batch_normalization(
            input_,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=None,
            variance_epsilon=self.epsilon,
            name=scope)


class InstanceNorm(object):
  """InstanceNorm object.

  See https://arxiv.org/pdf/1607.08022v1.pdf for details.

  Args:
    name: Name of the scope.
    axis: A list of axes that we run the normalization over. If your input is
      [B, H, W, C], then the default axis=[1, 2] would correspond to normalizing
      over height and width.
    epsilon: A small float number to avoid dividing by 0.
  """

  def __init__(self, name='InstanceNorm', axis=None, epsilon=0.001):
    self.name = name
    self.epsilon = epsilon
    axis = axis or [1, 2]
    self.axis = sorted(axis)

  def __call__(self, input_):
    with tf.name_scope(self.name):
      mean, variance = tf.nn.moments(input_, self.axis)
      for indice in self.axis:
        mean = tf.expand_dims(mean, indice)
        variance = tf.expand_dims(variance, indice)
      with tf.name_scope(self.name, 'batchnorm',
                         [input_, mean, variance]):
        inverse_stddev = tf.math_ops.rsqrt(variance + self.epsilon)
        return (input_ - mean) * inverse_stddev


class StateSavingLSTM(object):
  """StateSaving Implementation.

  Args:
    num_units: The number of units in each LSTM.
    num_layers: The number of layers in the LSTM.
    name: The optional scope name.
  """

  def __init__(self, num_units, num_layers, name='StateSavingLSTM'):
    self.name = name
    with tf.variable_scope(self.name):
      self.cell = tf.nn.rnn_cell.MultiRNNCell(
          [
              tf.nn.rnn_cell.BasicLSTMCell(
                  num_units=num_units, state_is_tuple=True)
          ] * num_layers,
          state_is_tuple=True)

  def __call__(self, inputs, state_saver, state_names):
    """Call this during training.

    Args:
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size].
      state_saver: A state saver object with methods state and save_state.
      state_names: Python tuple of strings to use with the state_saver.

    Returns:
      result: A pair (outputs, state) where outputs is a length T list of
        outputs (one for each input) and states is the final state.
    """
    with tf.variable_scope(self.name):
      return tf.nn.state_saving_rnn(
          self.cell, inputs, state_saver, state_names)

  def inference(self, inputs, initial_state=None):
    """Call this to run inference.

    Args:
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size].
      initial_state: An optional initial state for the inference.

    Returns:
      result: A pair (outputs, state) where outputs is a length T list of
        outputs (one for each input) and states is the final state.
    """
    with tf.variable_scope(self.name):
      return tf.nn.rnn(self.cell,
                       inputs,
                       initial_state=initial_state)


def input_sequence_example(file_list, hparams):
  """Deserializes a single SequenceExample from tfrecord.

  Args:
    file_list: List of paths to tfrecord files of SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    seq_key: Key of SequenceExample as a string.
    context: Context of SequenceExample as dictionary key -> Tensor.
    sequence: Sequence of SequenceExample as dictionary key -> Tensor.
  """
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.tfrecord_readder()
  seq_key, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(
          shape=[hparams.one_hot_length], dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(
          shape=[], dtype=tf.int64)
  }

  context, sequence = tf.parse_single_sequence_example(
      serialized_example, sequence_features=sequence_features)
  return seq_key, context, sequence


def dynamic_rnn_batch(file_list, hparams):
  """Reads batches of SequenceExamples from tfrecord and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: List of tfrecord files of SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    inputs: Tensor of shape [batch_size, examples_per_sequence, one_hot_length]
        with floats indicating the next note event.
    labels: Tensor of shape [batch_size, examples_per_sequence] with int64s
        indicating the prediction for next note event given the notes up to this
        point in the inputs sequence.
    lengths: Tensor vector of shape [batch_size] with the length of the
        SequenceExamples before padding.
  """
  _, _, sequences = input_sequence_example(file_list, hparams)

  length = tf.shape(sequences['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, hparams.one_hot_length), (None,), ()])

  # The number of threads for enqueuing.
  num_threads = 4
  enqueue_ops = [queue.enqueue([sequences['inputs'], sequences['labels'],
                                length])] * num_threads
  tf.train.add_queue_runner(tf.queue_runner.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(hparams.batch_size)


def make_cell(hparams):
  """Instantiates an RNNCell object.

  Will construct an appropriate RNN cell given hyperparameters. This will
  specifically be a stack of LSTM cells. The height of the stack is specified in
  hparams.

  Args:
    hparams: HParams instance containing model hyperparameters.

  Returns:
    RNNCell instance.
  """
  lstm_layers = [
      tf.nn.rnn_cell.LSTMCell(num_units=layer_size, state_is_tuple=False)
      for layer_size in hparams.rnn_layer_sizes
  ]
  multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers, state_is_tuple=False)
  return multi_cell


def log_perplexity_loss(logits, labels):
  """Computes the log-perplexity of the predictions given the labels.

  log-perplexity = -1/N sum{i=1..N}(log(p(x_i)))
      where x_i's are the correct classes given by labels,
      and p(x) is the model's prediction for class x.

  Softmax is applied to logits to obtain probability predictions.

  Both scaled and unscaled log-perplexities are returned (unscaled does not
  divide by N). Unscaled log-perplexity is simply cross entropy. Use cross
  entropy for training loss so that the gradient magnitudes are not affected
  by sequence length. Use log-perplexity to monitor training progress and
  compare models.

  Args:
    logits: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length]. Must be unscaled logits.
      Do not put through softmax! This function applies softmax.
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    cross_entropy: Unscaled average log-perplexityacross minibatch samples,
        which is just the cross entropy loss. Use this loss for backprop.
    log_perplexity: Average log-perplexity across minibatch samples. Use this
        loss for monitoring training progress and comparing models.
  """
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy = tf.reduce_sum(losses)
  log_perplexity = cross_entropy / tf.to_float(tf.size(losses))
  return cross_entropy, log_perplexity


def eval_accuracy(predictions, labels):
  """Evaluates the accuracy of the predictions.

  Checks how often the prediciton with the highest weight is correct on average.

  Args:
    predictions: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length].
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    The precision of the highest weighted predicted class.
  """
  correct_predictions = tf.nn.in_top_k(predictions, labels, k=1)
  return tf.reduce_mean(tf.to_float(correct_predictions))

