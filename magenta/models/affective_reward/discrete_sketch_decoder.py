"""Absolute, discrete sketch RNN model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os

import numpy as np
import tensorflow as tf

import svgwrite
from IPython.display import SVG, display

tf.logging.set_verbosity(tf.logging.INFO)


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())

def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
                     num_steps=250000, # train model for this number of steps steps.
                     max_seq_len=100,
                     rnn_size=2048,    # number of hidden units
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     learning_rate=0.001,
                     decay_rate=0.9999,
                     min_learning_rate=0.00005,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=1,
                     recurrent_dropout_prob=0.80,
                     use_input_dropout=1,
                     input_dropout_prob=0.80,
                     use_output_dropout=1,
                     output_dropout_prob=0.80,
                     label_smoothing_stdev=0.0,
                     vocab_size=256,
                     is_training=1,
                     relative_coords=False,
                     mu_law_mu=255,
  )
  return hparams

def get_hps_set(inference_mode=False, vocab_size=256, relative_coords=False):
  """Loads model for inference mode, used in jupyter notebook."""
  model_params = get_default_hparams()
  model_params.vocab_size = vocab_size
  model_params.relative_coords = relative_coords
  tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

  eval_model_params = copy_hparams(model_params)

  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 1

  if inference_mode:
    eval_model_params.batch_size = 1
    eval_model_params.is_training = 0

  sample_model_params = copy_hparams(eval_model_params)
  sample_model_params.batch_size = 1  # only sample one at a time
  sample_model_params.max_seq_len = 1  # sample one point at a time
  sample_model_params.is_training=0

  return [model_params, eval_model_params, sample_model_params]

def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def get_max_len(sketches):
  max_len = 0
  for sketch in sketches:
    length = len(sketch)
    if length > max_len:
      max_len = length
  return max_len

def draw_lines(data, vocab_size=256, num_drawings_horiz=1,
               svg_filename = '/tmp/sketch_discrete_decoder/svg/sample.svg',
               show_drawing=True, shrink_factor=1.0):
  factor = 256.0 / vocab_size * shrink_factor
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  dims = (vocab_size*factor*num_drawings_horiz, vocab_size*factor)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  p = ""
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "M"
    elif (command != "l"):
      command = "L"
    else:
      command = ""
    x = float(data[i,0])*factor
    y = float(data[i,1])*factor
    if data[i, 2] == 2:
      break
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()

  svg_str = dwg.tostring()
  if show_drawing:
    display(SVG(svg_str))
  return svg_str

def draw_strokes_relative(data, factor=0.5,
                          svg_filename = '/tmp/sketch_rnn/svg/sample.svg',
                          show_drawing=True):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  svg_str = dwg.tostring()
  if show_drawing:
    display(SVG(svg_str))
  return svg_str

def make_grid_svg(s_list, spacing_x=10):
  current_start_x = 0
  all_strokes = s_list[0]
  if all_strokes[-1,2] == 2:
    all_strokes[-1,2] = 1

  for i in range(1,len(s_list)):
    last_x = s_list[i-1][:,0]
    last_max = last_x.max(axis=0)
    current_start_x += last_max + spacing_x

    strokes = np.array(s_list[i], dtype=np.int32)
    strokes[:,0] += current_start_x
    if i < len(s_list)-1 and strokes[-1,2] == 2:
      strokes[-1,2] = 1

    all_strokes = np.concatenate((all_strokes,strokes))

  return np.array(all_strokes)


def make_grid_svg_relative(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)


def show_image(img):
  plt.imshow(1-img.reshape(64, 64), cmap='gray')
  plt.show()

def gaussian_label_smoothing(labels, variance=3.0, vocab_size=256):
  """labels: Tensor of size [batch_size, ?]
     vocab_size: Tensor representing the size of the vocabulary.
     variance: the variance to the gaussian distribution.
  """
  # Gaussian label smoothing
  labels = tf.cast(labels, tf.float32)

  normal_dist = tf.distributions.Normal(loc=labels, scale=variance)
  # Locations to evaluate the probability distributions.
  soft_targets = normal_dist.prob(tf.cast(tf.range(vocab_size), tf.float32)[:, None, None])
  # Reordering soft_targets from [vocab_size, batch_size, ?] to match
  # logits: [batch_size, ?, vocab_size]
  soft_targets = tf.transpose(soft_targets, perm=[1, 2, 0])
  norm_factor = tf.expand_dims(tf.reduce_sum(soft_targets, axis=2), -1)
  soft_targets = tf.divide(soft_targets, norm_factor)
  return soft_targets

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

# Discrete SketchRNN Decoder
class DiscreteSketchRNNDecoder():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    with tf.variable_scope('discrete_sketch_rnn_decoder', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu 0")
        with tf.device("/gpu:0"):
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
    self.init_session()
  def build_model(self, hps):

    LENGTH = self.hps.max_seq_len # 1000 timesteps

    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell # use LayerNormLSTM

    use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True
    is_training = False if self.hps.is_training == 0 else True
    use_layer_norm = False if self.hps.use_layer_norm == 0 else True

    if use_recurrent_dropout:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
    else:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)

    # multi-layer, and dropout:
    print("input dropout mode =", use_input_dropout)
    print("output dropout mode =", use_output_dropout)
    print("recurrent dropout mode =", use_recurrent_dropout)
    if use_input_dropout:
      print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence = tf.placeholder(dtype=tf.uint8, shape=[self.hps.batch_size, self.hps.max_seq_len+1, 3])

    self.x_label = self.sequence[:, :, 0]
    self.y_label = self.sequence[:, :, 1]
    self.s_label = self.sequence[:, :, 2] # pen state

    self.x = tf.one_hot(self.x_label, self.hps.vocab_size)
    self.y = tf.one_hot(self.y_label, self.hps.vocab_size)
    self.s = tf.one_hot(self.s_label, 3)

    if hps.label_smoothing_stdev > 0:
      self.smoothed_x = gaussian_label_smoothing(self.x_label, variance=hps.label_smoothing_stdev)
      self.smoothed_y = gaussian_label_smoothing(self.y_label, variance=hps.label_smoothing_stdev)
    else:
      self.smoothed_x = self.x
      self.smoothed_y = self.y

    self.input_x = self.x[:, :-1, :] # the last element is not part of the input
    self.input_y = self.y[:, :-1, :]
    self.input_s = self.s[:, :-1, :]

    self.output_x = self.smoothed_x[:, 1:, :]
    self.output_y = self.smoothed_y[:, 1:, :]
    self.output_s = self.s[:, 1:, :]

    self.input_seq = tf.concat([self.input_x, self.input_y, self.input_s], axis=2)

    actual_input_x = self.input_seq
    self.initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)

    NOUT = self.hps.vocab_size*2+3 # x, y, and pen states

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    output, last_state = tf.nn.dynamic_rnn(cell, actual_input_x, initial_state=self.initial_state,
                                           time_major=False, swap_memory=True, dtype=tf.float32, scope="RNN")

    output = tf.reshape(output, [-1, hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, self.hps.max_seq_len, NOUT])
    self.final_state = last_state

    self.logits_x = output[:, :, :self.hps.vocab_size]
    self.logits_y = output[:, :, self.hps.vocab_size:2*self.hps.vocab_size]
    self.logits_s = output[:, :, 2*self.hps.vocab_size:]

    self.predict_x = tf.nn.softmax(self.logits_x)
    self.predict_y = tf.nn.softmax(self.logits_y)
    self.predict_s = tf.nn.softmax(self.logits_s)

    self.cost_x = tf.nn.softmax_cross_entropy_with_logits(labels=self.output_x, logits=self.logits_x)
    self.cost_y = tf.nn.softmax_cross_entropy_with_logits(labels=self.output_y, logits=self.logits_y)

    self.cost_s = tf.nn.softmax_cross_entropy_with_logits(labels=self.output_s, logits=self.logits_s)

    # mask the errors of x and y up to the point before the end of sketch.
    fs = 1.0 - self.output_s[:, :, 2] # use training data for this
    self.cost_x = tf.multiply(self.cost_x, fs)
    self.cost_y = tf.multiply(self.cost_y, fs)

    self.cost_x = tf.reduce_mean(tf.reduce_sum(self.cost_x, axis=1))
    self.cost_y = tf.reduce_mean(tf.reduce_sum(self.cost_y, axis=1))
    self.cost_s = tf.reduce_mean(tf.reduce_sum(self.cost_s, axis=1))

    self.cost = self.cost_x + self.cost_y + self.cost_s

    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      gvs = optimizer.compute_gradients(self.cost)
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    # initialize vars
    self.init = tf.global_variables_initializer()

  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)

  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()

  def save_model(self, model_save_path, global_step):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'discrete_sketch_decoder')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, global_step=global_step) # just keep one

  def load_checkpoint(self, checkpoint_path, checkpoint_file=None):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    if not checkpoint_file:
      ckpt = tf.train.get_checkpoint_state(checkpoint_path)
      checkpoint_file = ckpt.model_checkpoint_path
    print('loading model', checkpoint_file)
    tf.logging.info('Loading model %s.', checkpoint_file)
    saver.restore(sess, checkpoint_file)


class DiscreteDataLoader:
  def __init__(self, filepath, hps_model, use_remote_data=True):
    self.filepath = filepath
    self.use_remote_data = use_remote_data # TODO: check
    self.hps = hps_model
    self.vocab_factor = 256.0 / self.hps.vocab_size
    if self.vocab_factor != 1.0:
      print('Altering the resolution of the output by a factor of %',
            self.vocab_factor)

    self.load_data()

  def load_data(self):
    if self.use_remote_data:
      tf.logging.info('Using remote data, path is: %s', self.filepath)
      input_file = open(self.filepath, 'r') # TODO: check
      data = np.load(input_file)
      tf.logging.info('Read remote data')
    else:
      data = np.load(self.filepath)

    train_strokes = data['train']
    valid_strokes = data['valid']
    test_strokes = data['test']

    self.max_len = self.hps.max_seq_len # define maximum length of our dataset to be this.

    self.sketches3, self.sketch_data, self.num_sketches = convert_strokes3_dataset(
        train_strokes, self.max_len, self.hps.vocab_size,
        self.hps.relative_coords, self.hps.mu_law_mu)

  def random_batch(self):
     # due to a bug, used to say: idx = np.random.permutation(1000)[0:self.hps.batch_size]
    idx = np.random.choice(np.arange(len(self.sketch_data)), self.hps.batch_size)
    return self.sketch_data[idx]

  def convert_strokes3_dataset(strokes, max_len, vocab_size, relative=False,
                             mu=255):
  sketches3 = []
  for i in range(len(strokes)):
    sketch = strokes[i]
    if len(sketch) <= max_len:
      sketches3.append(sketch)

  num_sketches = len(sketches3)
  print("number of sketches", num_sketches)

  # put each sketch data into a fixed length matrix of length max_len. everything in one np tensor
  if relative:
    sketch_data = np.zeros((num_sketches, max_len+1, 3), dtype=np.int8)
  else:
    sketch_data = np.zeros((num_sketches, max_len+1, 3), dtype=np.uint8)
  sketch_data[:, :, 2] = 2 # default to end of drawing
  sketch_data[:, 0, 2] = 1 # first pen state is an end-of-line
  for i in range(num_sketches):
    sketch = sketches3[i]
    if relative:
      sketch = convert_absolute_strokes_to_relative(sketch)
      sketch[:,0] = [mu_encode(x, new_vocab_size=vocab_size/2.0, mu=mu) for x in sketch[:,0]]
      sketch[:,1] = [mu_encode(y, new_vocab_size=vocab_size/2.0, mu=mu) for y in sketch[:,1]]
    sketch_data[i, 1:len(sketch)+1, :] = sketch

  if not relative:
    vocab_factor = 256.0 / vocab_size
    sketch_data[:, :, 0:2] = (sketch_data[:, :, 0:2].astype(
        np.float)/vocab_factor).astype(np.uint8)
  else:
    # Ensure all the data is > 0 before it is fed to the model
    sketch_data[:, :, 0:2] = sketch_data[:, :, 0:2] + vocab_size/2.0

  return sketches3, sketch_data, num_sketches

def convert_absolute_strokes_to_relative(abs_strokes):
  rel_strokes = np.zeros(np.shape(abs_strokes))
  rel_strokes[:,2] = abs_strokes[:,2]
  rel_strokes[1:,:2] = abs_strokes[1:, 0:2].astype(float) - abs_strokes[:-1, 0:2]
  rel_strokes[:,0:2] = rel_strokes[:,0:2] / 2
  rel_strokes = np.around(rel_strokes)
  return rel_strokes

def decode_relative_strokes(strokes, hps):
  strokes = np.copy(strokes)
  strokes[:, 0:2] = strokes[:, 0:2] - hps.vocab_size/2.0
  strokes[:,0] = [mu_decode(x, new_vocab_size=hps.vocab_size/2.0,
                            mu=hps.mu_law_mu) for x in strokes[:,0]]
  strokes[:,1] = [mu_decode(y,  new_vocab_size=hps.vocab_size/2.0,
                            mu=hps.mu_law_mu) for y in strokes[:,1]]
  return strokes

def mu_encode(x, orig_vocab_size=256, new_vocab_size=32, mu=255):
  x = float(x) / orig_vocab_size
  mu_x = np.sign(x) * np.log(1 + mu*np.abs(x)) / np.log(1+mu)
  return np.round(mu_x * new_vocab_size)

def mu_decode(y, orig_vocab_size=256, new_vocab_size=32, mu=255):
  y = float(y) / new_vocab_size
  exp = (1+mu)**np.abs(y)
  mu_y = np.sign(y) *(exp-1) / mu
  return np.round(mu_y * orig_vocab_size)

# functions for sampling sketches:
def sample(p):
  return np.argmax(np.random.multinomial(1, p)) # for some reason get_pi_idx works better.

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

def sample_sketch(model, max_seq_len=300, bias=0.5, greedy=False,
                  relative=False):
  # bias is kind of like a temperature parameter.
  prev_state = model.sess.run(model.initial_state)
  sketch = [[0, 0, 1]]
  for step in range(max_seq_len):
    prev_point = sketch[-1]
    sequence = np.array([prev_point, [0, 0, 0]]).reshape(1, 2, 3) # need to put a dummy point to be ignored
    feed = {model.sequence: sequence, model.initial_state:prev_state}
    (logits_x, logits_y, logits_s, next_state) = model.sess.run([model.logits_x, model.logits_y, model.logits_s, model.final_state], feed)
    logits_x = logits_x[0][0]
    logits_y = logits_y[0][0]
    logits_s = logits_s[0][0]
    if greedy:
      next_x = np.argmax(logits_x)
      next_y = np.argmax(logits_y)
      next_s = np.argmax(logits_s)
    else:
      pdf_x = softmax(logits_x*(1.+bias))
      pdf_y = softmax(logits_y*(1.+bias))
      pdf_s = softmax(logits_s*(1.+bias))
      next_x = get_pi_idx(np.random.rand(), pdf_x)
      next_y = get_pi_idx(np.random.rand(), pdf_y)
      next_s = get_pi_idx(np.random.rand(), pdf_s)
    sketch.append([next_x, next_y, next_s])
    prev_state = next_state
    if next_s == 2 or step > max_seq_len:
      break
  if relative:
    return np.array(sketch, dtype=np.int8)
  else:
    return np.array(sketch, dtype=np.uint8)
