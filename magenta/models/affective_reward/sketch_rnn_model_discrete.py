"""A discrete version of the Sketch-RNN VAE Model.

It's still a VAE, but the outputs are not a MDN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os

import numpy as np
import tensorflow as tf

import svgwrite
from IPython.display import SVG, display

import sketch_rnn
import sketch_rnn_utils

tf.logging.set_verbosity(tf.logging.INFO)

def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      data_set=['cat.simple_line.npz'],  # Our dataset.
      num_steps=100000,  # Total number of steps of training. Keep large.
      save_every=500,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=1024,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=512,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=1.0,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.9990,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.00,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True,  # Is model training? Recommend keeping true.
      label_smoothing_variance=1.0,
  )
  return hparams

def gaussian_label_smoothing(labels, variance=2.0, vocab_size=256):
  """labels: Tensor of size [batch_size, ?]
    vocab_size: Tensor representing the size of the vocabulary.
    variance: the variance to the gaussian distribution.
  """
  # Gaussian label smoothing
  labels = tf.cast(labels, tf.float32)

  normal_dist = tf.distributions.Normal(loc=labels, scale=variance)
  # Locations to evaluate the probability distributions.
  soft_targets = normal_dist.prob(
      tf.cast(tf.range(vocab_size), tf.float32)[:, None, None])
  # Reordering soft_targets from [vocab_size, batch_size, ?] to match
  # logits: [batch_size, ?, vocab_size]
  soft_targets = tf.transpose(soft_targets, perm=[1, 2, 0])
  return soft_targets

class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)
    tf.logging.info('Finished constructing graph.')

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = sketch_rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    presig = sketch_rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
    return mu, presig

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = sketch_rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = sketch_rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = sketch_rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = sketch_rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = sketch_rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = sketch_rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = self.hps.use_recurrent_dropout
    use_input_dropout = self.hps.use_input_dropout
    use_output_dropout = self.hps.use_output_dropout

    cell = cell_fn(
        hps.dec_rnn_size,
        use_recurrent_dropout=use_recurrent_dropout,
        dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # dropout:
    tf.logging.info('Input dropout mode = %s.', use_input_dropout)
    tf.logging.info('Output dropout mode = %s.', use_output_dropout)
    tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    # stuff for encoder
    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[None]) #self.hps.batch_size
#     self.input_data = tf.placeholder(
#         dtype=tf.float32,
#         shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
#
#     self.output_sequence = self.input_data[:, 1:self.hps.max_seq_len + 1, :]
#     # vectors of strokes to be fed to decoder (same as above, but lagged behind
#     # one step to include initial dummy value of (0, 0, 1, 0, 0))
#     self.input_sequence = self.input_data[:, :self.hps.max_seq_len, :]

    # stuff for decoder
    self.input_data = tf.placeholder(dtype=tf.int32, shape=[None, self.hps.max_seq_len+1, 5]) #self.hps.batch_size
    self.computed_batch_size = tf.shape(self.input_data)[0]

    self.s = tf.cast(self.input_data[:, :, 2:], tf.float32) # pen state # SHOULD TAKE MORE PEN STATES
    self.input_s = self.s[:, :-1, :]
    self.output_s = self.s[:, 1:, :]

    # labels come in as relative so convert them to be above 0
    self.x_label = self.input_data[:, :, 0] + 128
    self.y_label = self.input_data[:, :, 1] + 128

    # ground truth discrete input
    self.x = tf.one_hot(self.x_label, 256)
    self.y = tf.one_hot(self.y_label, 256)

    self.input_x = self.x[:, :-1, :] # the last element is not part of the input
    self.input_y = self.y[:, :-1, :]
    self.output_x = self.x[:, 1:, :]
    self.output_y = self.y[:, 1:, :]

    self.input_seq = tf.concat([self.input_x, self.input_y, self.input_s], axis=2)
    actual_input_x = self.input_seq

    # smoothed output
    self.smoothed_x = gaussian_label_smoothing(self.x_label, variance=hps.label_smoothing_variance)
    self.smoothed_y = gaussian_label_smoothing(self.y_label, variance=hps.label_smoothing_variance)
    self.smoothed_output_x = self.smoothed_x[:, 1:, :]
    self.smoothed_output_y = self.smoothed_y[:, 1:, :]

    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.mean, self.presig = self.encoder(self.input_seq,
                                            self.sequence_lengths)
      self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      eps = tf.random_normal(
          (self.computed_batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
      self.batch_z = self.mean + tf.multiply(self.sigma, eps)
      # KL cost
      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.computed_batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          sketch_rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.computed_batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=computed_batch_size, dtype=tf.float32)

    self.num_mixture = hps.num_mixture

    n_out = 256*2+3 # x, y, and pen states

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.dec_rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, self.hps.max_seq_len, n_out])
    self.final_state = last_state

    self.logits_x = output[:, :, :256]
    self.logits_y = output[:, :, 256:512]
    self.logits_s = output[:, :, 512:]

    self.predict_x = tf.nn.softmax(self.logits_x)
    self.predict_y = tf.nn.softmax(self.logits_y)
    self.predict_s = tf.nn.softmax(self.logits_s)

    self.cost_x = tf.nn.softmax_cross_entropy_with_logits(labels=self.smoothed_output_x, logits=self.logits_x)
    self.cost_y = tf.nn.softmax_cross_entropy_with_logits(labels=self.smoothed_output_y, logits=self.logits_y)
    #self.cost_x = tf.losses.softmax_cross_entropy(labels=self.smoothed_output_x, logits=self.logits_x, label_smoothing=True)
    #self.cost_y = tf.losses.softmax_cross_entropy(labels=self.smoothed_output_y, logits=self.logits_y, label_smoothing=True)
    self.cost_s = tf.nn.softmax_cross_entropy_with_logits(labels=self.output_s, logits=self.logits_s)

    # mask the errors of x and y up to the point before the end of sketch.
    fs = 1.0 - self.output_s[:, :, 2] # use training data for this_params
    self.cost_x = tf.multiply(self.cost_x, fs)
    self.cost_y = tf.multiply(self.cost_y, fs)

    self.cost_x = tf.reduce_mean(tf.reduce_sum(self.cost_x, axis=1))
    self.cost_y = tf.reduce_mean(tf.reduce_sum(self.cost_y, axis=1))
    self.cost_s = tf.reduce_mean(tf.reduce_sum(self.cost_s, axis=1))

    self.r_cost = self.cost_x + self.cost_y + self.cost_s

    self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
    self.cost = self.r_cost + self.kl_cost * self.kl_weight


    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      gvs = optimizer.compute_gradients(self.cost)
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')


def encode_strokes(sess, eval_model, input_strokes, draw_debug=True,
                   absolute=False):
  strokes = sketch_rnn_utils.to_big_strokes(input_strokes).tolist()
  seq_len = [len(input_strokes)]
  if draw_debug:
    if absolute:
      draw_lines_absolute(sketch_rnn_utils.to_normal_strokes(np.array(strokes)))
    else:
      draw_strokes_relative(sketch_rnn_utils.to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z,
                          feed_dict={eval_model.input_data: [strokes],
                                     eval_model.sequence_lengths: seq_len})[0]


def sample(sess, model, seq_len=250, greedy_mode=True,
           z=None):
  """Samples a sequence from a pre-trained model."""

  prev_point = np.zeros((1, 2, 5), dtype=np.float32)
  prev_point[0, 0, 3] = 1  # set lift pen to true so that it starts its own stroke
  if z is None:
    z = np.random.randn(1, model.hps.z_size)  # not used if unconditional
  else:
    z = np.atleast_2d(z)

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  strokes = np.zeros((seq_len, 5), dtype=np.float32)

  for step in range(seq_len):
    if not model.hps.conditional:
      feed = {model.input_data: prev_point,
              model.initial_state: prev_state,
              model.sequence_lengths: [1]}
    else:
      feed = {model.input_data: prev_point,
              model.initial_state: prev_state,
              model.sequence_lengths: [1],
              model.batch_z: z}
    (logits_x, logits_y, logits_s, next_state) = sess.run([model.predict_x, model.predict_y, model.predict_s, model.final_state], feed)

    logits_x = np.reshape(logits_x, [-1])
    logits_y = np.reshape(logits_y, [-1])
    logits_s = np.reshape(logits_s, [-1])

    if greedy_mode:
      next_x = np.argmax(logits_x)
      next_y = np.argmax(logits_y)
      next_s_idx = np.argmax(logits_s)
    else:
      next_x = np.random.choice(256, p=logits_x)
      next_y = np.random.choice(256, p=logits_y)
      next_s_idx = np.random.choice(3, p=logits_s)

    next_s = [0, 0, 0]
    next_s[next_s_idx] = 1

    # convert softmax positions into real relative coords
    coords = [next_x-128, next_y-128]
    coords.extend(next_s)

    strokes[step,:] = coords

    prev_state = next_state
    prev_point = np.zeros((1,2, 5), dtype=np.float32)
    prev_point[:,0,:] = coords

    if next_s_idx == 2 or step > seq_len:
      break

  strokes3 = sketch_rnn_utils.to_normal_strokes(np.array(strokes, dtype=np.uint8))
  return strokes3

def draw_lines_absolute(data, factor=1.0, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  dims = (256*factor, 256*factor)
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
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

def draw_strokes_relative(data, factor=0.5,
                          svg_filename = '/tmp/sketch_rnn/svg/sample.svg',
                          show_drawing=True):
  if np.min(data) >= 0:
    print("Subtracting 128 from data to get relative strokes")
    data[:,0:2] = data[:,0:2] - 128
    print("data now looks like...")
    print(data[0:10])
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = sketch_rnn_utils.get_bounds(data, factor)
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
