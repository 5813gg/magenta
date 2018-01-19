"""SketchRNN training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import urllib
import zipfile

import numpy as np
import requests
import six
import tensorflow as tf

import discrete_sketch_decoder as dsd

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'TODO/path/sketch_rnn/data/',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'image_class', 'face',
    'The sketch class to on which to train.')
tf.app.flags.DEFINE_string(
    'log_root', '/tmp/discrete_decoder_32bit',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')
tf.app.flags.DEFINE_integer(
    'save_every', 1000,
    'The number of iterations between saving the model.')
tf.app.flags.DEFINE_string(
    'master', '',
    'dummy to fix weird probs')
tf.app.flags.DEFINE_integer(
    'vocab_size', 256,
    'The number of outputs that will be mapped to the 256 image pixels. If less'
    'than 256, will get a lower resolution version.')
tf.app.flags.DEFINE_boolean(
    'relative_coords', True,
    'Set to true if using relative coordinates rather than absolute.')


def train(datapath, image_class, log_root, load_checkpoint=False,
          output_every=20, save_every=20):
  checkpoint_path = log_root + image_class + '/train/'

  hps, hps_eval, hps_sample = dsd.get_hps_set(vocab_size=FLAGS.vocab_size,
                                              relative_coords=FLAGS.relative_coords)
  if hps.relative_coords:
    tf.logging.info('Using relative coordinates')

  # Write config file to json file.
  tf.gfile.MakeDirs(checkpoint_path)
  with tf.gfile.Open(
      os.path.join(checkpoint_path, 'model_config.json'), 'w') as f:
    json.dump(hps.values(), f, indent=True)
  tf.logging.info('Log directory created at %s', checkpoint_path)

  data_file = datapath + image_class + '.simple_line.npz'
  loader = dsd.DiscreteDataLoader(data_file, hps)

  dsd.reset_graph()
  model = dsd.DiscreteSketchRNNDecoder(hps)
  eval_model = dsd.DiscreteSketchRNNDecoder(hps_eval, reuse=True)
  sample_model = dsd.DiscreteSketchRNNDecoder(hps_sample, reuse=True)

  if load_checkpoint:
    model.load_checkpoint(checkpoint_path)

  start = time.time()
  for local_step in range(hps.num_steps):

    step = model.sess.run(model.global_step)
    curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

    sequence = loader.random_batch()

    feed = {model.sequence: sequence, model.lr: curr_learning_rate}
    (train_cost, state, train_step, _) = model.sess.run([model.cost, model.final_state, model.global_step, model.train_op], feed)
    if (step % output_every==0 and step > 0):
      end = time.time()
      time_taken = end-start
      start = time.time()
      output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
      tf.logging.info(output_log)
    if (step % save_every == 0 and step > 0):
      tf.logging.info('saving')
      model.save_model(checkpoint_path, train_step)


def main(unused_argv):
  """Load model params, save config file and start trainer."""
  train(FLAGS.data_dir, FLAGS.image_class, FLAGS.log_root,
        FLAGS.resume_training, save_every=FLAGS.save_every)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
