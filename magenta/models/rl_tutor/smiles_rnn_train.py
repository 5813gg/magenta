r"""Code to train a SmilesRNN model.

To run this code on your local machine:
$ bazel run magenta/models/rl_tutor:smiles_rnn_train -- \
--training_data_path 'path.tfrecord'
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import tensorflow as tf

from magenta.common import tf_lib

import smiles_rnn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox/Google/SMILES-Project/output/smiles_rnn_512/',
                           'Directory where the model will save its'
                           'compositions and checkpoints (midi files)')
tf.app.flags.DEFINE_string('data_file', '/home/natasha/Dropbox/Google/SMILES-Project/data/250k_drugs_clean.smi',
                           'Filename of a file containing text strings of '
                           'SMILES encodings.')
tf.app.flags.DEFINE_string('vocab_file', '/home/natasha/Dropbox/Google/SMILES-Project/data/zinc_char_list.json',
                           'Filename of a JSON file containing character '
                           'vocabulary for SMILES encodings.')
tf.app.flags.DEFINE_string('pickle_file', '/home/natasha/Dropbox/Google/SMILES-Project/data/smiles.p',
                           'Filename of a pickle file containing pre-processed '
                           'SMILES training batches.')
tf.app.flags.DEFINE_integer('training_steps', 50000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('output_every_nth', 1000,
                            'The number of steps before the model will evaluate'
                            'itself and store a checkpoint')

def main(_):
  print 'Initializing SMILES RNN'
  smiles_params = tf_lib.HParams(use_dynamic_rnn=True,
                                 batch_size=128,
                                 lr=0.0002,
                                 l2_reg=2.5e-5,
                                 clip_norm=5,
                                 initial_learning_rate=0.01,
                                 decay_steps=1000,
                                 decay_rate=0.85,
                                 rnn_layer_sizes=[512],
                                 one_hot_length=35,
                                 exponentially_decay_learning_rate=True)

  srnn = smiles_rnn.SmilesRNN(FLAGS.output_dir, hparams=smiles_params, 
                              load_training_data=True, data_file=FLAGS.data_file, 
                              vocab_file=FLAGS.vocab_file, pickle_file=FLAGS.pickle_file,
                              output_every=FLAGS.output_every_nth)
  print 'Will save models to:', srnn.checkpoint_dir

  print '\nTraining...'
  srnn.train(num_steps=FLAGS.training_steps)

  print '\nFinished training. Saving output figures'
  srnn.plot_training_progress(save_fig=True)

  print '\nFINAL STATS:'
  print '\tFinal training accuracy:', srnn.training_accuracies[-1]
  print '\tFinal training perplexity:', srnn.training_perplexities[-1]
  print '\tFinal validation accuracy:', srnn.val_accuracies[-1]
  print '\tFinal validation perplexity:', srnn.val_perplexities[-1]


if __name__ == '__main__':
  tf.app.run()
