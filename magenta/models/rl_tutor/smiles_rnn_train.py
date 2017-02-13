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
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox (MIT)/Google/SMILES-Project/output/',
                           'Directory where the model will save its'
                           'compositions and checkpoints (midi files)')
tf.app.flags.DEFINE_string('data_file', '/home/natasha/Dropbox (MIT)/Google/SMILES-Project/data/250k_drugs_clean.smi',
                           'Filename of a file containing text strings of '
                           'SMILES encodings.')
tf.app.flags.DEFINE_string('vocab_file', '/home/natasha/Dropbox (MIT)/Google/SMILES-Project/data/zinc_char_list.json',
                           'Filename of a JSON file containing character '
                           'vocabulary for SMILES encodings.')
tf.app.flags.DEFINE_string('pickle_file', '/home/natasha/Dropbox (MIT)/Google/SMILES-Project/data/smiles.p',
                           'Filename of a pickle file containing pre-processed '
                           'SMILES training batches.')
tf.app.flags.DEFINE_integer('training_steps', 1000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('output_every_nth', 50000,
                            'The number of steps before the model will evaluate'
                            'itself and store a checkpoint')

checkpoint_dir, graph=None, scope='smiles_rnn', checkpoint_file=None, 
               hparams=None, rnn_type='default', checkpoint_scope='smiles_rnn', 
               load_training_data=False, data_file=SMILES_DATA+'250k_drugs_clean.smi', 
               vocab_file=SMILES_DATA+'zinc_char_list.json', pickle_file=SMILES_DATA+'smiles.p',
               vocab_size=rl_tutor_ops.NUM_CLASSES_SMILE)

def main(_):

  smiles_params = tf_lib.HParams(use_dynamic_rnn=True,
                                 batch_size=128,
                                 lr=0.0002,
                                 l2_reg=2.5e-5,
                                 clip_norm=5,
                                 initial_learning_rate=0.01,
                                 decay_steps=1000,
                                 decay_rate=0.85,
                                 rnn_layer_sizes=[100],
                                 one_hot_length=35,
                                 exponentially_decay_learning_rate=True)

  srnn = smiles_rnn.SmilesRNN(output_dir, hparams=smiles_params, 
                              load_training_data=True, data_file=FLAGS.data_file, 
                              vocab_file=FLAGS.vocab_file, pickle_file=FLAGS.pickle_file,
                              output_every=FLAGS.output_every_nth):
  tf.logging.info('Saving models to: %s', srnn.output_dir)

  tf.logging.info('\nTraining...')
  srnn.train(num_steps=FLAGS.training_steps)

  tf.logging.info('\nFinished training. Saving output figures and composition.')
  rlt.plot_rewards(image_name='Rewards-' + FLAGS.algorithm + '.eps')

  rlt.generate_sample(visualize_probs=True, title='trained_smiles_rnn',
                                 prob_image_name='trained_smiles_rnn.png')

  rlt.save_model_and_figs(FLAGS.algorithm)


if __name__ == '__main__':
  tf.app.run()
