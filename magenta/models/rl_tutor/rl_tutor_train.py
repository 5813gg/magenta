r"""Code to train a MelodyQ model.

To run this code on your local machine:
$ bazel run magenta/models/rl_tutor:rl_tutor_train -- \
--note_rnn_checkpoint_dir 'path' --midi_primer 'primer.mid' \
--training_data_path 'path.tfrecord'
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import tensorflow as tf

import rl_tutor_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('domain_application', 'smiles',
                           'Application for which you are training model. Can either '
                           'be smiles (for molecule generation) or melody (for music).')
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox/Google/SMILES-Project/output/round4/',
                           'Directory where the model will save its'
                           'compositions and checkpoints (midi files)')
tf.app.flags.DEFINE_string('rnn_checkpoint_dir', '/home/natasha/Dropbox/Google/SMILES-Project/models/',
                           'Path to directory holding checkpoints for pre-trained rnn'
                           'model. These will be loaded into the NoteRNNLoader class '
                           'object. The directory should contain a train subdirectory')
tf.app.flags.DEFINE_string('rnn_checkpoint_name', 'smiles_rnn_100_350001.ckpt',
                           'Filename of a checkpoint within the '
                           'rnn_checkpoint_dir directory.')
tf.app.flags.DEFINE_float('reward_scaler', 1,
                          'The weight placed on music theory rewards')
tf.app.flags.DEFINE_string('algorithm', 'psi',
                           'The name of the algorithm to use for training the'
                           'model. Can be q, psi, or g')
tf.app.flags.DEFINE_integer('training_steps', 1000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('exploration_steps', 500000,
                            'The number of steps over which the models'
                            'probability of taking a random action (exploring)'
                            'will be annealed from 1.0 to its normal'
                            'exploration probability. Typically about half the'
                            'training_steps')
tf.app.flags.DEFINE_string('exploration_mode', 'boltzmann',
                           'Can be either egreedy for epsilon-greedy or '
                           'boltzmann, which will sample from the models'
                           'output distribution to select the next action')
tf.app.flags.DEFINE_integer('output_every_nth', 50000,
                            'The number of steps before the model will evaluate'
                            'itself and store a checkpoint')
tf.app.flags.DEFINE_string('training_data_path', '',
                           'Directory where the model will get melody training'
                           'examples')
tf.app.flags.DEFINE_integer('num_notes_in_melody', 32,
                            'The number of notes in each composition')
tf.app.flags.DEFINE_string('midi_primer', '/home/natasha/Dropbox/Google/code/testdata/primer.mid',
                           'A midi file that can be used to prime the model')



def main(_):
  dqn_hparams = rl_tutor_ops.HParams(random_action_probability=0.1,
                               store_every_nth=1,
                               train_every_nth=5,
                               minibatch_size=32,
                               discount_rate=0.95,
                               max_experience=100000,
                               target_network_update_rate=0.01)

  output_dir = os.path.join(FLAGS.output_dir, FLAGS.algorithm)
  output_ckpt = FLAGS.algorithm + '.ckpt'
  backup_checkpoint_file = os.path.join(FLAGS.rnn_checkpoint_dir, 
                                        FLAGS.rnn_checkpoint_name)

  if FLAGS.domain_application == 'melody':
    import rl_tuner
    hparams = rl_tutor_ops.default_hparams()

    rlt = rl_tuner.RLTuner(output_dir,
                          midi_primer=FLAGS.midi_primer, 
                          dqn_hparams=dqn_hparams, 
                          reward_scaler=FLAGS.reward_scaler,
                          save_name = output_ckpt,
                          output_every_nth=FLAGS.output_every_nth, 
                          note_rnn_checkpoint_dir=FLAGS.rnn_checkpoint_dir,
                          note_rnn_checkpoint_file=backup_checkpoint_file,
                          note_rnn_hparams=hparams, 
                          num_notes_in_melody=FLAGS.num_notes_in_melody,
                          exploration_mode=FLAGS.exploration_mode,
                          algorithm=FLAGS.algorithm)
  elif FLAGS.domain_application == 'smiles':
    import smiles_tutor
    hparams = rl_tutor_ops.smiles_hparams()

    rlt = smiles_tutor.SmilesTutor(output_dir,
                                  dqn_hparams=dqn_hparams, 
                                  reward_scaler=FLAGS.reward_scaler,
                                  save_name = output_ckpt,
                                  output_every_nth=FLAGS.output_every_nth, 
                                  rnn_checkpoint_dir=FLAGS.rnn_checkpoint_dir,
                                  rnn_checkpoint_file=backup_checkpoint_file,
                                  rnn_hparams=hparams,
                                  exploration_mode=FLAGS.exploration_mode,
                                  algorithm=FLAGS.algorithm) 

  tf.logging.info('Saving images and melodies to: %s', rlt.output_dir)

  tf.logging.info('\nTraining...')
  rlt.train(num_steps=FLAGS.training_steps,
               exploration_period=FLAGS.exploration_steps)

  tf.logging.info('\nFinished training. Saving output figures and renders.')
  rlt.plot_rewards(image_name='Rewards-' + FLAGS.algorithm + '.eps')

  rlt.generate_sample(visualize_probs=True, title=FLAGS.algorithm,
                                 prob_image_name=FLAGS.algorithm + '.png')

  rlt.save_model_and_figs(FLAGS.algorithm)

  tf.logging.info('\nCalculating domain metric stats for 1000 '
                  'compositions')
  rlt.evaluate_domain_metrics(num_compositions=1000)


if __name__ == '__main__':
  tf.app.run()
