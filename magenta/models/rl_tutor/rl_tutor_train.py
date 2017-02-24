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
import sys

import tensorflow as tf

import rl_tutor_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('domain_application', 'smiles',
                           'Application for which you are training model. Can either '
                           'be smiles (for molecule generation) or melody (for music).')
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox/Google/SMILES-Project/output/prioritized_replay_100_all_models/',
                           'Directory where the model will save its'
                           'compositions and checkpoints (midi files)')
tf.app.flags.DEFINE_string('rnn_checkpoint_dir','/home/natasha/Dropbox/Google/SMILES-Project/models/',
                           'Path to directory holding checkpoints for pre-trained rnn'
                           'model. These will be loaded into the NoteRNNLoader class '
                           'object. The directory should contain a train subdirectory')
tf.app.flags.DEFINE_string('rnn_checkpoint_name', 'smiles_rnn_100_350001.ckpt',
                           'Filename of a checkpoint within the '
                           'rnn_checkpoint_dir directory.')
tf.app.flags.DEFINE_float('reward_scaler', 0.35,
                          'The weight placed on music theory rewards')
tf.app.flags.DEFINE_string('algorithm', 'q',
                           'The name of the algorithm to use for training the'
                           'model. Can be q, psi, or g')
tf.app.flags.DEFINE_integer('training_steps', 4000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('exploration_steps', 0,
                            'The number of steps over which the models'
                            'probability of taking a random action (exploring)'
                            'will be annealed from 1.0 to its normal'
                            'exploration probability. Typically about half the'
                            'training_steps')
tf.app.flags.DEFINE_string('exploration_mode', 'egreedy',
                           'Can be either egreedy for epsilon-greedy or '
                           'boltzmann, which will sample from the models'
                           'output distribution to select the next action')
tf.app.flags.DEFINE_integer('output_every_nth', 100000,
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
  output_dir = os.path.join(FLAGS.output_dir, FLAGS.algorithm)
  output_ckpt = FLAGS.algorithm + '.ckpt'
  backup_checkpoint_file = os.path.join(FLAGS.rnn_checkpoint_dir, 
                                        FLAGS.rnn_checkpoint_name)

  if FLAGS.domain_application == 'melody':
    import rl_tuner
    hparams = rl_tutor_ops.default_hparams()

    dqn_hparams = rl_tutor_ops.HParams(random_action_probability=0.1,
                               store_every_nth=1,
                               train_every_nth=5,
                               minibatch_size=32,
                               discount_rate=0.95,
                               max_experience=100000,
                               target_network_update_rate=0.01)

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
    #hparams = rl_tutor_ops.smiles_hparams()

    hparams = rl_tutor_ops.HParams(use_dynamic_rnn=True,
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

    rl_tutor_hparams = rl_tutor_ops.HParams(random_action_probability=0.01,
                                        store_every_nth=1,
                                        train_every_nth=50,
                                        minibatch_size=512,
                                        discount_rate=0.95,
                                        max_experience=500000,
                                        target_network_update_rate=0.01,
                                        initial_learning_rate=0.0001)

    reward_values = rl_tutor_ops.HParams(valid_length_multiplier=0,
                                        valid_length_bonus_cap=0,
                                        invalid_length_multiplier=0,
                                        sa_multiplier=2,
                                        logp_multiplier=3,
                                        ringp_multiplier=5,
                                        qed_multiplier=40,
                                        shortish_seq=-25,
                                        short_seq=-200,
                                        longish_seq=0,
                                        long_seq=0,
                                        data_scalar=1,
                                        any_valid_bonus=5,
                                        any_invalid_penalty=-5,
                                        end_invalid_penalty=0,
                                        end_valid_bonus=600,
                                        repeated_C_penalty=-150)

    print "Trying to load rnn checkpoint from", FLAGS.rnn_checkpoint_dir
    print "Or else:", backup_checkpoint_file
    tf.logging.info('Trying to load rnn checkpoint from %s', FLAGS.rnn_checkpoint_dir)
    sys.stdout.flush()
    rlt = smiles_tutor.SmilesTutor(output_dir,
                                  reward_values=reward_values,
                                  dqn_hparams=rl_tutor_hparams,
                                  reward_scaler=FLAGS.reward_scaler,
                                  save_name = output_ckpt,
                                  output_every_nth=FLAGS.output_every_nth, 
                                  rnn_checkpoint_dir=FLAGS.rnn_checkpoint_dir,
                                  rnn_checkpoint_file=backup_checkpoint_file,
                                  rnn_hparams=hparams,
                                  exploration_mode=FLAGS.exploration_mode,
                                  algorithm=FLAGS.algorithm) 

                                
  tf.logging.info('Will save models, images, and melodies to: %s', rlt.output_dir)

  tf.logging.info('\nTraining...')
  rlt.train(num_steps=FLAGS.training_steps/2, exploration_period=FLAGS.exploration_steps)

  tf.logging.info('\nHalfway through training. Saving model.')
  rlt.save_model_and_figs(FLAGS.algorithm + '-halfway')

  tf.logging.info('\nResuming training...')
  rlt.train(num_steps=FLAGS.training_steps/2, exploration_period=FLAGS.exploration_steps)

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
