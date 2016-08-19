r"""Tests for MelodyQNetwork and by proxy MelodyRNN.

To run this code:
$ bazel test genre_rnn:genre_dqn_test
"""

import os

import numpy as np

from ....genre_rnn import genre_dqn

FLAGS = flags.FLAGS


class GenreDQNTest(magic_test_thingy):

  def setUp(self):
    self.output_dir = os.path.join(FLAGS.test_tmpdir, 'genre_dqn_test')
    self.base_dir = os.path.join(
        FLAGS.test_srcdir,
        '.../genre_rnn/')
    self.checkpoint_dir = self.base_dir + 'testdata/'
    self.classifier_checkpoint = self.checkpoint_dir + 'genre_classifier.ckpt'
    self.genre_rnn_checkpoint = self.checkpoint_dir + 'genre_rnn.ckpt'
    self.midi_primer = self.base_dir + 'testdata/primer.mid'

  def testDataAvailable(self):
    self.assertTrue(os.path.exists(self.classifier_checkpoint))
    self.assertTrue(os.path.exists(self.genre_rnn_checkpoint))
    self.assertTrue(os.path.exists(self.midi_primer))

  def testInitialization(self):
    gdqn = genre_dqn.GenreDQN(
        '', '', self.output_dir, midi_primer=self.midi_primer,
        genre_rnn_backup_checkpoint_file=self.genre_rnn_checkpoint,
        genre_classifier_backup_checkpoint_file=self.classifier_checkpoint)

    self.assertTrue(gdqn.q_network is not None)
    self.assertTrue(gdqn.genre_classifier is not None)

  def testRestoringQNetworks(self):
    gdqn = genre_dqn.GenreDQN(
        '', '', self.output_dir, midi_primer=self.midi_primer,
        genre_rnn_backup_checkpoint_file=self.genre_rnn_checkpoint,
        genre_classifier_backup_checkpoint_file=self.classifier_checkpoint)

    q_vars = gdqn.q_network.variables()
    target_q_vars = gdqn.target_q_network.variables()

    q1 = gdqn.session.run(q_vars[0])
    tq1 = gdqn.session.run(target_q_vars[0])

    self.assertTrue(np.sum((q1 - tq1)**2) == 0)

  def testInitialGeneration(self):
    gdqn = genre_dqn.GenreDQN(
        '', '', self.output_dir, midi_primer=self.midi_primer,
        genre_rnn_backup_checkpoint_file=self.genre_rnn_checkpoint,
        genre_classifier_backup_checkpoint_file=self.classifier_checkpoint)

    plot_name = 'test_initial_pop.png'
    gdqn.generate_genre_music_sequence('pop', visualize_probs=True,
                                       prob_image_name=plot_name)
    output_path = os.path.join(self.output_dir, plot_name)
    self.assertTrue(os.path.exists(output_path))

  def testAction(self):
    gdqn = genre_dqn.GenreDQN(
        '', '', self.output_dir, midi_primer=self.midi_primer,
        genre_rnn_backup_checkpoint_file=self.genre_rnn_checkpoint,
        genre_classifier_backup_checkpoint_file=self.classifier_checkpoint)

    initial_note = gdqn.prime_q_model()

    action = gdqn.action(initial_note, 100, enable_random=False)
    self.assertTrue(action is not None)

  def testTraining(self):
    gdqn = genre_dqn.GenreDQN(
        '', '', self.output_dir, midi_primer=self.midi_primer,
        output_every_nth=30,
        genre_rnn_backup_checkpoint_file=self.genre_rnn_checkpoint,
        genre_classifier_backup_checkpoint_file=self.classifier_checkpoint)
    gdqn.train(num_steps=31, exploration_period=3)

    self.assertTrue(len(gdqn.composition) == 31)
    self.assertTrue(os.path.exists(gdqn.log_dir + '-30'))
    self.assertTrue(len(gdqn.rewards_batched) >= 1)


if __name__ == '__main__':
  magic_test_thingy.main()
