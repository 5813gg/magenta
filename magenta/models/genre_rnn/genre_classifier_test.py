r"""Tests for GenreClassifier.

To run this code:
$ bazel test genre_rnn:genre_classifier_test
"""

import os

from ....genre_rnn import genre_classifier

FLAGS = flags.FLAGS


class GenreClassifierTest(magic_test_thingy):

  def setUp(self):
    self.training_dir = os.path.join(
        FLAGS.test_srcdir, 'testdata',
        'test_genre_melodies.tfrecord')
    self.output_dir = os.path.join(FLAGS.test_tmpdir, 'genre_classifier_test')
    self.output_tfrecord = self.output_dir + 'test.tfrecord'
    self.hparams = genre_classifier.default_hparams()

  def testInitialization(self):
    gclassifier = genre_classifier.GenreClassifier(
        hparams=self.hparams, training_data_path=self.training_dir,
        output_dir=self.output_dir)

    self.assertTrue(gclassifier.graph is not None)
    self.assertTrue(gclassifier.gradients is not None)

  def testTraining(self):
    gclassifier = genre_classifier.GenreClassifier(
        hparams=self.hparams, training_data_path=self.training_dir,
        output_dir=self.output_dir, num_training_steps=5, summary_frequency=5)

    for step in gclassifier.training_loop():
      self.assertTrue(isinstance(step, dict))
      self.assertTrue(step['cost'] > 0)

    checkpoint_loc = os.path.join(self.output_dir, 'model.ckpt-5')
    self.assertTrue(os.path.exists(checkpoint_loc))


if __name__ == '__main__':
  magic_test_thingy.main()
