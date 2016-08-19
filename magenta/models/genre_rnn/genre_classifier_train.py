r"""Defines a class and operations for the GenreRNN model.

------run locally---------------

$ bazel build genre_rnn:genre_classifier_train

$ bazel-bin/.../genre_rnn/genre_classifier_train --alsologtostderr

"""
import tensorflow as tf

from ....genre_rnn import genre_classifier
from ....genre_rnn import genre_ops

FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('master', 'local',
                           'Name of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('ps_tasks', 0,
                            'Number of tasks in the ps job. If 0 no ps job is '
                            'used.')
tf.app.flags.DEFINE_string('sequence_examples', 'something.tfrecord',
                           'Path to tfrecord file(s) containing '
                           'tf.SequenceExample records for training. Use "*" '
                           'wildcards to match multiple files.')
tf.app.flags.DEFINE_string('experiment_dir', '/tmp/genre_classifier',
                           'Path to directory holding runs for this '
                           'experiment. An experiment is typically one '
                           'combination of hyperparameters. A new subdirectory '
                           'will be created for every training run, and '
                           'train/eval directories will be created within '
                           'runs. TensorBoard event files and model '
                           'checkpoints are saved for each run.')
tf.app.flags.DEFINE_string('hparams', '',
                           'Comma separated list of name=value pairs. For '
                           'example, "batch_size=64,rnn_layer_sizes=[100,100],'
                           'use_dynamic_rnn=". To set something False, just '
                           'set it to the empty string: "use_dynamic_rnn=".')
tf.app.flags.DEFINE_integer('num_training_steps', 30000,
                            'The the number of training steps to take in this '
                            'training session.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps of training.')
tf.app.flags.DEFINE_integer('steps_to_average', 20,
                            'Accuracy averaged over the last '
                            '`steps_to_average` steps is reported.')


def train_genre_classifier(output_dir, num_iters=None):
  """Trains a GenreClassifier and saves the results in output_dir.

  Args:
    output_dir: Directory in which the model checkpoints will be saved.
    num_iters: The number of training steps taken to train the model.
  """
  run_dir = genre_ops.get_next_run_dir(output_dir)
  train_dir = run_dir + '/train'
  make_dirs(train_dir)

  hparams = genre_classifier.default_hparams()
  if num_iters is not None:
    gclassifier = genre_classifier.GenreClassifier(
        hparams=hparams,
        training_data_path=FLAGS.sequence_examples,
        output_dir=train_dir,
        tf_master=FLAGS.master,
        task_id=FLAGS.task,
        summary_frequency=FLAGS.summary_frequency,
        steps_to_average=FLAGS.steps_to_average,
        num_training_steps=num_iters,
        start_fresh=True)
  else:
    gclassifier = genre_classifier.GenreClassifier(
        hparams=hparams,
        training_data_path=FLAGS.sequence_examples,
        output_dir=train_dir,
        tf_master=FLAGS.master,
        summary_frequency=FLAGS.summary_frequency,
        steps_to_average=FLAGS.steps_to_average,
        task_id=FLAGS.task,
        start_fresh=True)

  logging.info('Train dir: %s', train_dir)

  for _ in gclassifier.training_loop():
    pass


def main(_):
  train_genre_classifier(
      FLAGS.experiment_dir, num_iters=FLAGS.num_training_steps)


if __name__ == '__main__':
  app.run()
