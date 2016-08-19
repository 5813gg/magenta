"""Defines the GenreClassifier class, which predicts genre from melody data.

A GenreClassifier consists of an RNN (containing LSTM cells) that operates on
monophonic melody data and is trained to predict the next note. The hidden
states of the RNN are collected, and then connected to averaging and max
pooling layers, and finally MLP layers. The MLP output is used to predict the
genre of the melody.

Gradients are propagated backwards through the entire model, so genre prediction
error is used to train the weights of the LSTM. The total cost used to compute
gradients is as follows:
    cost = genre_weight * genre_cost + note_weight * note_cost,
Where genre_cost is the prediction error on predicting genre, and note_cost is
the total prediction error for predicting the next notes in the sequence using
the RNN. The genre_weight is set to be several orders of magnitude larger than
the note_weight, so that the cost emphasizes correct prediction of the genre.

The melody input to the model is assumed to contain several bits encoding the
note, and then several bits encoding the genre.
"""

import collections
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from ....rl_rnn import rl_rnn_ops

from ... import tensorflow_ops

NOTE_ONE_HOT_LENGTH = 38
DEFAULT_NUM_GENRES = 2


def reload_files():
  reload(tensorflow_ops)


def default_hparams():
  """Generates the default hparams used to train a GenreClassifier."""
  return tf.HParams(
      batch_size=1,
      l2_reg=2.5e-5,
      clip_norm=5,
      initial_learning_rate=0.001,
      decay_steps=128000,
      decay_rate=0.85,
      rnn_layer_sizes=[1024],
      mlp_layer_sizes=[256, 32],
      skip_first_n_losses=8,
      one_hot_length=NOTE_ONE_HOT_LENGTH + DEFAULT_NUM_GENRES,
      exponentially_decay_learning_rate=True)


class GenreClassifier(object):
  """Implements a model that can predict the genre of monophonic melodies."""

  def __init__(self,
               hparams=None,
               genre_weight=40.0,
               note_weight=.001,
               max_gradient_norm=10,
               training_data_path='path/myfile.tfrecord',
               output_dir='/tmp/genre_classifier',
               note_input_length=NOTE_ONE_HOT_LENGTH,
               num_genres=DEFAULT_NUM_GENRES,
               scope='genre_classifier',
               graph=None,
               num_training_steps=30000,
               summary_frequency=10,
               steps_to_average=20,
               tf_master='local',
               task_id=0,
               parallel_iterations=1,
               swap_memory=True,
               start_fresh=True):
    """Initializes GenreClassifier class.

    Args:
      hparams: Hyperparameters of the model, including layer sizes and learning
        rate.
      genre_weight: The weight placed on genre prediction error in the cost
        function.
      note_weight: The weight placed on next note prediction error in the cost
        function.
      max_gradient_norm: The value to which gradients will be clipped.
      training_data_path: A tfrecord file containing the melody sequences used
        for training.
      output_dir: A path where the model checkpoints will be saved.
      note_input_length: The number of bits of element of an input sequence that
        encode the note. The following bits encode the genre.
      num_genres: The number of genres the model needs to discriminate between.
      scope: A string containing the Tensorflow scope for this model.
      graph: A Tensorflow graph object to which to add this graph. If None, the
        model will create a new graph.
      num_training_steps: The number of batches used to train the model.
      summary_frequency: The number of training steps before the model will
        output information about accuracy, etc.
      steps_to_average: The number of training steps over which the model will
        compute average accuracy when outputting a summary.
      tf_master: BNS name of the TensorFlow master to use.
      task_id: Task id of the replica running the training.
      parallel_iterations: The number of iterations to run in parallel. Those
        operations which do not have any temporal dependency
        and can be run in parallel, will be. This parameter trades off
        time for space. Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU. This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      start_fresh: If True will erase the checkpoint files in output_dir and
        learn a new model from scratch. If False will resume from the latest
        checkpoint file in output_dir.
    """
    self.num_genres = num_genres
    self.note_input_length = note_input_length
    self.one_hot_length = note_input_length + num_genres
    self.scope = scope
    self.transpose_amount = 0
    self.output_dir = output_dir
    self.num_training_steps = num_training_steps
    self.summary_frequency = summary_frequency
    self.steps_to_average = steps_to_average
    self.tf_master = tf_master
    self.task_id = task_id
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    self.genre_weight = genre_weight
    self.note_weight = note_weight
    self.max_gradient_norm = max_gradient_norm

    if start_fresh and os.path.exists(self.output_dir):
      logging.info('Starting fresh - erasing the checkpoint files in %s',
                   self.output_dir)
      files = os.listdir(self.output_dir)
      for f in files:
        os.remove(os.path.join(self.output_dir, f))

    if hparams is not None:
      logging.info('Using custom hparams')
      self.hparams = hparams
    else:
      logging.info('Empty hparams string. Using defaults')
      self.hparams = default_hparams()

    # check logistics to make sure the model can work
    if self.hparams.one_hot_length != self.one_hot_length:
      logging.fatal('ERROR! The total length of the one-hot encoding of the '
                    'input in hparams (%s) must equal the number of bits '
                    'encoding the note, plus the number of bits encoding the '
                    'genre (%s).', self.hparams.one_hot_length,
                    self.one_hot_length)

    if self.hparams.batch_size != 1:
      logging.fatal('ERROR! This code can currently only handle a batch size of'
                    ' 1 due to its ability to handle variable length sequences'
                    ' in the tf.while_loop')

    self.training_file_list = gfile.Glob(training_data_path)
    if not self.training_file_list:
      logging.fatal('No files found matching %s', training_data_path)
    logging.info('Dataset files: %s', self.training_file_list)

    if graph is None:
      self.graph = tf.Graph()
    else:
      self.graph = graph
    self.build_graph()

    self.state_value = self.__get_zero_state()

    self.session = None
    self.supervisor = None

  def __get_zero_state(self):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell

    Returns:
      matrix of batch_size x cell size zeros
    """
    return np.zeros((self.hparams.batch_size, self.cell.state_size))

  def build_graph(self, zero_initial_state=True):
    """Constructs the portion of the TF graph that belongs to this model."""

    logging.info('Initializing melody RNN graph for scope %s', self.scope)

    with self.graph.as_default():
      with tf.variable_scope(self.scope):
        # make an LSTM cell with the number and size of layers specified in
        # hparams
        self.cell = tensorflow_ops.make_cell(self.hparams)
        self.note_layer_output_size = self.hparams.rnn_layer_sizes[-1]

        if zero_initial_state:
          self.initial_state = self.cell.zero_state(
              batch_size=self.hparams.batch_size, dtype=tf.float32)
        else:
          self.initial_state = tf.placeholder(tf.float32,
                                              [self.hparams.batch_size,
                                               self.cell.state_size],
                                              name='initial_state')

        # input data comes from a FIFO tfrecord reader queue
        (self.genre_sequence, self.labels,
         self.lengths) = tensorflow_ops.dynamic_rnn_batch(
             self.training_file_list, self.hparams)

        # get rid of the genre bits on the input data
        self.melody_sequence = self.genre_sequence[:, :, :
                                                   self.note_input_length]

        self.genre_sequence = tf.check_numerics(self.genre_sequence,
                                                'genre_sequence has nans')
        self.melody_sequence = tf.check_numerics(self.melody_sequence,
                                                 'melody_sequence has nans')

        # get the length of this sequence (note... only works with batch size
        # of 1)
        num_steps = self.lengths[0]

        # create tensor arrays in order to create dynamic loop over length
        # of input sequence
        def create_tensor_array(name):
          return tensor_array_ops.TensorArray(
              tf.float32, size=num_steps, tensor_array_name=self.scope + name)

        self.output_ta = create_tensor_array('output')
        self.state_ta = create_tensor_array('states')
        self.input_ta = create_tensor_array('inputs')

        self.input_ta = self.input_ta.unpack(
            tf.reshape(self.melody_sequence, [-1, self.note_input_length]))

        state = self.initial_state
        time_step = array_ops.constant(0, dtype=tf.int32, name='time_step')

        def compute_next_time_step(time_step, output_ta_t, state_ta_t, state):
          """Function that comprises the body of the tf.while_loop.

          Runs the LSTM cell on inputs retrieved from the appropriate time
          step of the input sequence tensor array. Stores outputs and LSTM
          states in other tensor arrays.

          Args:
            time_step: The beat of the input melody sequence.
            output_ta_t: A tensor array that stores the outputs of the model.
            state_ta_t: A tensor array that stores the hidden states of the
              model.
            state: The current hidden state of the model.

          Returns:
            The time step, incremented by 1, the output tensor array with
            another output written to it, the state tensor array with another
            state written to it, and the hidden state of the model. These will
            be used in the next iteration of this function.
          """
          input_t = self.input_ta.read(time_step)
          input_t = tf.reshape(input_t, [1, self.note_input_length])

          # run the LSTM cell to produce an output and the internal LSTM
          # state
          output_t, state_t = self.cell(input_t, state)

          # save to tensor arrays
          output_ta_t = output_ta_t.write(time_step, output_t)
          state_ta_t = state_ta_t.write(time_step, state_t)

          return time_step + 1, output_ta_t, state_ta_t, state_t

        # run the tf.while_loop to process the melody sequence, saving the
        # outputs and states at each step
        (_, self.output_final_ta, self.state_final_ta,
         self.final_state) = control_flow_ops.while_loop(
             cond=lambda time_step, *_: time_step < num_steps,
             body=compute_next_time_step,
             loop_vars=(time_step, self.output_ta, self.state_ta, state),
             parallel_iterations=self.parallel_iterations,
             swap_memory=self.swap_memory)

        # get values out of the tensor arrays used by the loop
        self.final_outputs = self.output_final_ta.pack()
        self.states = self.state_final_ta.pack()

        # reshape outputs
        self.final_outputs = tf.reshape(self.final_outputs,
                                        [-1, self.note_layer_output_size])
        self.states = tf.reshape(self.states, [-1, self.cell.state_size])

        # final fully connected layer to predict next note
        self.reshaped_outputs = tf.reshape(self.final_outputs,
                                           [-1, self.note_layer_output_size])
        self.logits = tf.contrib.layers.fully_connected(
            inputs=self.reshaped_outputs,
            num_outputs=self.note_input_length,
            activation_fn=None)

        # skip the first n losses and reshape
        self.trunc_logits = self.logits[self.hparams.skip_first_n_losses:, :]
        self.flat_labels = tf.reshape(
            self.labels[:, self.hparams.skip_first_n_losses:], [-1])

        # compute the note loss
        seq_length = num_steps - self.hparams.skip_first_n_losses
        self.timestep_weights = tf.ones(
            [self.hparams.batch_size * (seq_length)])
        self.note_loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.trunc_logits], [self.flat_labels], [self.timestep_weights])
        self.note_cost = tf.reduce_sum(
            self.note_loss) / self.hparams.batch_size

        # attach another layer to max pool and average the RNN hidden states
        self.states = self.states[self.hparams.skip_first_n_losses:, :]
        self.average_states = tf.reduce_mean(self.states, 0)
        self.max_pool_output = tf.nn.max_pool(
            tf.reshape(self.states, [1, -1, self.cell.state_size, 1]),
            ksize=[1, 32, 4, 1],
            strides=[1, 16, 2, 1],
            padding='SAME')
        max_pool_length = self.cell.state_size / 2
        self.max_pool = tf.reshape(self.max_pool_output,
                                   [-1, max_pool_length])
        self.max_pool_avg = tf.reduce_mean(self.max_pool, 0)
        self.genre_input = tf.concat(0,
                                     [self.average_states, self.max_pool_avg])

        # build final genre prediction MLP portion of graph
        self.genre_input_size = self.cell.state_size + max_pool_length
        self.genre_input = tf.reshape(self.genre_input,
                                      [1, self.genre_input_size])
        self.genre_hidden = tf.contrib.layers.stack(
            inputs=self.genre_input,
            layer=tf.contrib.layers.fully_connected,
            stack_args=self.hparams.mlp_layer_sizes,
            activation_fn=tf.nn.relu)
        # TODO(natashajaques): add dropout here
        self.genre_logits = tf.contrib.layers.fully_connected(
            inputs=self.genre_hidden,
            num_outputs=self.num_genres,
            activation_fn=None)

        # compute genre prediction cost
        num_genre_bits = self.one_hot_length - self.num_genres
        self.genre_bits = self.genre_sequence[0, 0, num_genre_bits:]
        self.genre_bits = tf.reshape(self.genre_bits, [1, self.num_genres])
        self.genre_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.genre_logits,
                                                    self.genre_bits))

        # compute total cost
        self.cost = self.note_weight * self.note_cost
        self.cost += self.genre_weight * self.genre_cost

        # learning rate stuff
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.hparams.exponentially_decay_learning_rate:
          self.learning_rate = tf.train.exponential_decay(
              self.hparams.initial_learning_rate,
              self.global_step,
              self.hparams.decay_steps,
              self.hparams.decay_rate,
              staircase=True,
              name='learning_rate')
        else:
          self.learning_rate = tf.Variable(
              self.hparams.initial_learning_rate, trainable=False)

        # backproporama
        self.trainable_variables = tf.trainable_variables()
        self.gradients = tf.gradients(self.cost, self.trainable_variables)
        self.gradients, _ = tf.clip_by_global_norm(self.gradients,
                                                   self.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(self.gradients, self.trainable_variables),
            global_step=self.global_step)

        # compute measures of accuracy, loss, etc to help in evaluating and
        # debugging the model
        (self.cross_entropy,
         self.log_perplexity) = tensorflow_ops.log_perplexity_loss(
             self.trunc_logits, self.flat_labels)
        self.note_accuracy = tensorflow_ops.eval_accuracy(self.trunc_logits,
                                                          self.flat_labels)
        self.genre_accuracy = tensorflow_ops.eval_accuracy(
            self.genre_logits, tf.argmax(self.genre_bits, 1))

  def initialize_new_session(self):
    """Initializes the session using a supervisor, starts queue runners."""
    with self.graph.as_default():
      self.supervisor = tf.Supervisor(
          graph=self.graph,
          logdir=self.output_dir,
          is_chief=(True),
          save_model_secs=30,
          global_step=self.global_step)
      self.session = self.supervisor.PrepareSession('local')
      self.supervisor.StartQueueRunners(self.session)

  def training_loop(self):
    """A generator function which trains the model and outputs summaries.

    Runs summary_frequency training steps between each yield.

    Yields:
      A dict of training metrics
    """

    with self.graph.as_default():
      summary_op = tf.merge_summary([
          tf.scalar_summary('cross_entropy_loss', self.cross_entropy),
          tf.scalar_summary('log_perplexity', self.log_perplexity),
          tf.scalar_summary('learning_rate', self.learning_rate),
          tf.scalar_summary('note_accuracy', self.note_accuracy),
          tf.scalar_summary('genre_accuracy', self.genre_accuracy),
          tf.scalar_summary('note_cost', self.note_cost),
          tf.scalar_summary('genre_cost', self.genre_cost),
          tf.scalar_summary('cost', self.cost)
      ])

    # Run training loop.
    self.initialize_new_session()
    summary_writer = tf.SummaryWriter(self.output_dir, self.session.graph)
    step = 0
    global_step = 0

    logging.info('Starting training loop')
    try:
      note_accuracies = collections.deque(maxlen=self.steps_to_average)
      genre_accuracies = collections.deque(maxlen=self.steps_to_average)
      while not self.supervisor.ShouldStop() and (
          global_step < self.num_training_steps):
        (genre_cost, note_cost, total_cost, cross_ent, log_perplexity,
         genre_acc, note_acc, global_step, learn_rate, _,
         serialized_summaries) = self.session.run(
             [self.genre_cost, self.note_cost, self.cost, self.cross_entropy,
              self.log_perplexity, self.genre_accuracy, self.note_accuracy,
              self.global_step, self.learning_rate, self.train_op, summary_op])

        note_accuracies.append(note_acc)
        genre_accuracies.append(genre_acc)
        if step % self.summary_frequency == 0:
          note_avg_accuracy = sum(note_accuracies) / len(note_accuracies)
          genre_avg_accuracy = sum(genre_accuracies) / len(genre_accuracies)
          logging.info('Cost: %s - Genre Cost: %s, - Note Cost: %s', total_cost,
                       genre_cost, note_cost)
          logging.info('Session Step: %s - Global Step: %s',
                       '{:,}'.format(step), '{:,}'.format(global_step))
          logging.info('Learning Rate: %f', learn_rate)
          logging.info('Cross-entropy: %.3f - Log-perplexity: %.3f', cross_ent,
                       log_perplexity)
          logging.info('Note accuracy - step: %.2f - avg (last %d): %.2f',
                       note_acc, self.steps_to_average, note_avg_accuracy)
          logging.info('Genre accuracy - step: %.2f - avg (last %d): %.2f',
                       genre_acc, self.steps_to_average, genre_avg_accuracy)
          logging.info('')

          summary_writer.add_summary(serialized_summaries,
                                     global_step=global_step)
          summary_writer.flush()

          yield {'step': step,
                 'global_step': global_step,
                 'cost': total_cost,
                 'genre_cost': genre_cost,
                 'note_cost': note_cost,
                 'loss': cross_ent,
                 'log_perplexity': log_perplexity,
                 'note_accuracy': note_acc,
                 'note_average_accuracy': note_avg_accuracy,
                 'genre_accuracy': genre_acc,
                 'genre_average_accuracy': genre_avg_accuracy,
                 'learning_rate': learn_rate}
        step += 1
      self.supervisor.saver.save(
          self.session,
          self.supervisor.save_path,
          global_step=self.supervisor.global_step)
    except tf.errors.OutOfRangeError as e:
      logging.warn('Got error reported to coordinator: %s', e)
    finally:
      try:
        self.supervisor.Stop()
        summary_writer.close()
      except RuntimeError as e:
        logging.warn('Got runtime error: %s', e)

  def load_from_checkpoint(self, checkpoint_dir, backup_checkpoint_file=None):
    """Restores model weights from a saved checkpoint into a new session.

    Args:
      checkpoint_dir: Path to a directory containing a checkpointed version of
        the model.
      backup_checkpoint_file: Path to a checkpoint file to use if a checkpoint
        cannot be found in the 'checkpoint_dir'.
    """
    with self.graph.as_default():
      saver = tf.train.Saver()
    self.session = tf.Session(self.tf_master, graph=self.graph)

    checkpoint_file = tf.latest_checkpoint(checkpoint_dir)
    if not checkpoint_file:
      if backup_checkpoint_file is None:
        logging.fatal('Error! Could not find a checkpoint file in the requested'
                      ' checkpoint directory and no backup file provided')
        return
      else:
        checkpoint_file = backup_checkpoint_file

    saver.restore(self.session, checkpoint_file)

  def load_from_checkpoint_into_existing_graph(self, session, checkpoint_dir,
                                               checkpoint_scope,
                                               backup_checkpoint_file=None):
    """Restores model weights from saved checkpoint into part of existing graph.

    Args:
      session: A tensorflow session which already contains a graph.
      checkpoint_dir: Path to a directory containing a checkpointed version of
        the model.
      checkpoint_scope: A string containing the scope that the variables in the
        checkpoint will have.
      backup_checkpoint_file: Path to a checkpoint file to use if a checkpoint
        cannot be found in the 'checkpoint_dir'.
    """
    self.session = session

    var_dict = dict()
    for var in self.variables():
      inner_name = rl_rnn_ops.get_inner_scope(var.name)
      inner_name = rl_rnn_ops.trim_variable_postfixes(inner_name)
      var_dict[checkpoint_scope + '/' + inner_name] = var

    logging.info('Restoring variables from checkpoint into GenreClassifier')

    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)

    logging.info('Checkpoint dir: %s', checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if not checkpoint_file:
      if backup_checkpoint_file is None:
        logging.fatal('Error! Could not find a checkpoint file in the requested'
                      ' checkpoint directory and no backup file provided')
        return
      else:
        checkpoint_file = backup_checkpoint_file
    logging.info('Checkpoint file: %s', checkpoint_file)

    saver.restore(self.session, checkpoint_file)

  def variables(self):
    """Gets names of all the variables in the graph belonging to this model.

    Returns:
      List of variable names.
    """
    with self.graph.as_default():
      return [v for v in tf.all_variables() if v.name.startswith(self.scope)]

  def evaluate(self, checkpoint_dir, num_evaluations=1000,
               summary_frequency=None):
    """Computes average note and genre training accuracy over 1000 examples.

    The model's session must be initialized before calling this function.

    Args:
      checkpoint_dir: Path to a directory containing a checkpoint of a trained
        model.
      num_evaluations: The number of training examples over which to compute
        the prediction accuracy.
      summary_frequency: The number of evaluations to perform before outputting
        an average. Defaults to the same summary frequency used in training if
        not provided.
    """
    if summary_frequency is None:
      summary_frequency = self.summary_frequency

    with self.graph.as_default():
      saver = tf.train.Saver()
    self.session = tf.Session(self.tf_master, graph=self.graph)

    checkpoint_path = tf.latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
      logging.fatal('Error! Could not find a checkpoint file in the requested'
                    ' checkpoint directory.')
      return

    saver.restore(self.session, checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.session, coord=coord)

    genre_accuracies = []
    note_accuracies = []

    logging.info('Starting evaluation. Will report average results every %s '
                 'steps, and run for a total of %s steps', summary_frequency,
                 num_evaluations)
    for i in range(num_evaluations):
      genre_acc, note_acc = self.session.run([self.genre_accuracy,
                                              self.note_accuracy])
      genre_accuracies.append(genre_acc)
      note_accuracies.append(note_acc)

      if i % summary_frequency == 0:
        mean_genre_acc = np.mean(genre_accuracies)
        mean_note_acc = np.mean(note_accuracies)

        logging.info('Batches: %s', i)
        logging.info('Genre accuracy: %s', mean_genre_acc)
        logging.info('Note accuracy: %s', mean_note_acc)
        logging.info('')

    mean_genre_acc = np.mean(genre_accuracies)
    mean_note_acc = np.mean(note_accuracies)

    logging.info('Done!')
    logging.info('Final genre accuracy: %s', mean_genre_acc)
    logging.info('Final note accuracy: %s', mean_note_acc)

    coord.request_stop()
    coord.join(threads)
