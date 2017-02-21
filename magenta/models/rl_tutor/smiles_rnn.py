"""Defines a class and operations for the SmilesRNN model.

Can create a train a basic RNN to predict the next character in a SMILES 
molecule sequence, or allow such a model to be loaded from a checkpoint file, 
primed, and used to predict next tokens.

This class can be used as the q_network and target_q_network for the RLTutor
class.

The graph structure of this model is similar to basic_rnn, but more flexible.
It allows you to either train it with data from a SmilesLoader object, or 
just 'call' it to produce the next action.

It also provides the ability to add the model's graph to an existing graph as a
subcomponent, and then load variables from a checkpoint file into only that
piece of the overall graph.

These functions are necessary for use with the RL Tutor class.
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rl_tutor_ops
import smiles_data_loader

SMILES_DATA = '/home/natasha/Dropbox/Google/SMILES-Project/data/'

def reload_files():
  """Used to reload the imported dependency files (necessary for jupyter 
  notebooks).
  """
  reload(rl_tutor_ops)


class SmilesRNN(object):
  """Builds graph for a Smiles RNN and instantiates weights from a checkpoint.

  Can either train an RNN on SMILES molecule data for scratch, or load a pre-
  trained version to be included as part of an RL Tutor graph.
  """

  def __init__(self, checkpoint_dir, graph=None, scope='smiles_rnn', checkpoint_file=None, 
               hparams=None, rnn_type='default', checkpoint_scope='smiles_rnn', 
               load_training_data=False, data_file=SMILES_DATA+'250k_drugs_clean.smi', 
               vocab_file=SMILES_DATA+'zinc_char_list.json', pickle_file=SMILES_DATA+'smiles.p',
               output_every=1000, vocab_size=rl_tutor_ops.NUM_CLASSES_SMILE):
    """Initialize by building the graph and loading a previous checkpoint.

    Args:
      checkpoint_dir: Path to the directory where the checkpoint file is saved or 
        will be saved.
      graph: A tensorflow graph where the SmilesRNN's graph will be added. If 
        None, class will create its own graph.
      scope: The tensorflow scope where this network will be saved.
      checkpoint_file: Path to a backup checkpoint file to be used if none can 
        be found in the checkpoint_dir
      hparams: A tf_lib.HParams object. Must match the hparams used to create 
        the checkpoint file.
      rnn_type: If 'default', will use the basic LSTM described in the 
        research paper. 
      checkpoint_scope: The scope in lstm which the model was originally defined
        when it was first trained.
      load_training_data: A bool that should be true if the model is going to load
        SMILES training data for training from scratch.
      data_file: A file containing text strings representing SMILES encodings of 
        molecules. Needs to be provided if RNN will be trained from scratch.
      vocab_file: A json file containing a list of all the characters in the SMILES
        vocabulary
      pickle_file: A pickle file containing pre-processed batches of SMILES data
    """
    self.session = None
    self.scope = scope
    self.checkpoint_scope = checkpoint_scope
    self.rnn_type = rnn_type
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = checkpoint_file
    self.load_training_data = load_training_data
    self.output_every=output_every
    self.vocab_size = vocab_size

    if graph is None:
      self.graph = tf.Graph()
    else:
      self.graph = graph

    if hparams is not None:
      tf.logging.info('Using custom hparams')
      self.hparams = hparams
    else:
      tf.logging.info('Empty hparams string. Using defaults')
      self.hparams = rl_tutor_ops.smiles_hparams()

    if load_training_data:
      self.data_file = data_file
      self.vocab_file = vocab_file
      self.pickle_file = pickle_file

      self.data_loader = smiles_data_loader.SmilesLoader(vocab_file, data_file, 
                                                         pickle_file, self.hparams.batch_size)
      self.vocab_size = self.data_loader.vocab_size
      self.train_accuracies = []
      self.val_accuracies = []
      self.train_perplexities = []
      self.val_perplexities = []

    self.build_graph()
    self.state_value = self.get_zero_state()

    self.variable_names = rl_tutor_ops.get_variable_names(self.graph, 
                                                          self.scope)

    self.session = None
    self.saver = None

    self.flat = None
    self.index = None
    self.relevant = None

  def get_zero_state(self):
    """Gets an initial state of zeros appropriate for a single input.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of 1 x cell size zeros.
    """
    return np.zeros((1, self.cell.state_size))
  
  def get_zero_state_batch(self):
    """Gets an initial state of zeros of the appropriate size for a batch.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """
    return np.zeros((self.hparams.batch_size, self.cell.state_size))

  def restore_initialize_prime(self, session):
    """Saves the session, restores variables from checkpoint, primes model.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)
    self.prime_model()

  def initialize_and_restore(self, session=None):
    """Saves the session, restores variables from checkpoint.

    Args:
      session: A tensorflow session.
    """
    if session is None:
      self.session = tf.Session(graph=self.graph)
    else:
      self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)

  def initialize_new(self, session=None):
    """Saves the session, initializes all variables to random values.

    Args:
      session: A tensorflow session.
    """
    with self.graph.as_default():
      if session is None:
        self.session = tf.Session(graph=self.graph)
      else:
        self.session = session
      self.session.run(tf.initialize_all_variables())

  def get_variable_name_dict(self):
    """Constructs a dict mapping the checkpoint variables to those in new graph.

    Returns:
      A dict mapping variable names in the checkpoint to variables in the graph.
    """
    var_dict = dict()
    for var in self.variables():
      inner_name = rl_tutor_ops.get_inner_scope(var.name)
      inner_name = rl_tutor_ops.trim_variable_postfixes(inner_name)
      if self.rnn_type == 'basic_rnn':
        if 'fully_connected' in inner_name and 'bias' in inner_name:
          # 'fully_connected/bias' has been changed to 'fully_connected/biases'
          # in newest checkpoints.
          var_dict[inner_name + 'es'] = var
        else:
          var_dict[inner_name] = var
      else:
        var_dict[self.checkpoint_scope + '/' + inner_name] = var
      
    return var_dict

  def build_graph(self):
    """Constructs the portion of the graph that belongs to this model."""

    tf.logging.info('Initializing smiles RNN graph for scope %s', self.scope)

    with self.graph.as_default():
      with tf.variable_scope(self.scope):
        # Make an LSTM cell with the number and size of layers specified in
        # hparams.
        self.cell = rl_tutor_ops.make_cell(self.hparams, self.rnn_type)

        # Shape of input sequence is batch size, seq length, number of
        # output token actions.
        self.input_sequence = tf.placeholder(tf.float32,
                                              [None, None,
                                               self.hparams.one_hot_length],
                                              name='input_sequence')
        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
        self.initial_state = tf.placeholder(tf.float32,
                                            [None, self.cell.state_size],
                                            name='initial_state')
        self.train_labels = tf.placeholder(tf.int64,
                                           [None, None],
                                           name='train_labels')

        # Closure function is used so that this part of the graph can be
        # re-run in multiple places, such as __call__.
        def run_network(smiles_seq, lens, initial_state, swap_memory=True,
                        parallel_iterations=1):
          """Internal function that defines the RNN network structure.

          Args:
            smiles_seq: A batch of smiles sequences of one-hot tokens
            lens: Lengths of the input_sequence.
            initial_state: Vector representing the initial state of the RNN.
            swap_memory: Uses more memory and is faster.
            parallel_iterations: Argument to tf.nn.dynamic_rnn.
          Returns:
            Output of network (either softmax or logits) and RNN state.
          """
          outputs, final_state = tf.nn.dynamic_rnn(
              self.cell,
              smiles_seq,
              sequence_length=lens,
              initial_state=initial_state,
              swap_memory=swap_memory,
              parallel_iterations=parallel_iterations)

          outputs_flat = tf.reshape(outputs,
                                    [-1, self.hparams.rnn_layer_sizes[-1]])
          logits_flat = tf.contrib.layers.legacy_linear(
              outputs_flat, self.hparams.one_hot_length)
          return logits_flat, final_state

        self.run_network = run_network

        (self.logits, self.state_tensor) = run_network(
              self.input_sequence, self.lengths, self.initial_state)
        self.softmax = tf.nn.softmax(self.logits)

        # Code for training the model
        self.labels_flat = tf.reshape(self.train_labels, [-1])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          self.logits, self.labels_flat))
        
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(
            self.hparams.initial_learning_rate, self.global_step, self.hparams.decay_steps,
            self.hparams.decay_rate, staircase=True, name='learning_rate')

        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.hparams.clip_norm)
        self.train_op = self.opt.apply_gradients(zip(clipped_gradients, self.params),
                                            self.global_step)

        # Code for evaluating the model
        self.correct_predictions = tf.to_float(
          tf.nn.in_top_k(self.logits, self.labels_flat, 1))
        self.accuracy = tf.reduce_mean(self.correct_predictions) * 100
        self.perplexity = tf.exp(self.loss)

  def restore_vars_from_checkpoint(self, checkpoint_dir):
    """Loads model weights from a saved checkpoint.

    Args:
      checkpoint_dir: Directory which contains a saved checkpoint of the
        model.
    """
    tf.logging.info('Restoring variables from checkpoint')

    var_dict = self.get_variable_name_dict()
    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)

    tf.logging.info('Checkpoint dir: %s', checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
      tf.logging.warn("Can't find checkpoint file, using backup, which is %s", 
                      self.checkpoint_file)
      checkpoint_file = self.checkpoint_file
    tf.logging.info('Checkpoint file: %s', checkpoint_file)

    saver.restore(self.session, checkpoint_file)

  def prime_model(self, primer_input):
    """Primes the model with a sequence that has already been correctly encoded."""
    with self.graph.as_default():
      # Run model over primer sequence.
      primer_input_batch = np.tile([primer_input], (self.hparams.batch_size, 1, 1))
      lengths = np.full(self.hparams.batch_size, len(self.primer), dtype=int)

      self.state_value, softmax = self.session.run(
          [self.state_tensor, self.softmax],
          feed_dict={self.initial_state: self.state_value,
                     self.input_sequence: primer_input_batch,
                     self.lengths: lengths})
      priming_output = softmax[-1, :]
      self.priming_token = self.get_token_from_softmax(priming_output)

  def get_token_from_softmax(self, softmax):
    """Extracts a one-hot encoding of the most probable token.

    Args:
      softmax: Softmax probabilities over possible next tokens.
    Returns:
      One-hot encoding of most probable token.
    """

    token_idx = np.argmax(softmax)
    encoding = rl_tutor_ops.make_onehot([token_idx], self.vocab_size)
    return np.reshape(encoding, (self.vocab_size))

  def __call__(self):
    """Allows the network to be called, as in the following code snippet!

        q_network = SmilesRNN(...)
        q_network()

    The q_network() operation can then be placed into a larger graph as a tf op.

    Note that to get actual values from call, must do session.run and feed in
    input_sequence, lengths, and initial_state in the feed dict.

    Returns:
      Either softmax probabilities over tokens, or raw logit scores.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        batch_size = tf.shape(self.input_sequence)[0]
        max_length = tf.shape(self.input_sequence)[1]
        logits, self.state_tensor = self.run_network(self.input_sequence, 
          self.lengths, self.initial_state)
        
        # Get last relevant states
        self.index = tf.range(0, batch_size) * max_length + (self.lengths - 1)
        self.relevant = tf.gather(self.flat, self.index)
        
        return self.relevant

  def train(self, num_steps=30000):
    """Runs one batch of training data through the model.

    Uses a queue runner to pull one batch of data from the training files
    and run it through the model.

    Returns:
      A batch of softmax probabilities and model state vectors.
    """
    if not self.load_training_data or not self.data_loader:
      print "Error! must load training data in order to train"

    if not self.session: self.initialize_new()

    with self.graph.as_default():
      self.saver = tf.train.Saver()

      zero_state = self.get_zero_state_batch()

      step = 0
      while step < num_steps:
        X, Y, lens = self.data_loader.next_batch()
        feed_dict = {self.input_sequence: X,
                   self.train_labels: Y,
                   self.lengths: lens, 
                   self.initial_state: zero_state}
        if step % self.output_every == 0:
          _, step, train_acc, train_pplex = self.session.run([self.train_op, 
                                                              self.global_step,
                                                              self.accuracy,
                                                              self.perplexity], feed_dict)
          X, Y, lens = self.data_loader.next_batch(dataset='val')
          feed_dict = {self.input_sequence: X,
                     self.train_labels: Y,
                     self.lengths: lens, 
                     self.initial_state: zero_state}
          _, val_acc, val_pplex = self.session.run([self.train_op, 
                                                    self.accuracy,
                                                    self.perplexity], feed_dict)

          print "Training iteration", step
          print "\t Training accuracy", train_acc, "perplexity", train_pplex
          print "\t Validation accuracy", val_acc, "perplexity", val_pplex
          self.saver.save(self.session, self.checkpoint_dir+self.scope, 
                          global_step=step)

          self.train_perplexities.append(train_pplex)
          self.train_accuracies.append(train_acc)
          self.val_perplexities.append(val_pplex)
          self.val_accuracies.append(val_acc)
        else:
          _, step = self.session.run([self.train_op, self.global_step], feed_dict)


  def plot_training_progress(self, save_fig=False, directory=None):
    """Plots the cumulative rewards received as the model was trained.

    If image_name is None, should be used in jupyter notebook. If 
    called outside of jupyter, execution of the program will halt and 
    a pop-up with the graph will appear. Execution will not continue 
    until the pop-up is closed.

    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
    """
    if directory is None:
      directory = self.checkpoint_dir

    reward_batch = self.output_every
    x = [reward_batch * i for i in np.arange(len(self.train_accuracies))]
    
    plt.figure()
    plt.plot(x, self.train_accuracies)
    plt.plot(x, self.val_accuracies)
    plt.xlabel('Training epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='best')
    if save_fig:
      plt.savefig(directory + '/' + self.scope + '_training_accuracies.png')
    else:
      plt.show()

    plt.figure()
    plt.plot(x, self.train_perplexities)
    plt.plot(x, self.val_perplexities)
    plt.xlabel('Training epoch')
    plt.ylabel('Perplexity')
    plt.legend(['Train', 'Validation'], loc='best')
    if save_fig:
      plt.savefig(directory + '/' + self.scope + '_training_perplexities.png')
    else:
      plt.show()

  def get_next_token_from_token(self, token):
    """Given a token, uses the model to predict the most probable next token.

    Args:
      token: A one-hot encoding of the token.
    Returns:
      Next token in the same format.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        singleton_lengths = np.full(self.hparams.batch_size, 1, dtype=int)

        input_batch = np.reshape(token, 
                                 (self.hparams.batch_size, 1, self.vocab_size))

        softmax, self.state_value = self.session.run(
            [self.softmax, self.state_tensor],
            {self.input_sequence: input_batch,
             self.initial_state: self.state_value,
             self.lengths: singleton_lengths})

        return self.get_token_from_softmax(softmax)

  def variables(self):
    """Gets names of all the variables in the graph belonging to this model.

    Returns:
      List of variable names.
    """
    with self.graph.as_default():
      return [v for v in tf.all_variables() 
              if v.name.startswith(self.scope) 
              and 'global_step' not in v.name and 'Adam' not in v.name]
