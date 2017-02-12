"""Defines a Deep Q Network (DQN) with augmented reward to create sequences
by using reinforcement learning to fine-tune a pre-trained RNN according
to some domain-specific rewards. This file defines a generic version that 
can be inherited by more domain-specific child classes.

Also implements two alternatives to Q learning: Psi and G learning. The 
algorithm can be switched using the 'algorithm' hyperparameter. 

For more information, please consult the README.md file in this directory.
"""

from collections import deque
import os
from os import makedirs
from os.path import exists
import urllib
import random

import matplotlib.pyplot as plt

import numpy as np
from scipy.misc import logsumexp
import tensorflow as tf

from magenta.music import melodies_lib as mlib
from magenta.music import midi_io

import rl_tutor_ops

# Training data sequences are limited to this length, so the padding queue pads
# to this length.
TRAIN_SEQUENCE_LENGTH = 192

def reload_files():
  """Used to reload the imported dependency files (necessary for jupyter 
  notebooks).
  """
  reload(rl_tutor_ops)
  reload(rl_tuner_eval_metrics)


class RLTutor(object):
  """Implements a recurrent DQN designed to produce melody sequences."""

  def __init__(self, output_dir,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='default',
               reward_scaler=1.0,
               exploration_mode='egreedy',
               priming_mode='random',
               stochastic_observations=False,
               algorithm='q',

               # Pre-trained RNN to load and train
               rnn_checkpoint_dir=None,
               rnn_checkpoint_file=None,
               rnn_type='default',
               rnn_hparams=None,

               # Logistics.
               input_size=rl_tutor_ops.NUM_CLASSES,
               num_actions=rl_tutor_ops.NUM_CLASSES,
               midi_primer=None,
               save_name='rl_tuner.ckpt',
               output_every_nth=1000,
               training_file_list=None,
               summary_writer=None,
               initialize_immediately=True):
    """Initializes the RLTutor class.

    Args:
      output_dir: Where the model will save its compositions (midi files).
      dqn_hparams: A tf_lib.hparams() object containing the hyperparameters of 
        the DQN algorithm, including minibatch size, exploration probability, 
        etc.
      reward_mode: Controls which reward function can be applied. Each domain
        can have several different ones, called within the collect_domain_reward
        function.
      reward_scaler: Controls the emphasis placed on the domain rewards. 
        This value is the inverse of 'c' in the academic paper.
      exploration_mode: can be 'egreedy' which is an epsilon greedy policy, or
        it can be 'boltzmann', in which the model will sample from its output
        distribution to choose the next action.
      priming_mode: Each time the model begins a new composition, it is primed
        with a different method; possibly a random token, or a random training
        sequence.
      stochastic_observations: If False, the token that the model chooses to
        play next (the argmax of its softmax probabilities) deterministically
        becomes the next token it will observe. If True, the next observation
        will be sampled from the model's softmax output.
      algorithm: can be 'default', 'psi', 'g' or 'pure_rl', for different 
        learning algorithms
      rnn_checkpoint_dir: The directory from which the internal 
        RNN loader class will load its checkpointed LSTM.
      rnn_checkpoint_file: A checkpoint file to use in case one cannot be
        found in the rnn_checkpoint_dir.
      rnn_type: If 'default', will use the basic LSTM described in the 
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      rnn_hparams: A tf.HParams object which defines the hyper parameters
        used to train the MelodyRNN model that will be loaded from a checkpoint.
      input_size: the size of the one-hot vector encoding a token that is input
        to the model.
      num_actions: The size of the one-hot vector encoding a token that is
        output by the model.
      save_name: Name the model will use to save checkpoints.
      output_every_nth: How many training steps before the model will print
        an output saying the cumulative reward, and save a checkpoint.
      training_file_list: A list of paths to tfrecord files containing melody 
        training data. This is necessary to use the 'random_midi' priming mode.
      summary_writer: A tf.train.SummaryWriter used to log metrics.
      initialize_immediately: if True, the class will instantiate its component
        MelodyRNN networks and build the graph in the constructor.
    """
    print "In parent class RL Tutor"

    # Make graph.
    self.graph = tf.Graph()

    with self.graph.as_default():
      # Memorize arguments.
      self.input_size = input_size
      self.num_actions = num_actions
      self.output_every_nth = output_every_nth
      self.output_dir = output_dir
      self.save_path = os.path.join(output_dir, save_name)
      self.reward_scaler = reward_scaler
      self.reward_mode = reward_mode
      self.exploration_mode = exploration_mode
      self.stochastic_observations = stochastic_observations
      self.algorithm = algorithm
      self.priming_mode = priming_mode
      self.rnn_checkpoint_dir = rnn_checkpoint_dir
      self.rnn_checkpoint_file = rnn_checkpoint_file
      self.rnn_hparams = rnn_hparams
      self.rnn_type = rnn_type

      self.domain_rewards_only = False
      if self.algorithm == 'g' or self.algorithm == 'pure_rl':
        self.domain_rewards_only = True
      
      if dqn_hparams is None:
        self.dqn_hparams = rl_tutor_ops.default_dqn_hparams()
      else:
        self.dqn_hparams = dqn_hparams
      self.discount_rate = tf.constant(self.dqn_hparams.discount_rate)
      self.target_network_update_rate = tf.constant(
          self.dqn_hparams.target_network_update_rate)

      self.optimizer = tf.train.AdamOptimizer()

      # DQN state.
      self.actions_executed_so_far = 0
      self.experience = deque(maxlen=self.dqn_hparams.max_experience)
      self.iteration = 0
      self.summary_writer = summary_writer
      self.num_times_store_called = 0
      self.num_times_train_called = 0

    # Stored reward metrics.
    self.reward_last_n = 0
    self.rewards_batched = []
    self.domain_reward_last_n = 0
    self.domain_rewards_batched = []
    self.data_reward_last_n = 0
    self.data_rewards_batched = []
    self.eval_avg_reward = []
    self.eval_avg_domain_reward = []
    self.eval_avg_data_reward = []
    self.target_val_list = []

    # Variables to keep track of characteristics of the current composition
    #TODO(natashajaques): Implement composition as a class to obtain data 
    # encapsulation so that you can't accidentally change the leap direction.
    self.generated_seq_step = 0
    self.generated_seq = []

    if not exists(self.output_dir):
      makedirs(self.output_dir)

    if initialize_immediately:
      self.initialize()

  # The following functions should be overwritten in classes inheriting from this one.
  def initialize_internal_models(self):
    """Initializes internal RNN models: q_network, target_q_network, reward_rnn.

    Adds the graphs of the internal RNN models to this graph, by having a separate
    function for this rather than doing it in the constructor, it allows classes
    inheriting from this class to define their own type of q_network.

    This should be overwritten in the child class. 
    """
    raise NotImplementedError

  def prime_internal_model(self, model):
    """Prime an internal model such as the q_network based on priming mode.

    Args:
      model: The internal model that should be primed. 

    Returns:
      The first observation to feed into the model.

    Should be overwritten in child class.
    """
    raise NotImplementedError

  def is_end_of_sequence(self):
    """Returns true if the sequence generated by the model has reached its end.

    Should be overwritten by child class.
    """
    raise NotImplementedError

  def collect_domain_reward(self, obs, action, reward_scores, verbose=False):
    """Domain-specific function that calculates domain reward.
    
    Should be overwritten in child class. 
    """
    raise NotImplementedError

  def render_sequence(self, generated_seq, title='rltuner_sample'):
    """Renders a generated token sequence into required domain format.
    
    Example: may want to render music sequence into MIDI audio.

    Args:
      generated_seq: A list of integer action values.
      title: A title to use in the sequence filename.

    Should be overwritten in child class.
    """
    pass

  def initialize(self, restore_from_checkpoint=True):
    """Initializes internal RNN models, builds the graph, starts the session.

    Adds the graphs of the internal RNN models to this graph, adds the DQN ops
    to the graph, and starts a new Saver and session. 

    Args:
      restore_from_checkpoint: If True, the weights for the 'q_network',
        'target_q_network', and 'reward_rnn' will be loaded from a checkpoint.
        If false, these models will be initialized with random weights. Useful
        for checking how pure RL (with no influence from training data) performs
    """
    with self.graph.as_default():
      # Adds q_network, target_q_network, reward_rnn
      self.initialize_internal_models()
      
      # Add rest of variables to graph.
      tf.logging.info('Adding RL graph variables')
      self.build_graph()

      # Prepare saver and session.
      self.saver = tf.train.Saver()
      self.session = tf.Session(graph=self.graph)
      self.session.run(tf.initialize_all_variables())

      # Initialize internal networks.
      if restore_from_checkpoint:
        self.q_network.initialize_and_restore(self.session)
        self.target_q_network.initialize_and_restore(self.session)
        self.reward_rnn.initialize_and_restore(self.session)

        # Double check that the model was initialized from checkpoint properly.
        reward_vars = self.reward_rnn.variables()
        q_vars = self.q_network.variables()

        reward1 = self.session.run(reward_vars[0])
        q1 = self.session.run(q_vars[0])

        if np.sum((q1 - reward1)**2) == 0.0:
          print "\nSuccessfully initialized internal nets from checkpoint!"
        else:
          print "Error! The model was not initialized from checkpoint properly"
      else:
        self.q_network.initialize_new(self.session)
        self.target_q_network.initialize_new(self.session)
        self.reward_rnn.initialize_new(self.session)

  def build_graph(self):
    """Builds the reinforcement learning tensorflow graph."""

    tf.logging.info('Adding reward computation portion of the graph')
    with tf.name_scope('reward_computation'):
      self.reward_scores = tf.identity(self.reward_rnn(), name='reward_scores')

    tf.logging.info('Adding taking action portion of graph')
    with tf.name_scope('taking_action'):
      # Output of the q network gives the value of taking each action
      self.action_scores = tf.identity(self.q_network(), name='action_scores')
      tf.histogram_summary('action_scores', self.action_scores)

      # The action values for the G algorithm are computed differently.
      if self.algorithm == 'g':
        self.g_action_scores = self.action_scores + self.reward_scores

        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.g_action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.g_action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                                      self.num_actions)
      else:
        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                                      self.num_actions)

    tf.logging.info('Add estimating future rewards portion of graph')
    with tf.name_scope('estimating_future_rewards'):
      # The target q network is used to estimate the value of the best action at
      # the state resulting from the current action.
      self.next_action_scores = tf.stop_gradient(self.target_q_network())
      tf.histogram_summary('target_action_scores', self.next_action_scores)

      # Rewards are observed from the environment and are fed in later.
      self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')

      # Each algorithm is attempting to model future rewards with a different 
      # function.
      if self.algorithm == 'psi':
        self.target_vals = tf.reduce_logsumexp(self.next_action_scores,
                                       reduction_indices=[1,])
      elif self.algorithm == 'g':
        self.g_normalizer = tf.reduce_logsumexp(self.reward_scores, 
                                                reduction_indices=[1,])
        self.g_normalizer = tf.reshape(self.g_normalizer, [-1,1])
        self.g_normalizer = tf.tile(self.g_normalizer, [1,self.num_actions])
        self.g_action_scores = tf.sub(
          (self.next_action_scores + self.reward_scores), self.g_normalizer)
        self.target_vals = tf.reduce_logsumexp(self.g_action_scores, 
                                               reduction_indices=[1,])
      else:
        # Use default based on Q learning.
        self.target_vals = tf.reduce_max(self.next_action_scores, 
                                         reduction_indices=[1,])
        
      # Total rewards are the observed rewards plus discounted estimated future
      # rewards.
      self.future_rewards = self.rewards + self.discount_rate * self.target_vals

    tf.logging.info('Adding q value prediction portion of graph')
    with tf.name_scope('q_value_prediction'):
      # Action mask will be a one-hot encoding of the action the network
      # actually took.
      self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions),
                                        name='action_mask')
      self.masked_action_scores = tf.reduce_sum(self.action_scores *
                                                self.action_mask,
                                                reduction_indices=[1,])

      temp_diff = self.masked_action_scores - self.future_rewards

      # Prediction error is the mean squared error between the reward the
      # network actually received for a given action, and what it expected to
      # receive.
      self.prediction_error = tf.reduce_mean(tf.square(temp_diff))

      # Compute gradients.
      self.params = tf.trainable_variables()
      self.gradients = self.optimizer.compute_gradients(self.prediction_error)

      # Clip gradients.
      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, 5), var)

      for grad, var in self.gradients:
        tf.histogram_summary(var.name, var)
        if grad is not None:
          tf.histogram_summary(var.name + '/gradients', grad)

      # Backprop.
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    tf.logging.info('Adding target network update portion of graph')
    with tf.name_scope('target_network_update'):
      # Updates the target_q_network to be similar to the q_network based on
      # the target_network_update_rate.
      self.target_network_update = []
      for v_source, v_target in zip(self.q_network.variables(),
                                    self.target_q_network.variables()):
        # Equivalent to target = (1-alpha) * target + alpha * source
        update_op = v_target.assign_sub(self.target_network_update_rate *
                                        (v_target - v_source))
        self.target_network_update.append(update_op)
      self.target_network_update = tf.group(*self.target_network_update)

    tf.scalar_summary('prediction_error', self.prediction_error)

    self.summarize = tf.merge_all_summaries()
    self.no_op1 = tf.no_op()

  def get_random_action(self):
    """Sample an action uniformly at random.

    Returns:
      one-hot encoding of random action
    """
    act_idx = np.random.randint(0, self.num_actions - 1)
    return np.array(rl_tutor_ops.make_onehot([act_idx], 
                                             self.num_actions)).flatten()

  def reset_for_new_sequence(self):
    """Resets any model state variables for a new sequence.

    Should be overwritten in child class.
    """
    self.generated_seq_step = 0
    self.generated_seq = []

  def train(self, num_steps=10000, exploration_period=5000, enable_random=True, 
            verbose=False):
    """Main training function that allows model to act, collects reward, trains.

    Iterates a number of times, getting the model to act each time, saving the
    experience, and performing backprop.

    Args:
      num_steps: The number of training steps to execute.
      exploration_period: The number of steps over which the probability of
        exploring (taking a random action) is annealed from 1.0 to the model's
        random_action_probability.
      enable_random: If False, the model will not be able to act randomly /
        explore.
      verbose: If True, will output debugging statements
    """
    print "Evaluating initial model..."
    self.evaluate_model()

    self.actions_executed_so_far = 0

    if self.stochastic_observations:
      tf.logging.info('Using stochastic environment')

    self.reset_for_new_sequence()
    last_observation = self.prime_internal_models()

    for i in range(num_steps):
      # Experiencing observation, state, action, reward, new observation,
      # new state tuples, and storing them.
      state = np.array(self.q_network.state_value).flatten()
      reward_rnn_state = np.array(self.reward_rnn.state_value).flatten()

      if self.exploration_mode == 'boltzmann' or self.stochastic_observations:
        action, new_observation, reward_scores = self.action(
          last_observation, exploration_period, enable_random=enable_random,
          sample_next_obs=True)
      else:
        action, reward_scores = self.action(last_observation,
                                            exploration_period,
                                            enable_random=enable_random,
                                            sample_next_obs=False)
        new_observation = action
      new_state = np.array(self.q_network.state_value).flatten()
      new_reward_state = np.array(self.reward_rnn.state_value).flatten()

      if verbose:
        print "Action (in train func):", np.argmax(action)
        print "New obs (in train func):", np.argmax(new_observation)
        r_act = self.reward_from_reward_rnn_scores(action, reward_scores)
        print "reward_rnn output for action (in train func):", r_act
        r_obs = self.reward_from_reward_rnn_scores(new_observation, 
                                                   reward_scores)
        print "reward_rnn output for new obs (in train func):", r_obs
        r_diff = np.sum((reward_rnn_state - new_reward_state)**2)
        print "Diff between successive reward_rnn states:", r_diff
        s_diff = np.sum((new_state - new_reward_state)**2)
        print "Diff between reward_rnn state and q_network state:", s_diff

      reward = self.collect_reward(last_observation, new_observation, 
                                   reward_scores, verbose=verbose)

      self.store(last_observation, state, action, reward, new_observation,
                 new_state, new_reward_state)

      # Used to keep track of how the reward is changing over time.
      self.reward_last_n += reward

      # Used to keep track of the current musical composition and beat for
      # the reward functions.
      self.generated_seq.append(np.argmax(new_observation))
      self.generated_seq_step += 1

      if i > 0 and i % self.output_every_nth == 0:
        print "Evaluating model..."
        self.evaluate_model()
        self.save_model(self.algorithm)

        if self.algorithm == 'g':
          self.rewards_batched.append(
            self.domain_reward_last_n + self.data_reward_last_n)
        else:
          self.rewards_batched.append(self.reward_last_n)
        self.domain_rewards_batched.append(
          self.domain_reward_last_n)
        self.data_rewards_batched.append(self.data_reward_last_n)

        # Save a checkpoint.
        save_step = len(self.rewards_batched)*self.output_every_nth
        self.saver.save(self.session, self.save_path, global_step=save_step)

        r = self.reward_last_n
        tf.logging.info('Training iteration %s', i)
        tf.logging.info('\tReward for last %s steps: %s', 
                        self.output_every_nth, r)
        tf.logging.info('\t\tDomain reward: %s', 
                        self.domain_reward_last_n)
        tf.logging.info('\t\tReward RNN reward: %s', self.data_reward_last_n)
        
        print 'Training iteration', i
        print '\tReward for last', self.output_every_nth, 'steps:', r
        print '\t\tDomain reward:', self.domain_reward_last_n
        print '\t\tReward RNN reward:', self.data_reward_last_n

        if self.exploration_mode == 'egreedy':
          exploration_p = rl_tutor_ops.linear_annealing(
              self.actions_executed_so_far, exploration_period, 1.0,
              self.dqn_hparams.random_action_probability)
          tf.logging.info('\tExploration probability is %s', exploration_p)
          print '\tExploration probability is', exploration_p
        
        self.reward_last_n = 0
        self.domain_reward_last_n = 0
        self.data_reward_last_n = 0

      # Backprop.
      self.training_step()

      # Update current state as last state.
      last_observation = new_observation

      # Reset the state after each composition is complete.
      if self.is_end_of_sequence():
        if verbose: print "\nResetting composition!\n"
        self.reset_for_new_sequence()
        last_observation = self.prime_internal_models()

  def action(self, observation, exploration_period=0, enable_random=True,
             sample_next_obs=False):
    """Given an observation, runs the q_network to choose the current action.

    Does not backprop.

    Args:
      observation: A one-hot encoding of a single observation.
      exploration_period: The total length of the period the network will
        spend exploring, as set in the train function.
      enable_random: If False, the network cannot act randomly.
      sample_next_obs: If True, the next observation will be sampled from
        the softmax probabilities produced by the model, and passed back
        along with the action. If False, only the action is passed back.

    Returns:
      The action chosen, the reward_scores returned by the reward_rnn, and if 
      sample_next_obs is True, also returns the next observation.
    """
    assert len(observation.shape) == 1, 'Single observation only'

    self.actions_executed_so_far += 1

    if self.exploration_mode == 'egreedy':
      # Compute the exploration probability.
      exploration_p = rl_tutor_ops.linear_annealing(
          self.actions_executed_so_far, exploration_period, 1.0,
          self.dqn_hparams.random_action_probability)
    elif self.exploration_mode == 'boltzmann':
      enable_random = False
      sample_next_obs = True

    # Run the observation through the q_network.
    input_batch = np.reshape(observation,
                             (self.q_network.batch_size, 1, self.input_size))
    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    (action, action_softmax, self.q_network.state_value, 
    reward_scores, self.reward_rnn.state_value) = self.session.run(
      [self.predicted_actions, self.action_softmax,
       self.q_network.state_tensor, self.reward_scores, 
       self.reward_rnn.state_tensor],
      {self.q_network.melody_sequence: input_batch,
       self.q_network.initial_state: self.q_network.state_value,
       self.q_network.lengths: lengths,
       self.reward_rnn.melody_sequence: input_batch,
       self.reward_rnn.initial_state: self.reward_rnn.state_value,
       self.reward_rnn.lengths: lengths})

    # this is apparently not needed
    #if self.algorithm == 'psi':
    #  action_scores = np.exp(action_scores)

    reward_scores = np.reshape(reward_scores, (self.num_actions))
    action_softmax = np.reshape(action_softmax, (self.num_actions))
    action = np.reshape(action, (self.num_actions))

    if enable_random and random.random() < exploration_p:
      action = self.get_random_action()
      if sample_next_obs:
        return action, action, reward_scores
      else:
        return action, reward_scores
    else:
      if not sample_next_obs:
        return action, reward_scores
      else:
        obs = rl_tutor_ops.sample_softmax(action_softmax)
        next_obs = np.array(rl_tutor_ops.make_onehot([obs],
                                                   self.num_actions)).flatten()
        return action, next_obs, reward_scores

  def store(self, observation, state, action, reward, newobservation, newstate, 
            new_reward_state):
    """Stores an experience in the model's experience replay buffer.

    One experience consists of an initial observation and internal LSTM state,
    which led to the execution of an action, the receipt of a reward, and
    finally a new observation and a new LSTM internal state.

    Args:
      observation: A one hot encoding of an observed token.
      state: The internal state of the q_network MelodyRNN LSTM model.
      action: A one hot encoding of action taken by network.
      reward: Reward received for taking the action.
      newobservation: The next observation that resulted from the action.
        Unless stochastic_observations is True, the action and new
        observation will be the same.
      newstate: The internal state of the q_network MelodyRNN that is
        observed after taking the action.
      new_reward_state: The internal state of the reward_rnn network that is 
        observed after taking the action
    """
    if self.num_times_store_called % self.dqn_hparams.store_every_nth == 0:
      self.experience.append((observation, state, action, reward,
                              newobservation, newstate, new_reward_state))
    self.num_times_store_called += 1

  def training_step(self):
    """Backpropagate prediction error from a randomly sampled experience batch.

    A minibatch of experiences is randomly sampled from the model's experience
    replay buffer and used to update the weights of the q_network and
    target_q_network.
    """
    if self.num_times_train_called % self.dqn_hparams.train_every_nth == 0:
      if len(self.experience) < self.dqn_hparams.minibatch_size:
        return

      # Sample experience.
      samples = random.sample(range(len(self.experience)),
                              self.dqn_hparams.minibatch_size)
      samples = [self.experience[i] for i in samples]

      # Batch states.
      states = np.empty((len(samples), self.q_network.cell.state_size))
      new_states = np.empty((len(samples),
                             self.target_q_network.cell.state_size))
      reward_new_states = np.empty((len(samples), 
                                   self.reward_rnn.cell.state_size))
      observations = np.empty((len(samples), self.input_size))
      new_observations = np.empty((len(samples), self.input_size))
      action_mask = np.zeros((len(samples), self.num_actions))
      rewards = np.empty((len(samples),))
      lengths = np.full(len(samples), 1, dtype=int)

      for i, (o, s, a, r, new_o, new_s, reward_s) in enumerate(samples):
        observations[i, :] = o
        new_observations[i, :] = new_o
        states[i, :] = s
        new_states[i, :] = new_s
        action_mask[i, :] = a
        rewards[i] = r
        reward_new_states[i, :] = reward_s

      observations = np.reshape(observations,
                                (len(samples), 1, self.input_size))
      new_observations = np.reshape(new_observations,
                                    (len(samples), 1, self.input_size))

      calc_summaries = self.iteration % 100 == 0
      calc_summaries = calc_summaries and self.summary_writer is not None

      if self.algorithm == 'g':
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.reward_rnn.melody_sequence: new_observations,
            self.reward_rnn.initial_state: reward_new_states,
            self.reward_rnn.lengths: lengths,
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })
      else:
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })

      total_logs = (self.iteration * self.dqn_hparams.train_every_nth)
      if total_logs % self.output_every_nth == 0:
        self.target_val_list.append(np.mean(target_vals))

      self.session.run(self.target_network_update)

      if calc_summaries:
        self.summary_writer.add_summary(summary_str, self.iteration)

      self.iteration += 1

    self.num_times_train_called += 1

  def evaluate_model(self, num_trials=100, sample_next_obs=True):
    """Used to evaluate the rewards the model receives without exploring.

    Generates num_trials compositions and computes the reward_rnn and 
    domain rewards. Uses no exploration so rewards directly relate to the 
    model's policy. Stores result in internal variables.

    Args:
      num_trials: The number of compositions to use for evaluation.
      sample_next_obs: If True, the next token the model picks will be 
        sampled from its output distribution. If False, the model will 
        deterministically choose the token with maximum value.
    """

    data_rewards = [0] * num_trials
    domain_rewards = [0] * num_trials
    total_rewards = [0] * num_trials

    for t in range(num_trials):

      last_observation = self.prime_internal_models()
      self.reset_for_new_sequence()

      while not self.is_end_of_sequence():
        if sample_next_obs:
          action, new_observation, reward_scores = self.action(
              last_observation,
              0,
              enable_random=False,
              sample_next_obs=sample_next_obs)
        else:
          action, reward_scores = self.action(
              last_observation,
              0,
              enable_random=False,
              sample_next_obs=sample_next_obs)
          new_observation = action

        data_reward = self.reward_from_reward_rnn_scores(new_observation, 
                                                         reward_scores)
        domain_reward = self.reward_music_theory(new_observation)
        adjusted_domain_reward = self.reward_scaler * domain_reward
        total_reward = data_reward + adjusted_domain_reward

        data_rewards[t] = data_reward
        domain_rewards[t] = adjusted_domain_reward
        total_rewards[t] = total_reward

        self.generated_seq.append(np.argmax(new_observation))
        self.generated_seq_step += 1
        last_observation = new_observation

    self.eval_avg_reward.append(np.mean(total_rewards))
    self.eval_avg_data_reward.append(np.mean(data_rewards))
    self.eval_avg_domain_reward.append(np.mean(domain_rewards))

  def collect_reward(self, obs, action, reward_scores, verbose=False):
    """Collects reward from pre-trained RNN and domain-specific functions.

    Args:
      obs: A one-hot encoding of the observed token.
      action: A one-hot encoding of the chosen action.
      reward_scores: The value for each token as output by the reward_rnn.
      verbose: If True, additional logging statements about the reward after
        each function will be printed.
    Returns:
      Float reward value.
    """
    # Gets and saves log p(a|s) as output by reward_rnn.
    data_reward = self.reward_from_reward_rnn_scores(action, reward_scores)
    self.data_reward_last_n += data_reward

    reward = self.collect_domain_reward(obs, action, verbose=verbose)
    self.domain_reward_last_n += reward * self.reward_scaler

    if verbose:
      print 'Pre-trained RNN reward:', data_reward
      print 'Total domain reward:', self.reward_scaler * reward
      print ""
      
    if not self.domain_rewards_only:
      return reward * self.reward_scaler + data_reward
    else:
      return reward * self.reward_scaler 

  def reward_from_reward_rnn_scores(self, action, reward_scores):
    """Rewards based on probabilities learned from data by trained RNN

    Computes the reward_network's learned softmax probabilities. When used as
    rewards, allows the model to maintain information it learned from data.

    Args:
      obs: One-hot encoding of the observed token.
      action: One-hot encoding of the chosen action.
      state: Vector representing the internal state of the q_network.
    Returns:
      Float reward value.
    """
    action_token = np.argmax(action)
    normalization_constant = logsumexp(reward_scores)
    return reward_scores[action_token] - normalization_constant

  def get_reward_rnn_scores(self, observation, state):
    """Get token scores from the reward_rnn to use as a reward based on data.

    Runs the reward_rnn on an observation and initial state. Useful for
    maintaining the probabilities of the original LSTM model while training with
    reinforcement learning.

    Args:
      observation: One-hot encoding of the observed token.
      state: Vector representing the internal state of the target_q_network
        LSTM.

    Returns:
      Action scores produced by reward_rnn.
    """
    state = np.atleast_2d(state)

    input_batch = np.reshape(observation, (self.reward_rnn.batch_size, 1,
                                           self.num_actions))
    lengths = np.full(self.reward_rnn.batch_size, 1, dtype=int)

    rewards, = self.session.run(
        self.reward_scores,
        {self.reward_rnn.melody_sequence: input_batch,
         self.reward_rnn.initial_state: state,
         self.reward_rnn.lengths: lengths})
    return rewards

  def generate_music_sequence(self, title='rltutor_sample', 
    visualize_probs=False, prob_image_name=None, length=120, 
    most_probable=False):
    """Generates a music sequence with the current model, and saves it to MIDI.

    The resulting MIDI file is saved to the model's output_dir directory. The
    sequence is generated by sampling from the output probabilities at each
    timestep, and feeding the resulting note back in as input to the model.

    Args:
      title: The name that will be used to save the output MIDI file.
      visualize_probs: If True, the function will plot the softmax
        probabilities of the model for each note that occur throughout the
        sequence. Useful for debugging.
      prob_image_name: The name of a file in which to save the softmax
        probability image. If None, the image will simply be displayed.
      length: The max length of the sequence to be generated. If not set, will
        default to stopping when the is_end_of_sequence function returns True.
      most_probable: If True, instead of sampling each note in the sequence,
        the model will always choose the argmax, most probable note.
    """
    self.reset_for_new_sequence()
    next_obs = self.prime_internal_models()
    tf.logging.info('Priming with observation %s', np.argmax(next_obs))

    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    if visualize_probs:
      prob_image = np.zeros((self.input_size, length))
    
    i = 0
    while not self.is_end_of_sequence():
      input_batch = np.reshape(next_obs, (self.q_network.batch_size, 1,
                                          self.num_actions))
      if self.algorithm == 'g':
        (softmax, self.q_network.state_value, 
          self.reward_rnn.state_value) = self.session.run(
          [self.action_softmax, self.q_network.state_tensor, 
          self.reward_rnn.state_tensor],
          {self.q_network.melody_sequence: input_batch,
           self.q_network.initial_state: self.q_network.state_value,
           self.q_network.lengths: lengths,
           self.reward_rnn.melody_sequence: input_batch,
           self.reward_rnn.initial_state: self.reward_rnn.state_value,
           self.reward_rnn.lengths: lengths})
      else:
        softmax, self.q_network.state_value = self.session.run(
            [self.action_softmax, self.q_network.state_tensor],
            {self.q_network.melody_sequence: input_batch,
             self.q_network.initial_state: self.q_network.state_value,
             self.q_network.lengths: lengths})
      softmax = np.reshape(softmax, (self.num_actions))

      if visualize_probs:
        prob_image[:, i] = softmax #np.log(1.0 + softmax)

      if most_probable:
        sample = np.argmax(softmax)
      else:
        sample = rl_tutor_ops.sample_softmax(softmax)
      self.generated_seq.append(sample)
      self.generated_seq_step += 1
      next_obs = np.array(rl_tutor_ops.make_onehot([sample],
                                                 self.num_actions)).flatten()
      i += 1

    # Trim excess image columns
    prob_image = prob_image[:,0:i]
    print "trimmed prob image to dimensions:", np.shape(prob_image)

    tf.logging.info('Generated sequence: %s', self.generated_seq)
    print 'Generated sequence:', self.generated_seq

    self.render_sequence(self.generated_seq, title=title)

    if visualize_probs:
      tf.logging.info('Visualizing action selection probabilities:')
      plt.figure()
      plt.imshow(prob_image, interpolation='none', cmap='Reds')
      plt.ylabel('Action probability')
      plt.xlabel('Time step')
      plt.gca().invert_yaxis()
      if prob_image_name is not None:
        plt.savefig(self.output_dir + '/' + prob_image_name)
      else:
        plt.show()

  def save_model(self, name, directory=None):
    """Saves a checkpoint of the model and a .npz file with stored rewards.

    Args:
      name: String name to use for the checkpoint and rewards files.
      directory: Path to directory where the data will be saved. Defaults to
        self.output_dir if None is provided.
    """
    if directory is None:
      directory = self.output_dir

    save_loc = os.path.join(directory, name)
    self.saver.save(self.session, save_loc, 
                    global_step=len(self.rewards_batched)*self.output_every_nth)

    self.save_stored_rewards(name)

  def save_stored_rewards(self, file_name):
    """Saves the models stored rewards over time in a .npz file.

    Args:
      file_name: Name of the file that will be saved.
    """
    training_epochs = len(self.rewards_batched) * self.output_every_nth
    filename = os.path.join(self.output_dir, 
                            file_name + '-' + str(training_epochs))
    np.savez(filename,
             train_rewards=self.rewards_batched,
             train_domain_rewards=self.domain_rewards_batched,
             train_data_rewards=self.data_rewards_batched,
             eval_rewards=self.eval_avg_reward,
             eval_domain_rewards=self.eval_avg_domain_reward,
             eval_data_rewards=self.eval_avg_data_reward,
             target_val_list=self.target_val_list)

  def save_model_and_figs(self, name, directory=None):
    """Saves the model checkpoint, .npz file, and reward plots.

    Args:
      name: Name of the model that will be used on the images,
        checkpoint, and .npz files.
      directory: Path to directory where files will be saved. 
        If None defaults to self.output_dir.
    """

    self.save_model(name, directory=directory)
    self.plot_rewards(image_name='TrainRewards-' + name + '.eps', 
                      directory=directory)
    self.plot_evaluation(image_name='EvaluationRewards-' + name + '.eps', 
                         directory=directory)
    self.plot_target_vals(image_name='TargetVals-' + name + '.eps', 
                          directory=directory)

  def plot_rewards(self, image_name=None, directory=None):
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
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.rewards_batched))]
    plt.figure()
    plt.plot(x, self.rewards_batched)
    plt.plot(x, self.domain_rewards_batched)
    plt.plot(x, self.data_rewards_batched)
    plt.xlabel('Training epoch')
    plt.ylabel('Cumulative reward for last ' + str(reward_batch) + ' steps')
    plt.legend(['Total', 'Domain', 'Reward RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_evaluation(self, image_name=None, directory=None, start_at_epoch=0):
    """Plots the rewards received as the model was evaluated during training.

    If image_name is None, should be used in jupyter notebook. If 
    called outside of jupyter, execution of the program will halt and 
    a pop-up with the graph will appear. Execution will not continue 
    until the pop-up is closed.

    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
      start_at_epoch: Training epoch where the plot should begin.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.eval_avg_reward))]
    start_index = start_at_epoch / self.output_every_nth
    plt.figure()
    plt.plot(x[start_index:], self.eval_avg_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_domain_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_data_reward[start_index:])
    plt.xlabel('Training epoch')
    plt.ylabel('Average reward')
    plt.legend(['Total', 'Domain', 'Reward RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_target_vals(self, image_name=None, directory=None):
    """Plots the target values used to train the model over time.

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
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.target_val_list))]

    plt.figure()
    plt.plot(x,self.target_val_list)
    plt.xlabel('Training epoch')
    plt.ylabel('Target value')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def prime_internal_models(self):
    """Primes both internal models based on self.priming_mode.

    Returns:
      A one-hot encoding of the action output by the q_network to be used 
      as the initial observation. 
    """
    self.prime_internal_model(self.target_q_network)
    self.prime_internal_model(self.reward_rnn)
    next_obs = self.prime_internal_model(self.q_network)
    return next_obs

  def restore_from_directory(self, directory=None, checkpoint_name=None, 
                             reward_file_name=None):
    """Restores this model from a saved checkpoint.

    Args:
      directory: Path to directory where checkpoint is located. If 
        None, defaults to self.output_dir.
      checkpoint_name: The name of the checkpoint within the 
        directory.
      reward_file_name: The name of the .npz file where the stored
        rewards are saved. If None, will not attempt to load stored
        rewards.
    """
    if directory is None:
      directory = self.output_dir

    if checkpoint_name is not None:
      checkpoint_file = os.path.join(directory, checkpoint_name)
    else:
      print "directory", directory
      checkpoint_file = tf.train.latest_checkpoint(directory)

    if checkpoint_file is None:
      print "Error! Cannot locate checkpoint in the directory"
      return
    print "Attempting to restore from checkpoint", checkpoint_file

    self.saver.restore(self.session, checkpoint_file)

    if reward_file_name is not None:
      npz_file_name = os.path.join(directory, reward_file_name)
      print "Attempting to load saved reward values from file", npz_file_name
      npz_file = np.load(npz_file_name)

      self.rewards_batched = npz_file['train_rewards']
      self.domain_rewards_batched = npz_file['train_domain_rewards']
      self.data_rewards_batched = npz_file['train_data_rewards']
      self.eval_avg_reward = npz_file['eval_rewards']
      self.eval_avg_domain_reward = npz_file['eval_domain_rewards']
      self.eval_avg_data_reward = npz_file['eval_data_rewards']
      self.target_val_list = npz_file['target_val_list']