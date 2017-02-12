"""Defines a Deep Q Network (DQN) with augmented reward to create melodies 
by using reinforcement learning to fine-tune a trained Note RNN according
to some music theory rewards. 

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

import note_rnn_loader
import rl_tutor_ops
import rl_tuner_eval_metrics
from rl_tutor import RLTutor

# Note values of pecial actions.
NOTE_OFF = 0
NO_EVENT = 1

# Training data sequences are limited to this length, so the padding queue pads
# to this length.
TRAIN_SEQUENCE_LENGTH = 192

def reload_files():
  """Used to reload the imported dependency files (necessary for jupyter 
  notebooks).
  """
  reload(note_rnn_loader)
  reload(rl_tutor_ops)
  reload(rl_tuner_eval_metrics)


class RLTuner(RLTutor):
  """Implements a recurrent DQN designed to produce melody sequences."""

  def __init__(self, output_dir,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='music_theory_all',
               reward_scaler=1.0,
               exploration_mode='egreedy',
               priming_mode='random_note',
               stochastic_observations=False,
               algorithm='q',      

               # Pre-trained RNN to load and tune
               note_rnn_checkpoint_dir=None,
               note_rnn_checkpoint_file=None,
               note_rnn_type='default',
               note_rnn_hparams=None,

               # Logistics.
               input_size=rl_tutor_ops.NUM_CLASSES,
               num_actions=rl_tutor_ops.NUM_CLASSES,
               save_name='rl_tuner.ckpt',
               output_every_nth=1000,
               summary_writer=None,
               initialize_immediately=True,

               # Settings specific to RLTuner
               num_notes_in_melody=32,
               midi_primer=None,
               training_file_list=None,):
    """Initializes the MelodyQNetwork class.

    Args:
      output_dir: Where the model will save its compositions (midi files).
      dqn_hparams: A tf_lib.hparams() object containing the hyperparameters of 
        the DQN algorithm, including minibatch size, exploration probability, 
        etc.
      reward_mode: Controls which reward function can be applied. There are
        several, including 'scale', which teaches the model to play a scale,
        and of course 'music_theory_all', which is a music-theory-based reward
        function composed of other functions. 
      reward_scaler: Controls the emphasis placed on the music theory rewards. 
        This value is the inverse of 'c' in the academic paper.
      exploration_mode: can be 'egreedy' which is an epsilon greedy policy, or
        it can be 'boltzmann', in which the model will sample from its output
        distribution to choose the next action.
      priming_mode: Each time the model begins a new composition, it is primed
        with either a random note ('random_note'), a random MIDI file from the
        training data ('random_midi'), or a particular MIDI file
        ('single_midi').
      stochastic_observations: If False, the note that the model chooses to
        play next (the argmax of its softmax probabilities) deterministically
        becomes the next note it will observe. If True, the next observation
        will be sampled from the model's softmax output.
      algorithm: can be 'default', 'psi', 'g' or 'pure_rl', for different 
        learning algorithms
      note_rnn_checkpoint_dir: The directory from which the internal 
        NoteRNNLoader will load its checkpointed LSTM.
      note_rnn_checkpoint_file: A checkpoint file to use in case one cannot be
        found in the note_rnn_checkpoint_dir.
      note_rnn_type: If 'default', will use the basic LSTM described in the 
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      note_rnn_hparams: A tf.HParams object which defines the hyper parameters
        used to train the MelodyRNN model that will be loaded from a checkpoint.
      input_size: the size of the one-hot vector encoding a note that is input
        to the model.
      num_actions: The size of the one-hot vector encoding a note that is
        output by the model.
      save_name: Name the model will use to save checkpoints.
      output_every_nth: How many training steps before the model will print
        an output saying the cumulative reward, and save a checkpoint.
      summary_writer: A tf.train.SummaryWriter used to log metrics.
      initialize_immediately: if True, the class will instantiate its component
        MelodyRNN networks and build the graph in the constructor.
      num_notes_in_melody: The length of a composition of the model
      midi_primer: A midi file that can be used to prime the model if
        priming_mode is set to 'single_midi'.
      training_file_list: A list of paths to tfrecord files containing melody 
        training data. This is necessary to use the 'random_midi' priming mode.
    """
    print "In child class RL Tuner"

    self.num_notes_in_melody = num_notes_in_melody
    print "num notes in melody came in as:", num_notes_in_melody
    print "num notes in melody in class as:", self.num_notes_in_melody
    self.midi_primer = midi_primer
    self.training_file_list = training_file_list

    if priming_mode == 'single_midi' and midi_primer is None:
      tf.logging.fatal('A midi primer file is required when using'
                       'the single_midi priming mode.')

    if note_rnn_checkpoint_dir is None or note_rnn_checkpoint_dir == '':
      print 'Retrieving checkpoint of Note RNN from Magenta download server.'
      urllib.urlretrieve(
        'http://download.magenta.tensorflow.org/models/rl_tuner_note_rnn.ckpt', 
        'note_rnn.ckpt')
      note_rnn_checkpoint_dir = os.getcwd()
      note_rnn_checkpoint_file = os.path.join(os.getcwd(), 
                                                  'note_rnn.ckpt')

    if note_rnn_hparams is None:
      if note_rnn_type == 'basic_rnn':
        note_rnn_hparams = rl_tutor_ops.basic_rnn_hparams()
      else:
        note_rnn_hparams = rl_tutor_ops.default_hparams()

    RLTutor.__init__(self, output_dir, dqn_hparams=dqn_hparams, 
      reward_mode=reward_mode, reward_scaler=reward_scaler, 
      exploration_mode=exploration_mode, priming_mode=priming_mode,
      stochastic_observations=stochastic_observations, algorithm=algorithm,
      rnn_checkpoint_dir=note_rnn_checkpoint_dir, 
      rnn_checkpoint_file=note_rnn_checkpoint_file, 
      rnn_type=note_rnn_type, rnn_hparams=note_rnn_hparams, input_size=input_size,
      num_actions=num_actions, midi_primer=midi_primer, save_name=save_name,
      output_every_nth=output_every_nth, training_file_list=training_file_list,
      summary_writer=summary_writer, initialize_immediately=initialize_immediately)

    # State variables needed by reward functions.
    self.composition_direction = 0
    self.leapt_from = None  # stores the note at which composition leapt
    self.steps_since_last_leap = 0

    if self.priming_mode == 'random_midi':
      tf.logging.info('Getting priming melodies')
      self.get_priming_melodies()

  def initialize_internal_models(self, ):
    """Initializes internal RNN models: q_network, target_q_network, reward_rnn.

    Adds the graphs of the internal RNN models to this graph, by having a separate
    function for this rather than doing it in the constructor, it allows classes
    inheriting from this class to define their own type of q_network.
    """
    # Add internal networks to the graph.
    tf.logging.info('Initializing q network')
    self.q_network = note_rnn_loader.NoteRNNLoader(
      self.graph, 'q_network',
      self.rnn_checkpoint_dir,
      midi_primer=self.midi_primer,
      training_file_list=
      self.training_file_list,
      checkpoint_file=
      self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      note_rnn_type=self.rnn_type)

    tf.logging.info('Initializing target q network')
    self.target_q_network = note_rnn_loader.NoteRNNLoader(
      self.graph,
      'target_q_network',
      self.rnn_checkpoint_dir,
      midi_primer=self.midi_primer,
      training_file_list=
      self.training_file_list,
      checkpoint_file=
      self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      note_rnn_type=self.rnn_type)

    tf.logging.info('Initializing reward network')
    self.reward_rnn = note_rnn_loader.NoteRNNLoader(
      self.graph, 'reward_rnn',
      self.rnn_checkpoint_dir,
      midi_primer=self.midi_primer,
      training_file_list=
      self.training_file_list,
      checkpoint_file=
      self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      note_rnn_type=self.rnn_type)

    tf.logging.info('Q network cell: %s', self.q_network.cell)

  def get_priming_melodies(self):
    """Runs a batch of training data through MelodyRNN model.

    If the priming mode is 'random_midi', priming the q-network requires a
    random training melody. Therefore this function runs a batch of data from
    the training directory through the internal model, and the resulting
    internal states of the LSTM are stored in a list. The next note in each
    training melody is also stored in a corresponding list called
    'priming_notes'. Therefore, to prime the model with a random melody, it is
    only necessary to select a random index from 0 to batch_size-1 and use the
    hidden states and note at that index as input to the model.
    """
    (next_note_softmax,
     self.priming_states, lengths) = self.q_network.run_training_batch()

    # Get the next note that was predicted for each priming melody to be used
    # in priming.
    self.priming_notes = [0] * len(lengths)
    for i in range(len(lengths)):
      # Each melody has TRAIN_SEQUENCE_LENGTH outputs, but the last note is
      # actually stored at lengths[i]. The rest is padding.
      start_i = i * TRAIN_SEQUENCE_LENGTH
      end_i = start_i + lengths[i] - 1
      end_softmax = next_note_softmax[end_i, :]
      self.priming_notes[i] = np.argmax(end_softmax)

    tf.logging.info('Stored priming notes: %s', self.priming_notes)

  def prime_internal_model(self, model):
    """Prime an internal model such as the q_network based on priming mode.

    Args:
      model: The internal model that should be primed. 

    Returns:
      The first observation to feed into the model.
    """
    model.state_value = model.get_zero_state()

    if self.priming_mode == 'random_midi':
      priming_idx = np.random.randint(0, len(self.priming_states))
      model.state_value = np.reshape(
          self.priming_states[priming_idx, :],
          (1, model.cell.state_size))
      priming_note = self.priming_notes[priming_idx]
      next_obs = np.array(
          rl_tutor_ops.make_onehot([priming_note], self.num_actions)).flatten()
      tf.logging.debug(
        'Feeding priming state for midi file %s and corresponding note %s',
        priming_idx, priming_note)
    elif self.priming_mode == 'single_midi':
      model.prime_model()
      next_obs = model.priming_note
    elif self.priming_mode == 'random_note':
      next_obs = self.get_random_action()
    else:
      tf.logging.warn('Error! Invalid priming mode. Priming with random note')
      next_obs = self.get_random_action()

    return next_obs

  def reset_for_new_sequence(self):
    """Starts the models internal composition over at beat 0, with no notes.

    Also resets statistics about whether the composition is in the middle of a
    melodic leap.
    """
    self.generated_seq_step = 0
    self.generated_seq = []
    self.composition_direction = 0
    self.leapt_from = None
    self.steps_since_last_leap = 0

  def is_end_of_sequence(self):
    """Returns true if the sequence generated by the model has reached its end.

    Should be overwritten by child class.
    """
    if (self.generated_seq_step + 1 == self.num_notes_in_melody) or (
      len(self.generated_seq) == self.num_notes_in_melody):
      return True
    return False

  def render_sequence(self, generated_seq, title='rltuner_sample'):
    """Renders a generated melody sequence into an audio MIDI file. 

    Args:
      generated_seq: A list of integer note/action values.
      title: A title to use in the sequence filename.
    """
    melody = mlib.Melody(rl_tutor_ops.decoder(generated_seq, 
                                              self.q_network.transpose_amount))

    sequence = melody.to_sequence(qpm=rl_tutor_ops.DEFAULT_QPM)
    filename = rl_tutor_ops.get_next_file_name(self.output_dir, title, 'mid')
    midi_io.sequence_proto_to_midi_file(sequence, filename)

    tf.logging.info('Wrote a melody to %s', self.output_dir)


  def collect_domain_reward(self, obs, action, verbose=False):
    """Calls whatever reward function is indicated in the reward_mode field.

    New reward functions can be written and called from here. Note that the
    reward functions can make use of the musical composition that has been
    played so far, which is stored in self.generated_seq. Some reward functions
    are made up of many smaller functions, such as those related to music
    theory.

    Args:
      obs: A one-hot encoding of the observed note.
      action: A one-hot encoding of the chosen action.
      verbose: If True, additional logging statements about the reward after
        each function will be printed.
    Returns:
      Float reward value.
    """

    if self.reward_mode == 'scale':
      # Makes the model play a scale (defaults to c major).
      reward = self.reward_scale(obs, action)
    elif self.reward_mode == 'key':
      # Makes the model play within a key.
      reward = self.reward_key_distribute_prob(action)
    elif self.reward_mode == 'key_and_tonic':
      # Makes the model play within a key, while starting and ending on the
      # tonic note.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
    elif self.reward_mode == 'non_repeating':
      # The model can play any composition it wants, but receives a large
      # negative reward for playing the same note repeatedly.
      reward = self.reward_non_repeating(action)
    elif self.reward_mode == 'music_theory_random':
      # The model receives reward for playing in key, playing tonic notes,
      # and not playing repeated notes. However the rewards it receives are
      # uniformly distributed over all notes that do not violate these rules.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)
    elif self.reward_mode == 'music_theory_basic':
      # As above, the model receives reward for playing in key, tonic notes
      # at the appropriate times, and not playing repeated notes. However, the
      # rewards it receives are based on the note probabilities learned from
      # data in the original model.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)
    elif self.reward_mode == 'music_theory_basic_plus_variety':
      # Uses the same reward function as above, but adds a penalty for
      # compositions with a high autocorrelation (aka those that don't have
      # sufficient variety).
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)
      reward += self.reward_penalize_autocorrelation(action)
    elif self.reward_mode == 'preferred_intervals':
      reward = self.reward_preferred_intervals(action)
    elif self.reward_mode == 'music_theory_all':
      reward = self.reward_music_theory(action, verbose=verbose)
    else:
      tf.logging.fatal('ERROR! Not a valid reward mode. Cannot compute reward')

    return reward


  def reward_music_theory(self, action, verbose=False):
    reward = self.reward_key(action)
    if verbose:
      print 'Key:', reward
    prev_reward = reward

    reward += self.reward_tonic(action)
    if verbose and reward != prev_reward:
      print 'Tonic:', reward
    prev_reward = reward

    reward += self.reward_penalize_repeating(action)
    if verbose and reward != prev_reward:
      print 'Penalize repeating:', reward
    prev_reward = reward

    reward += self.reward_penalize_autocorrelation(action)
    if verbose and reward != prev_reward:
      print 'Penalize autocorr:', reward
    prev_reward = reward

    reward += self.reward_motif(action)
    if verbose and reward != prev_reward:
      print 'Reward motif:', reward
    prev_reward = reward

    reward += self.reward_repeated_motif(action)
    if verbose and reward != prev_reward:
      print 'Reward repeated motif:', reward
    prev_reward = reward

    # New rewards based on Gauldin's book, "A Practical Approach to Eighteenth
    # Century Counterpoint"
    reward += self.reward_preferred_intervals(action)
    if verbose and reward != prev_reward:
      print 'Reward preferred_intervals:', reward
    prev_reward = reward

    reward += self.reward_leap_up_back(action)
    if verbose and reward != prev_reward:
      print 'Reward leap up back:', reward
    prev_reward = reward

    reward += self.reward_high_low_unique(action)
    if verbose and reward != prev_reward:
      print 'Reward high low unique:', reward

    return reward

  def random_reward_shift_to_mean(self, reward):
    """Modifies reward by a small random values s to pull it towards the mean.

    If reward is above the mean, s is subtracted; if reward is below the mean,
    s is added. The random value is in the range 0-0.2. This function is helpful
    to ensure that the model does not become too certain about playing a
    particular note.

    Args:
      reward: A reward value that has already been computed by another reward
        function.
    Returns:
      Original float reward value modified by scaler.
    """
    s = np.random.randint(0, 2) * .1
    if reward > .5:
      reward -= s
    else:
      reward += s
    return reward

  def reward_scale(self, obs, action, scale=None):
    """Reward function that trains the model to play a scale.

    Gives rewards for increasing notes, notes within the desired scale, and two
    consecutive notes from the scale.

    Args:
      obs: A one-hot encoding of the observed note.
      action: A one-hot encoding of the chosen action.
      scale: The scale the model should learn. Defaults to C Major if not
        provided.
    Returns:
      Float reward value.
    """

    if scale is None:
      scale = rl_tutor_ops.C_MAJOR_SCALE

    obs = np.argmax(obs)
    action = np.argmax(action)
    reward = 0
    if action == 1:
      reward += .1
    if action > obs and action < obs + 3:
      reward += .05

    if action in scale:
      reward += .01
      if obs in scale:
        action_pos = scale.index(action)
        obs_pos = scale.index(obs)
        if obs_pos == len(scale) - 1 and action_pos == 0:
          reward += .8
        elif action_pos == obs_pos + 1:
          reward += .8

    return reward

  def reward_key_distribute_prob(self, action, key=None):
    """Reward function that rewards the model for playing within a given key.

    Any note within the key is given equal reward, which can cause the model to
    learn random sounding compositions.

    Args:
      action: One-hot encoding of the chosen action.
      key: The numeric values of notes belonging to this key. Defaults to C
        Major if not provided.
    Returns:
      Float reward value.
    """
    if key is None:
      key = rl_tutor_ops.C_MAJOR_KEY

    reward = 0

    action_note = np.argmax(action)
    if action_note in key:
      num_notes_in_key = len(key)
      extra_prob = 1.0 / num_notes_in_key

      reward = extra_prob

    return reward

  def reward_key(self, action, penalty_amount=-1.0, key=None):
    """Applies a penalty for playing notes not in a specific key.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_amount: The amount the model will be penalized if it plays
        a note outside the key.
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      Float reward value.
    """
    if key is None:
      key = rl_tutor_ops.C_MAJOR_KEY

    reward = 0

    action_note = np.argmax(action)
    if action_note not in key:
      reward = penalty_amount

    return reward

  def reward_tonic(self, action, tonic_note=rl_tutor_ops.C_MAJOR_TONIC, 
                   reward_amount=3.0):
    """Rewards for playing the tonic note at the right times.

    Rewards for playing the tonic as the first note of the first bar, and the
    first note of the final bar. 

    Args:
      action: One-hot encoding of the chosen action.
      tonic_note: The tonic/1st note of the desired key.
      reward_amount: The amount the model will be awarded if it plays the 
        tonic note at the right time. 
    Returns:
      Float reward value.
    """
    action_note = np.argmax(action)
    first_note_of_final_bar = self.num_notes_in_melody - 4

    if self.generated_seq_step == 0 or self.generated_seq_step == first_note_of_final_bar:
      if action_note == tonic_note:
        return reward_amount
    elif self.generated_seq_step == first_note_of_final_bar + 1:
      if action_note == NO_EVENT:
          return reward_amount
    elif self.generated_seq_step > first_note_of_final_bar + 1:
      if action_note == NO_EVENT or action_note == NOTE_OFF:
        return reward_amount
    return 0.0

  def reward_non_repeating(self, action):
    """Rewards the model for not playing the same note over and over.

    Penalizes the model for playing the same note repeatedly, although more
    repeititions are allowed if it occasionally holds the note or rests in
    between. Reward is uniform when there is no penalty.

    Args:
      action: One-hot encoding of the chosen action.
    Returns:
      Float reward value.
    """
    penalty = self.reward_penalize_repeating(action)
    if penalty >= 0:
      return .1

  def detect_repeating_notes(self, action_note):
    """Detects whether the note played is repeating previous notes excessively.

    Args:
      action_note: An integer representing the note just played.
    Returns:
      True if the note just played is excessively repeated, False otherwise.
    """
    num_repeated = 0
    contains_held_notes = False
    contains_breaks = False

    # Note that the current action yas not yet been added to the composition
    for i in xrange(len(self.generated_seq)-1, -1, -1):
      if self.generated_seq[i] == action_note:
        num_repeated += 1
      elif self.generated_seq[i] == NOTE_OFF:
        contains_breaks = True
      elif self.generated_seq[i] == NO_EVENT:
        contains_held_notes = True
      else:
        break

    if action_note == NOTE_OFF and num_repeated > 1:
      return True
    elif not contains_held_notes and not contains_breaks:
      if num_repeated > 4:
        return True
    elif contains_held_notes or contains_breaks:
      if num_repeated > 6:
        return True
    else:
      if num_repeated > 8:
        return True

    return False

  def reward_penalize_repeating(self,
                                action,
                                penalty_amount=-100.0):
    """Sets the previous reward to 0 if the same is played repeatedly.

    Allows more repeated notes if there are held notes or rests in between. If
    no penalty is applied will return the previous reward.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_amount: The amount the model will be penalized if it plays
        repeating notes.
    Returns:
      Previous reward or 'penalty_amount'.
    """
    action_note = np.argmax(action)
    is_repeating = self.detect_repeating_notes(action_note)
    if is_repeating:
      return penalty_amount
    else:
      return 0.0

  def reward_penalize_autocorrelation(self,
                                      action,
                                      penalty_weight=3.0):
    """Reduces the previous reward if the composition is highly autocorrelated.

    Penalizes the model for creating a composition that is highly correlated
    with itself at lags of 1, 2, and 3 beats previous. This is meant to
    encourage variety in compositions.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_weight: The default weight which will be multiplied by the sum
        of the autocorrelation coefficients, and subtracted from prev_reward.
    Returns:
      Float reward value.
    """
    composition = self.generated_seq + [np.argmax(action)]
    lags = [1, 2, 3]
    sum_penalty = 0
    for lag in lags:
      coeff = rl_tutor_ops.autocorrelate(composition, lag=lag)
      if not np.isnan(coeff):
        if np.abs(coeff) > 0.15:
          sum_penalty += np.abs(coeff) * penalty_weight
    return -sum_penalty

  def detect_last_motif(self, composition=None, bar_length=8):
    """Detects if a motif was just played and if so, returns it.

    A motif should contain at least three distinct notes that are not note_on
    or note_off, and occur within the course of one bar.

    Args:
      composition: The composition in which the function will look for a
        recent motif. Defaults to the model's composition.
      bar_length: The number of notes in one bar.
    Returns:
      None if there is no motif, otherwise the motif in the same format as the
      composition.
    """
    if composition is None:
      composition = self.generated_seq

    if len(composition) < bar_length:
      return None, 0

    last_bar = composition[-bar_length:]

    actual_notes = [a for a in last_bar if a != NO_EVENT and a != NOTE_OFF]
    num_unique_notes = len(set(actual_notes))
    if num_unique_notes >= 3:
      return last_bar, num_unique_notes
    else:
      return None, num_unique_notes

  def reward_motif(self, action, reward_amount=3.0):
    """Rewards the model for playing any motif.

    Motif must have at least three distinct notes in the course of one bar.
    There is a bonus for playing more complex motifs; that is, ones that involve
    a greater number of notes.

    Args:
      action: One-hot encoding of the chosen action.
      reward_amount: The amount that will be returned if the last note belongs
        to a motif.
    Returns:
      Float reward value.
    """

    composition = self.generated_seq + [np.argmax(action)]
    motif, num_notes_in_motif = self.detect_last_motif(composition=composition)
    if motif is not None:
      motif_complexity_bonus = max((num_notes_in_motif - 3)*.3, 0)
      return reward_amount + motif_complexity_bonus
    else:
      return 0.0

  def detect_repeated_motif(self, action, bar_length=8):
    """Detects whether the last motif played repeats an earlier motif played.

    Args:
      action: One-hot encoding of the chosen action.
      bar_length: The number of beats in one bar. This determines how many beats
        the model has in which to play the motif.
    Returns:
      True if the note just played belongs to a motif that is repeated. False
      otherwise.
    """
    composition = self.generated_seq + [np.argmax(action)]
    if len(composition) < bar_length:
      return False, None

    motif, _ = self.detect_last_motif(
        composition=composition, bar_length=bar_length)
    if motif is None:
      return False, None

    prev_composition = self.generated_seq[:-(bar_length-1)]

    # Check if the motif is in the previous composition.
    for i in range(len(prev_composition) - len(motif) + 1):
      for j in range(len(motif)):
        if prev_composition[i + j] != motif[j]:
          break
      else:
        return True, motif
    return False, None

  def reward_repeated_motif(self,
                            action,
                            bar_length=8,
                            reward_amount=4.0):
    """Adds a big bonus to previous reward if the model plays a repeated motif.

    Checks if the model has just played a motif that repeats an ealier motif in
    the composition.

    There is also a bonus for repeating more complex motifs.

    Args:
      action: One-hot encoding of the chosen action.
      bar_length: The number of notes in one bar.
      reward_amount: The amount that will be added to the reward if the last
        note belongs to a repeated motif.
    Returns:
      Float reward value.
    """
    is_repeated, motif = self.detect_repeated_motif(action, bar_length)
    if is_repeated:
      actual_notes = [a for a in motif if a != NO_EVENT and a != NOTE_OFF]
      num_notes_in_motif = len(set(actual_notes))
      motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
      return reward_amount + motif_complexity_bonus
    else:
      return 0.0

  def detect_sequential_interval(self, action, key=None, verbose=False):
    """Finds the melodic interval between the action and the last note played.

    Uses constants to represent special intervals like rests.

    Args:
      action: One-hot encoding of the chosen action
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      An integer value representing the interval, or a constant value for
      special intervals.
    """
    if not self.generated_seq:
      return 0, None, None

    prev_note = self.generated_seq[-1]
    action_note = np.argmax(action)

    c_major = False
    if key is None:
      key = rl_tutor_ops.C_MAJOR_KEY
      c_notes = [2, 14, 26]
      g_notes = [9, 21, 33]
      e_notes = [6, 18, 30]
      c_major = True
      tonic_notes = [2, 14, 26]
      fifth_notes = [9, 21, 33]

    # get rid of non-notes in prev_note
    prev_note_index = len(self.generated_seq) - 1
    while (prev_note == NO_EVENT or
           prev_note == NOTE_OFF) and prev_note_index >= 0:
      prev_note = self.generated_seq[prev_note_index]
      prev_note_index -= 1
    if prev_note == NOTE_OFF or prev_note == NO_EVENT:
      if verbose: print "action_note:", action_note, "prev_note:", prev_note
      return 0, action_note, prev_note

    if verbose: print "action_note:", action_note, "prev_note:", prev_note

    # get rid of non-notes in action_note
    if action_note == NO_EVENT:
      if prev_note in tonic_notes or prev_note in fifth_notes:
        return (rl_tutor_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH, 
                action_note, prev_note)
      else:
        return rl_tutor_ops.HOLD_INTERVAL, action_note, prev_note
    elif action_note == NOTE_OFF:
      if prev_note in tonic_notes or prev_note in fifth_notes:
        return (rl_tutor_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH, 
                action_note, prev_note)
      else:
        return rl_tutor_ops.REST_INTERVAL, action_note, prev_note

    interval = abs(action_note - prev_note)

    if c_major and interval == rl_tutor_ops.FIFTH and (
        prev_note in c_notes or prev_note in g_notes):
      return rl_tutor_ops.IN_KEY_FIFTH, action_note, prev_note
    if c_major and interval == rl_tutor_ops.THIRD and (
        prev_note in c_notes or prev_note in e_notes):
      return rl_tutor_ops.IN_KEY_THIRD, action_note, prev_note

    return interval, action_note, prev_note

  def reward_preferred_intervals(self, action, scaler=5.0, key=None, 
    verbose=False):
    """Dispenses reward based on the melodic interval just played.

    Args:
      action: One-hot encoding of the chosen action.
      scaler: This value will be multiplied by all rewards in this function.
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      Float reward value.
    """
    interval, _, _ = self.detect_sequential_interval(action, key, 
                                                     verbose=verbose)
    if verbose: print "interval:", interval

    if interval == 0:  # either no interval or involving uninteresting rests
      if verbose: print "no interval or uninteresting"
      return 0.0

    reward = 0.0

    # rests can be good
    if interval == rl_tutor_ops.REST_INTERVAL:
      reward = 0.05
      if verbose: print "rest interval"
    if interval == rl_tutor_ops.HOLD_INTERVAL:
      reward = 0.075
    if interval == rl_tutor_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
      reward = 0.15
      if verbose: print "rest interval after 1st or 5th"
    if interval == rl_tutor_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH:
      reward = 0.3

    # large leaps and awkward intervals bad
    if interval == rl_tutor_ops.SEVENTH:
      reward = -0.3
      if verbose: print "7th"
    if interval > rl_tutor_ops.OCTAVE:
      reward = -1.0
      if verbose: print "More than octave"

    # common major intervals are good
    if interval == rl_tutor_ops.IN_KEY_FIFTH:
      reward = 0.1
      if verbose: print "in key 5th"
    if interval == rl_tutor_ops.IN_KEY_THIRD:
      reward = 0.15
      if verbose: print "in key 3rd"

    # smaller steps are generally preferred
    if interval == rl_tutor_ops.THIRD:
      reward = 0.09
      if verbose: print "3rd"
    if interval == rl_tutor_ops.SECOND:
      reward = 0.08
      if verbose: print "2nd"
    if interval == rl_tutor_ops.FOURTH:
      reward = 0.07
      if verbose: print "4th"

    # larger leaps not as good, especially if not in key
    if interval == rl_tutor_ops.SIXTH:
      reward = 0.05
      if verbose: print "6th"
    if interval == rl_tutor_ops.FIFTH:
      reward = 0.02
      if verbose: print "5th"

    if verbose: print "interval reward", reward * scaler
    return reward * scaler

  def detect_high_unique(self, composition):
    """Checks a composition to see if the highest note within it is repeated.

    Args:
      composition: A list of integers representing the notes in the piece.
    Returns:
      True if the lowest note was unique, False otherwise.
    """
    max_note = max(composition)
    if list(composition).count(max_note) == 1:
      return True
    else:
      return False

  def detect_low_unique(self, composition):
    """Checks a composition to see if the lowest note within it is repeated.

    Args:
      composition: A list of integers representing the notes in the piece.
    Returns:
      True if the lowest note was unique, False otherwise.
    """
    no_special_events = [x for x in composition
                         if x != NO_EVENT and x != NOTE_OFF]
    if no_special_events:
      min_note = min(no_special_events)
      if list(composition).count(min_note) == 1:
        return True
    return False

  def reward_high_low_unique(self, action, reward_amount=3.0):
    """Evaluates if highest and lowest notes in composition occurred once.

    Args:
      action: One-hot encoding of the chosen action.
      reward_amount: Amount of reward that will be given for the highest note
        being unique, and again for the lowest note being unique.
    Returns:
      Float reward value.
    """
    if len(self.generated_seq) + 1 != self.num_notes_in_melody:
      return 0.0

    composition = np.array(self.generated_seq)
    composition = np.append(composition, np.argmax(action))

    reward = 0.0

    if self.detect_high_unique(composition):
      reward += reward_amount

    if self.detect_low_unique(composition):
      reward += reward_amount

    return reward

  def detect_leap_up_back(self, action, steps_between_leaps=6, verbose=False):
    """Detects when the composition takes a musical leap, and if it is resolved.

    When the composition jumps up or down by an interval of a fifth or more,
    it is a 'leap'. The model then remembers that is has a 'leap direction'. The
    function detects if it then takes another leap in the same direction, if it
    leaps back, or if it gradually resolves the leap.

    Args:
      action: One-hot encoding of the chosen action.
      steps_between_leaps: Leaping back immediately does not constitute a
        satisfactory resolution of a leap. Therefore the composition must wait
        'steps_between_leaps' beats before leaping back.
      verbose: If True, the model will output statements about whether it has
        detected a leap.
    Returns:
      0 if there is no leap, 'LEAP_RESOLVED' if an existing leap has been
      resolved, 'LEAP_DOUBLED' if 2 leaps in the same direction were made.
    """
    if not self.generated_seq:
      return 0

    outcome = 0

    interval, action_note, prev_note = self.detect_sequential_interval(action)

    if action_note == NOTE_OFF or action_note == NO_EVENT:
      self.steps_since_last_leap += 1
      if verbose:
        tf.logging.info('Rest, adding to steps since last leap. It is'
                     'now: %s', self.steps_since_last_leap)
      return 0

    # detect if leap
    if interval >= rl_tutor_ops.FIFTH or interval == rl_tutor_ops.IN_KEY_FIFTH:
      if action_note > prev_note:
        leap_direction = rl_tutor_ops.ASCENDING
        if verbose:
          tf.logging.info('Detected an ascending leap')
          print 'Detected an ascending leap'
      else:
        leap_direction = rl_tutor_ops.DESCENDING
        if verbose:
          tf.logging.info('Detected a descending leap')
          print 'Detected a descending leap'

      # there was already an unresolved leap
      if self.composition_direction != 0:
        if self.composition_direction != leap_direction:
          if verbose:
            tf.logging.info('Detected a resolved leap')
            tf.logging.info('Num steps since last leap: %s',
                         self.steps_since_last_leap)
            print('Leap resolved by a leap. Num steps since last leap:', 
                  self.steps_since_last_leap)
          if self.steps_since_last_leap > steps_between_leaps:
            outcome = rl_tutor_ops.LEAP_RESOLVED
            if verbose:
              tf.logging.info('Sufficient steps before leap resolved, '
                           'awarding bonus')
              print 'Sufficient steps were taken. Awarding bonus'
          self.composition_direction = 0
          self.leapt_from = None
        else:
          if verbose:
            tf.logging.info('Detected a double leap')
            print 'Detected a double leap!'
          outcome = rl_tutor_ops.LEAP_DOUBLED

      # the composition had no previous leaps
      else:
        if verbose:
          tf.logging.info('There was no previous leap direction')
          print 'No previous leap direction'
        self.composition_direction = leap_direction
        self.leapt_from = prev_note

      self.steps_since_last_leap = 0

    # there is no leap
    else:
      self.steps_since_last_leap += 1
      if verbose:
        tf.logging.info('No leap, adding to steps since last leap. '
                     'It is now: %s', self.steps_since_last_leap)

      # If there was a leap before, check if composition has gradually returned
      # This could be changed by requiring you to only go a 5th back in the 
      # opposite direction of the leap.
      if (self.composition_direction == rl_tutor_ops.ASCENDING and
          action_note <= self.leapt_from) or (
              self.composition_direction == rl_tutor_ops.DESCENDING and
              action_note >= self.leapt_from):
        if verbose:
          tf.logging.info('detected a gradually resolved leap')
          print 'Detected a gradually resolved leap'
        outcome = rl_tutor_ops.LEAP_RESOLVED
        self.composition_direction = 0
        self.leapt_from = None

    return outcome

  def reward_leap_up_back(self,
                          action,
                          resolving_leap_bonus=5.0,
                          leaping_twice_punishment=-5.0, 
                          verbose=False):
    """Applies punishment and reward based on the principle leap up leap back.

    Large interval jumps (more than a fifth) should be followed by moving back
    in the same direction.

    Args:
      action: One-hot encoding of the chosen action.
      resolving_leap_bonus: Amount of reward dispensed for resolving a previous
        leap.
      leaping_twice_punishment: Amount of reward received for leaping twice in
        the same direction.
      verbose: If True, model will print additional debugging statements.
    Returns:
      Float reward value.
    """

    leap_outcome = self.detect_leap_up_back(action, verbose=verbose)
    if leap_outcome == rl_tutor_ops.LEAP_RESOLVED:
      if verbose: print "leap resolved, awarding", resolving_leap_bonus
      return resolving_leap_bonus
    elif leap_outcome == rl_tutor_ops.LEAP_DOUBLED:
      if verbose: print "leap doubled, awarding", leaping_twice_punishment
      return leaping_twice_punishment
    else:
      return 0.0

  def reward_interval_diversity(self):
    # TODO(natashajaques): music theory book also suggests having a mix of steps
    # that are both incremental and larger. Want to write a function that
    # rewards this. Could have some kind of interval_stats stored by
    # reward_preferred_intervals function.
    pass

  def evaluate_music_theory_metrics(self, num_compositions=10000, key=None,
                                    tonic_note=rl_tutor_ops.C_MAJOR_TONIC):
    """Computes statistics about music theory rule adherence.

    Args: 
      num_compositions: How many compositions should be randomly generated
        for computing the statistics.
      key: The numeric values of notes belonging to this key. Defaults to C
        Major if not provided.
      tonic_note: The tonic/1st note of the desired key.

    Returns: A dictionary containing the statistics.
    """
    stat_dict = rl_tuner_eval_metrics.compute_composition_stats(
      self,
      num_compositions=num_compositions,
      composition_length=self.num_notes_in_melody,
      key=key,
      tonic_note=tonic_note)

    print rl_tuner_eval_metrics.get_stat_dict_string(stat_dict)

    return stat_dict