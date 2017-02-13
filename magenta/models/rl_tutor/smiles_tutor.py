"""Defines a Deep Q Network (DQN) with augmented reward to create SMILES 
molecule sequences by using reinforcement learning to fine-tune a pre-trained 
SMILES RNN according to rewards based on desirable molecular properties.

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

import smiles_rnn
import rl_tutor_ops
from rl_tutor import RLTutor

def reload_files():
  """Used to reload the imported dependency files (necessary for jupyter 
  notebooks).
  """
  reload(smiles_rnn)
  reload(rl_tutor_ops)

# Special token indicating EOS
EOS = 0

class SmilesTutor(RLTutor):
  """Implements a recurrent DQN designed to produce SMILES sequences."""

  def __init__(self, output_dir,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='default',
               reward_scaler=1.0,
               exploration_mode='boltzmann',
               priming_mode='random',
               stochastic_observations=False,
               algorithm='q',      

               # Pre-trained RNN to load and tune
               rnn_checkpoint_dir=None,
               rnn_checkpoint_file=None,
               rnn_type='default',
               rnn_hparams=None,

               # Logistics.
               input_size=rl_tutor_ops.NUM_CLASSES_SMILE,
               num_actions=rl_tutor_ops.NUM_CLASSES_SMILE,
               save_name='smiles_rnn.ckpt',
               output_every_nth=1000,
               summary_writer=None,
               initialize_immediately=True):
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
      rnn_checkpoint_dir: The directory from which the internal 
        NoteRNNLoader will load its checkpointed LSTM.
      rnn_checkpoint_file: A checkpoint file to use in case one cannot be
        found in the note_rnn_checkpoint_dir.
      rnn_type: If 'default', will use the basic LSTM described in the 
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      rnn_hparams: A tf.HParams object which defines the hyper parameters
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
    """
    print "In child class Smiles RNN"

    if rnn_hparams is None:
      rnn_hparams = rl_tutor_ops.smiles_hparams()

    RLTutor.__init__(self, output_dir, dqn_hparams=dqn_hparams, 
      reward_mode=reward_mode, reward_scaler=reward_scaler, 
      exploration_mode=exploration_mode, priming_mode=priming_mode,
      stochastic_observations=stochastic_observations, algorithm=algorithm,
      rnn_checkpoint_dir=rnn_checkpoint_dir, 
      rnn_checkpoint_file=rnn_checkpoint_file, 
      rnn_type=rnn_type, rnn_hparams=rnn_hparams, input_size=input_size,
      num_actions=num_actions, save_name=save_name, 
      output_every_nth=output_every_nth, summary_writer=summary_writer, 
      initialize_immediately=initialize_immediately)

  def initialize_internal_models(self, ):
    """Initializes internal RNN models: q_network, target_q_network, reward_rnn.

    Adds the graphs of the internal RNN models to this graph, by having a separate
    function for this rather than doing it in the constructor, it allows classes
    inheriting from this class to define their own type of q_network.
    """
    # Add internal networks to the graph.
    tf.logging.info('Initializing q network')
    self.q_network = smiles_rnn.SmilesRNN(
      self.rnn_checkpoint_dir,
      graph=self.graph, 
      scope='q_network', 
      checkpoint_file=self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      rnn_type=self.rnn_type,
      vocab_size=self.input_size)

    tf.logging.info('Initializing target q network')
    self.target_q_network = smiles_rnn.SmilesRNN(
      self.rnn_checkpoint_dir,
      graph=self.graph, 
      scope='target_q_network', 
      checkpoint_file=self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      rnn_type=self.rnn_type,
      vocab_size=self.input_size)

    tf.logging.info('Initializing reward network')
    self.reward_rnn = smiles_rnn.SmilesRNN(
      self.rnn_checkpoint_dir,
      graph=self.graph, 
      scope='reward_rnn', 
      checkpoint_file=self.rnn_checkpoint_file,
      hparams=self.rnn_hparams,
      rnn_type=self.rnn_type,
      vocab_size=self.input_size)

    tf.logging.info('Q network cell: %s', self.q_network.cell)

  def prime_internal_model(self, model):
    """Prime an internal model such as the q_network based on priming mode.

    Args:
      model: The internal model that should be primed. 

    Returns:
      The first observation to feed into the model.
    """
    model.state_value = model.get_zero_state()

    if self.priming_mode == 'random':
      next_obs = self.get_random_action()
    else:
      tf.logging.warn('Error! Invalid priming mode. Priming with random token')
      next_obs = self.get_random_action()

    return next_obs

  def is_end_of_sequence(self):
    """Returns true if the sequence generated by the model has reached its end.

    Should be overwritten by child class.
    """
    if len(self.generated_seq) > 0 and self.generated_seq[-1] == EOS:
      return True
    return False

  def collect_domain_reward(self, obs, action, verbose=False):
    """Calls whatever reward function is indicated in the reward_mode field.

    New reward functions can be written and called from here. Note that the
    reward functions can make use of the generated molecule that has been
    played so far, which is stored in self.generated_seq. 

    Args:
      obs: A one-hot encoding of the observed token.
      action: A one-hot encoding of the chosen action.
      verbose: If True, additional logging statements about the reward after
        each function will be printed.
    Returns:
      Float reward value.
    """
    return 0

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