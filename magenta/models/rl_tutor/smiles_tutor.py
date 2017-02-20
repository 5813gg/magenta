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
import sys
import json

import matplotlib.pyplot as plt

import numpy as np
from scipy.misc import logsumexp
import tensorflow as tf
import networkx as nx

from rdkit.Chem import MolFromSmiles, Descriptors, rdmolops

import qed
import smiles_rnn
import rl_tutor_ops
import sascorer
from rl_tutor import RLTutor

def reload_files():
  """Used to reload the imported dependency files (necessary for jupyter 
  notebooks).
  """
  reload(smiles_rnn)
  reload(rl_tutor_ops)
  reload(sascorer)
  reload(qed)

# Special token indicating EOS
EOS = 0

class SmilesTutor(RLTutor):
  """Implements a recurrent DQN designed to produce SMILES sequences."""

  def __init__(self, output_dir,
               vocab_file='/home/natasha/Dropbox/Google/SMILES-Project/data/zinc_char_list.json',
               max_seq_len=120,
               reward_values=None,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='default',
               reward_scaler=1.0,
               exploration_mode='egreedy',
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
      vocab_file: A JSON file containing a list of the characters in the 
        SMILES vocabulary.
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
    print "In child class SmilesTutor"

    if reward_values is None:
      self.reward_values = rl_tutor_ops.smiles_reward_values()
    else:
      self.reward_values = reward_values

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

    self.vocab_file = vocab_file
    self.max_seq_len = max_seq_len
    self.load_vocab()

    # State variables needed by reward functions.
    self.data_reward_this_sequence = 0

  def load_vocab(self):
    print "Loading vocabulary from file", self.vocab_file
    if not os.path.exists(self.vocab_file):
        print "ERROR! Vocab file", self.vocab_file, "does not exist!"
        sys.exit()
    self.char_list = json.load(open(self.vocab_file))
    self.vocab_size = len(self.char_list)
    self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
    self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))

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

  def get_random_action(self):
    """Sample an action uniformly at random. Exclude EOS

    Returns:
      one-hot encoding of random action
    """
    act_idx = np.random.randint(EOS + 1, self.num_actions - 1)
    return np.array(rl_tutor_ops.make_onehot([act_idx], 
                                             self.num_actions)).flatten()

  def is_end_of_sequence(self, seq=None):
    """Returns true if the sequence generated by the model has reached its end.

    Should be overwritten by child class.
    """
    if seq is None:
      seq = self.generated_seq

    if len(seq) > 0 and seq[-1] == EOS:
      return True
    if len(seq) >= self.max_seq_len:
      return True
    return False

  def reset_for_new_sequence(self):
    """Starts the models internal composition over at beat 0, with no notes.

    Also resets statistics about whether the composition is in the middle of a
    melodic leap.
    """
    self.generated_seq_step = 0
    self.generated_seq = []
    self.data_reward_this_sequence = 0

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
    data_reward = self.reward_values.data_scalar * self.reward_from_reward_rnn_scores(action, reward_scores)
    self.data_reward_this_sequence += data_reward
    self.data_reward_last_n += data_reward

    domain_reward = self.collect_domain_reward(obs, action, verbose=verbose)
    self.domain_reward_last_n += domain_reward * self.reward_scaler

    if verbose and self.is_end_of_sequence(self.generated_seq + [np.argmax(action)]):
      print 'Cumulative Reward RNN reward:', self.data_reward_this_sequence
      print 'Total domain reward:', self.reward_scaler * domain_reward
      print ""
      
    if not self.domain_rewards_only:
      return domain_reward * self.reward_scaler + data_reward
    else:
      return domain_reward * self.reward_scaler 

    """if self.is_valid_molecule(self.generated_seq):
      self.data_reward_last_n += self.data_reward_this_sequence
      if verbose:
        print 'Sequence is valid! Awarding data reward of', self.data_reward_this_sequence
        print 'Total domain reward:', self.reward_scaler * domain_reward
        print ""
      
      # only get data rewards if valid molecule and not pure rl
      if not self.domain_rewards_only:
        return domain_reward * self.reward_scaler + self.data_reward_this_sequence
    
    if verbose:
      print 'Invalid seq. No data reward.'
      print 'Total domain reward:', self.reward_scaler * reward
      print ""

    return domain_reward * self.reward_scaler 
  """


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

    mol = self.is_valid_molecule(self.generated_seq)
    
    # invalid molecule, sequence not over
    if not mol and not np.argmax(action) == EOS and len(self.generated_seq) + 1 < self.max_seq_len:
      return 0

    # valid molecule
    if mol:
      sa = self.reward_values.sa_multiplier * self.get_sa_score(mol)
      logp = self.reward_values.logp_multiplier * self.get_logp(mol)
      ringp = self.reward_values.ringp_multiplier * self.get_ring_penalty(mol)
      qed = self.reward_values.qed_multiplier * self.get_qed(mol)
      sum_mol_rewards = logp + ringp + qed + sa

      if verbose:
        print "logP reward:", logp * self.reward_scaler
        print "SA reward:", sa * self.reward_scaler
        print "QED reward:", qed * self.reward_scaler
        print "ring penalty reward:", ringp * self.reward_scaler
        
        print "Total:", sum_mol_rewards * self.reward_scaler

    # valid molecule, sequence not over
    if mol and not np.argmax(action) == EOS and len(self.generated_seq) + 1 < self.max_seq_len:
      if verbose:
        print "Temporarily valid reward:", self.reward_values.any_valid_bonus * self.reward_scaler
      return self.reward_values.any_valid_bonus + sum_mol_rewards

    # Applying end of sequence penalties and bonuses
    if verbose: print "Sequence finished!"

    length_penalties = self.get_length_penalty()
    if verbose and length_penalties != 0:
      print "Sequence is weird length. Penalty:", length_penalties

    if mol:
      end_length_bonus = self.reward_values.valid_length_multiplier * len(self.generated_seq)
      if verbose: 
        print "Applying final bonus for ending on VALID SEQUENCE!"
        print "Ending valid bonus:", self.reward_values.end_valid_bonus * self.reward_scaler
        print "Valid length bonus:", end_length_bonus * self.reward_scaler
      return sum_mol_rewards + length_penalties + self.reward_values.end_valid_bonus + end_length_bonus
    else:
      end_length_penalty = self.reward_values.invalid_length_multiplier * len(self.generated_seq)
      if verbose: 
        print "Penalizing ending on invalid sequence!"
        print "Ending invalid penalty:", self.reward_values.end_invalid_penalty * self.reward_scaler
        print "Invalid length penalty:", end_length_penalty * self.reward_scaler
      return length_penalties + self.reward_values.end_invalid_penalty + end_length_penalty
    

  def convert_seq_to_chars(self, seq):
    """Converts a list of ints to a SMILES string

    Args:
      seq: A list of ints
    Returns:
      A string representing the SMILES encoding.
    """
    char_list = [str(self.index_to_char[s]) for s in seq]
    return ''.join(char_list)

  def is_valid_molecule(self, seq):
    """Checks if a sequence is a valid SMILES encoding.

    Args:
      seq: A list of ints.
    Returns:
      An rdkit Molecule object if the sequence is valid, nothing
      otherwise.
    """
    if len(seq) == 0 or (len(seq) == 1 and seq[0] == EOS):
      return False
    smiles_string = self.convert_seq_to_chars(seq)
    return MolFromSmiles(smiles_string)

  def get_sa_score(self, mol):
    """Gets the Synthetic Accessibility score of an rdkit molecule.

    Args:
      mol: An rdkit molecule object
    Returns:
      A float SA score
    """
    try:
      return -1 * sascorer.calculateScore(mol)
    except Exception as error:
      print "Uh oh, SA scorer had an error:", repr(error) 
      print "Sequence was", self.generated_seq
      return -5

  def get_logp(self, mol):
    """Gets water-octanol partition coefficient (logP) score mol.

    Args:
      mol: An rdkit molecule object
    Returns:
      A float logP
    """
    try:
      logp = Descriptors.MolLogP(mol)
      return logp
    except:
      print "Error! logP could not be computed (most likely because getNumBonds() <= 0)"
      return 0

  def get_qed(self, mol):
    """Gets the Quantitative Estimation of Drug-likeness of mol.

    Args:
      mol: An rdkit molecule object
    Returns:
      A float QED score
    """
    try:
      return qed.default(mol)
    except:
      return 0

  def get_ring_penalty(self, mol):
    """Calculates a penalty based on carbon rings larger than 6.

    Args:
      mol: An rdkit molecule object
    Returns:
      A float penalty.
    """
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        return 0
    
    cycle_length = max([ len(j) for j in cycle_list ])
    
    if cycle_length <= 6:
        return 0
    return -1* (cycle_length - 6)

  def get_length_penalty(self):
    """Calculates a penalty based on unusually short or long seqs.

    Args:
      mol: An rdkit molecule object
    Returns:
      A float penalty.
    """
    training_data_average=45.82
    training_data_std=9.37
    training_data_min=19
    training_data_max=78

    penalty = 0
    if len(self.generated_seq) < training_data_min:
      penalty += self.reward_values.short_seq
    elif len(self.generated_seq) < training_data_average - 2 * training_data_std:
      penalty +=  self.reward_values.shortish_seq
  
    if len(self.generated_seq) > training_data_max:
      penalty += self.reward_values.long_seq
    elif len(self.generated_seq) > training_data_average + 2 * training_data_std:
      penalty +=  self.reward_values.longish_seq

    return penalty

  def render_sequence(self, generated_seq, title='smiles_seq'):
    """Renders a generated SMILES sequence its string version.

    Args:
      generated_seq: A list of integer note/action values.
      title: A title to use in the sequence filename.
    """
    print self.convert_seq_to_chars(generated_seq)
    if self.is_valid_molecule(generated_seq):
        print "VALID molecule"
    else:
        print "Invalid molecule :("
  
  # The following functions evaluate generated molecules for quality.
  # TODO: clean up since there is code repeated from rl_tuner_eval_metric
  def evaluate_domain_metrics(self, num_sequences=10000, sample_next=True):
    """Computes statistics about domain rule adherence.

    Args: 
      num_sequences: How many molecules should be randomly generated
        for computing the statistics.
      sample_next: If True, generated tokens are sampled from the output
        distribution. If False, the token with maximum value is selected.

    Returns: A dictionary containing the statistics.
    """
    stat_dict = self.initialize_stat_dict()

    for i in range(num_sequences):
      self.generate_sequence()
      stat_dict = self.add_sequence_stats(stat_dict)

    print self.get_stat_dict_string(stat_dict)
    return stat_dict

  def initialize_stat_dict(self):
    """Initializes a dictionary which will hold statistics about molecules.

    Returns:
      A dictionary containing the appropriate fields initialized to 0 or an
      empty list.
    """
    stat_dict = dict()

    stat_dict['num_sequences'] = 0
    stat_dict['num_tokens'] = 0
    stat_dict['num_valid_sequences'] = 0
    stat_dict['sum_logp'] = 0
    stat_dict['sum_ring_penalty'] = 0
    stat_dict['sum_sa'] = 0
    stat_dict['sum_qed'] = 0
    stat_dict['best_logp'] = None
    stat_dict['best_sa'] = None
    stat_dict['best_qed'] = None
    stat_dict['best_logp_seq'] = None
    stat_dict['best_sa_seq'] = None
    stat_dict['best_qed_seq'] = None
    stat_dict['num_seqs_w_no_ring_penalty'] = 0

    return stat_dict

  def get_stat_dict_string(self, stat_dict):
    """Makes string of interesting statistics from a composition stat_dict.

    Args:
      stat_dict: A dictionary storing statistics about a series of compositions.
    Returns:
      String containing several lines of formatted stats.
    """
    tot_seqs = float(stat_dict['num_sequences'])
    tot_toks = float(stat_dict['num_tokens'])
    avg_seq_len = tot_toks / tot_seqs

    return_str = 'Total sequences: ' + str(tot_seqs) + '\n'
    return_str += 'Total tokens: ' + str(tot_toks) + '\n'
    return_str += 'Average sequence length: ' + str(avg_seq_len) + '\n\n'

    return_str += 'Percent valid molecules: '
    return_str += str((stat_dict['num_valid_sequences'] / tot_seqs)*100.0) + '%\n'

    return_str += 'Percent with no carbon rings larger than six: '
    return_str += str((stat_dict['num_seqs_w_no_ring_penalty'] / tot_seqs)*100.0) + '%\n'
    return_str += 'Average logP: '
    return_str += str(float(stat_dict['sum_logp']) / tot_seqs) + '\n'
    return_str += 'Average ring penalty: '
    return_str += str(float(stat_dict['sum_ring_penalty']) / tot_seqs) + '\n'
    return_str += 'Average SA: '
    return_str += str(float(stat_dict['sum_sa']) / tot_seqs) + '\n'
    return_str += 'Average QED: '
    return_str += str(float(stat_dict['sum_qed']) / tot_seqs) + '\n'
    
    return_str += '\n'
    return_str += 'Best logP: ' + str(stat_dict['best_logp']) + '\n'
    return_str += 'Sequence with best logP: ' + str(stat_dict['best_logp_seq']) + '\n'
    return_str += 'Best SA: ' + str(stat_dict['best_sa']) + '\n'
    return_str += 'Sequence with best SA: ' + str(stat_dict['best_sa_seq']) + '\n'
    return_str += 'Best QED: ' + str(stat_dict['best_qed']) + '\n'
    return_str += 'Sequence with best QED: '  + str(stat_dict['best_qed_seq']) + '\n'

    return_str += '\n'

    return return_str

  def generate_sequence(self, sample_next_obs=True, verbose=False):
    """Generates a sequence using the model, stores statistics about it in a dict.

    Args:
      sample_next_obs: If True, each note will be sampled from the model's
        output distribution. If False, each note will be the one with maximum
        value according to the model.
    """
    last_observation = self.prime_internal_models()
    self.reset_for_new_sequence()

    i = 0
    while not self.is_end_of_sequence():
      if sample_next_obs:
        action, new_observation, reward_scores = self.action(
            last_observation,
            enable_random=False,
            sample_next_obs=sample_next_obs)
      else:
        action, reward_scores = self.action(
            last_observation,
            enable_random=False,
            sample_next_obs=sample_next_obs)
        new_observation = action

      action_idx = np.argmax(action)
      obs_idx = np.argmax(new_observation)

      if verbose:
        self.collect_reward(last_observation, new_observation, reward_scores, verbose=True)

      self.generated_seq.append(np.argmax(new_observation))
      self.generated_seq_step += 1
      last_observation = new_observation

  def add_sequence_stats(self, stat_dict):
    """Updates stat dict based on self.generated_seq and desired metrics

    Args:
      stat_dict: A dictionary containing fields for statistics about
        sequences.
    Returns:
      A dictionary of sequence statistics with fields updated.
    """
    stat_dict['num_sequences'] += 1
    stat_dict['num_tokens'] += len(self.generated_seq)
    mol = self.is_valid_molecule(self.generated_seq)
    
    if not mol:
      return stat_dict
    
    stat_dict['num_valid_sequences'] += 1
    
    logp = self.get_logp(mol)
    stat_dict['sum_logp'] += logp
    stat_dict = self._replace_stat_if_best(stat_dict, 'best_logp', logp)

    sa = self.get_sa_score(mol)
    stat_dict['sum_sa'] += sa
    stat_dict = self._replace_stat_if_best(stat_dict, 'best_sa', sa)

    qed = self.get_qed(mol)
    stat_dict['sum_qed'] += qed
    stat_dict = self._replace_stat_if_best(stat_dict, 'best_qed', qed)

    ring_penalty = self.get_ring_penalty(mol)
    stat_dict['sum_ring_penalty'] += ring_penalty
    if ring_penalty == 0:
      stat_dict['num_seqs_w_no_ring_penalty'] += 1

    return stat_dict

  def _replace_stat_if_best(self, stat_dict, stat_name, stat):
    if stat_dict[stat_name] is None or stat > stat_dict[stat_name]:
      stat_dict[stat_name] = stat
      stat_dict[stat_name + '_seq'] = self.convert_seq_to_chars(self.generated_seq)

    return stat_dict

  def debug_reward(self, num_times=10):
    for i in range(num_times):
      print "Starting a new sequence"
      self.generate_sequence(verbose=True)
      print "Generated sequence:", self.generated_seq
      print self.convert_seq_to_chars(self.generated_seq), '\n'
      print '\n'