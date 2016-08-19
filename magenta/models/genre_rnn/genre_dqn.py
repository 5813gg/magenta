"""Implements GenreDQN, which learns to play melodies with a genre using RL.

The GenreDQN class inherits from MelodyQNetwork, and therefore implements
reinforcement learning in order to train a recurrent network that plays
melodies.

GenreDQN is designed to use a GenreCLassifier model to provide rewards. When
training and composing, GenreDQN has a target genre for its composition. When
training, its reward will be proportional to the probability with which the
GenreClassifier believes its composition matches the target genre. I.e. if the
GenreDQN is trying to compose a 'classical' song, and the GenreClassifier
outputs a confidence of 0.6 for 'classical' and 0.4 for 'pop', then the reward
will be proportional to 0.6.

GenreRNNs, which are built to take in extra genre bits when predicting the next
note, and which can produced melodies from a specific genre, are used to
provide the 'q_network' and 'target_q_network' for this class.
"""
import os

import numpy as np
import tensorflow as tf


from ...genre_rnn import genre_classifier
from ....genre_rnn import genre_rnn
from ....rl_rnn import melody_q
from ....rl_rnn import rl_rnn_ops


def reload_files():
  reload(rl_rnn_ops)
  reload(melody_q)
  reload(genre_rnn)
  reload(genre_classifier)


def default_dqn_hparams():
  """Returns default hyperparameter settings for the GenreDQN."""
  return tf.HParams(
      random_action_probability=0.1,
      store_every_nth=1,
      train_every_nth=5,
      minibatch_size=32,
      discount_rate=0.95,
      max_experience=100000,
      target_network_update_rate=0.01)


class GenreDQN(melody_q.MelodyQNetwork):
  """Implements the GenreDQN class as described above."""

  def __init__(self,
               # file paths and directories
               genre_classifier_checkpoint_dir,
               genre_rnn_checkpoint_dir,
               output_dir,

               # genre-specific params
               genres=None,
               genre_classifier_checkpoint_scope='genre_classifier',
               genre_classifier_backup_checkpoint_file=None,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='genre_classifier',
               reward_scaler=1.0,
               priming_mode='random_note',

               # Other music related settings.
               num_notes_in_melody=192,
               reward_every_n_notes=32,
               final_composition_bonus=10,
               note_input_length=genre_rnn.NOTE_ONE_HOT_LENGTH,
               midi_primer=None,

               # Logistics.
               output_every_nth=1000,
               genre_rnn_backup_checkpoint_file=None,
               training_data_path=None,
               genre_rnn_hparams=None):
    """Initializes the GenreDQN class and its internal networks, graph, session.

    Args:
      genre_classifier_checkpoint_dir: Path to a directory containing a
        checkpoint of a trained GenreClassifier model.
      genre_rnn_checkpoint_dir: Path to a directory containing a checkpoint of
        a trained GenreRNN model.
      output_dir: Path to a directory where checkpoints of this model and
        generated composition MIDI files will be saved.
      genres: String names of the genres that will be
        classified. These must match the settings used to train the
        GenreClassifier and GenreRNN models.
      genre_classifier_checkpoint_scope: Tensorflow scope with which the
        checkpointed genre classifier was saved.
      genre_classifier_backup_checkpoint_file: A checkpoint file to use in
        case one cannot be found in the genre_classifier_checkpoint_dir.
      dqn_hparams: A tf.HParams object containing the hyperparameters of the
        DeepQNetwork parent class.
      reward_mode: Controls which reward function can be applied. If
        'genre_classifier', it will use the GenreClassifier model to give
        rewards for making a song that sounds like the desired genre. Any
        other reward mode will cause this class to call into the parent
        class's collect_reward function.
      reward_scaler: All the rewards are multiplied by this value. Used to
        change the magnitude of the rewards given, which strongly affects
        whether the algorithm can learn.
      priming_mode: Each time the model begins a new composition, it is primed
        with either a random note ('random_note'), a random MIDI file from the
        training data ('random_midi'), or a particular MIDI file
        ('single_midi').
      num_notes_in_melody: The length of a composition of the model.
      reward_every_n_notes: The number of notes the model has to play before
        its composition is fed to the GenreClassfiier to obtain a reward.
      final_composition_bonus: After the model has finished creating a
        composition of length 'num_notes_in_melody', a final reward will be
        applied based on whether the composition sounds like a certain genre.
        This reward will be multiplied by final_composition_bonus.
      note_input_length: The number of bits in the input bit vector dedicated
        to representing the note (rather than the genre).
      midi_primer: Path to a MIDI file that can be used to prime the GenreRNN.
      output_every_nth: How many training steps before the model will print
        an output saying the cumulative reward, and save a checkpoint.
      genre_rnn_backup_checkpoint_file: A checkpoint file to use in case one
        cannot be found in the genre_rnn_checkpoint_dir.
      training_data_path: A path to a tfrecord file containing melody training
        data. This is necessary to use the 'random_midi' priming mode.
      genre_rnn_hparams: A tf.HParams object representing the hyperparameters
        of the internal GenreRNN model.
    """
    if genres is None:
      self.genres = genre_rnn.DEFAULT_GENRES
    else:
      self.genres = genres
    self.num_genres = len(self.genres)
    self.genre_rnn_hparams = genre_rnn_hparams
    self.genre_classifier_checkpoint_dir = genre_classifier_checkpoint_dir
    self.genre_classifier_checkpoint_scope = genre_classifier_checkpoint_scope
    self.genre_classifier_backup_ckpt = genre_classifier_backup_checkpoint_file
    self.note_input_length = note_input_length
    self.reward_every_n_notes = reward_every_n_notes
    self.final_composition_bonus = final_composition_bonus
    input_size = self.note_input_length + self.num_genres
    log_dir = os.path.join(output_dir, 'genre_dqn_model')
    self.midi_primer = midi_primer
    logging.info('Genre DQN MIDI primer is: %s', self.midi_primer)

    if num_notes_in_melody % reward_every_n_notes != 0:
      logging.fatal("Error! 'reward_every_n_notes' must be a divisor of"
                    "'num_notes_in_melody'")

    if dqn_hparams is None:
      dqn_hparams = default_dqn_hparams()

    # initialize parent class
    super(GenreDQN, self).__init__(
        output_dir,
        log_dir,
        genre_rnn_checkpoint_dir,
        midi_primer=self.midi_primer,
        dqn_hparams=dqn_hparams,
        reward_mode=reward_mode,
        reward_scaler=reward_scaler,
        priming_mode=priming_mode,
        stochastic_observations=False,
        num_notes_in_melody=num_notes_in_melody,
        input_size=input_size,
        num_actions=input_size,
        output_every_nth=output_every_nth,
        backup_checkpoint_file=genre_rnn_backup_checkpoint_file,
        training_data_path=training_data_path,
        initialize_immediately=False)

    self.initialize_genre_dqn()

    # ID of the current composition's genre
    self.composition_genre = 0

  def initialize_genre_dqn(self):
    """Initializes internal models, including GenreRNNs and GenreClassifier.

    Also restores the weights for these models from their respective
    checkpoints.
    """
    with self.graph.as_default():
      # Add internal networks to the graph.
      logging.info('Initializing q network')
      self.q_network = genre_rnn.GenreRNN(
          hparams=self.genre_rnn_hparams,
          training_data_path=self.training_data_path,
          midi_primer=self.midi_primer,
          output_dir=self.output_dir,
          checkpoint_dir=self.melody_checkpoint_dir,
          backup_checkpoint_file=self.backup_checkpoint_file,
          note_input_length=self.note_input_length,
          genres=self.genres,
          scope='q_network',
          graph=self.graph,
          softmax_within_graph=False)
      logging.info('Q network cell: %s', self.q_network.cell)

      logging.info('Initializing target q network')
      self.target_q_network = genre_rnn.GenreRNN(
          hparams=self.genre_rnn_hparams,
          training_data_path=self.training_data_path,
          midi_primer=self.midi_primer,
          output_dir=self.output_dir,
          checkpoint_dir=self.melody_checkpoint_dir,
          backup_checkpoint_file=self.backup_checkpoint_file,
          note_input_length=self.note_input_length,
          genres=self.genres,
          scope='target_q_network',
          graph=self.graph,
          softmax_within_graph=False)

      logging.info('Initializing reward network')
      self.reward_rnn = genre_rnn.GenreRNN(
          hparams=self.genre_rnn_hparams,
          training_data_path=self.training_data_path,
          midi_primer=self.midi_primer,
          output_dir=self.output_dir,
          checkpoint_dir=self.melody_checkpoint_dir,
          backup_checkpoint_file=self.backup_checkpoint_file,
          note_input_length=self.note_input_length,
          genres=self.genres,
          scope='reward_rnn',
          graph=self.graph,
          softmax_within_graph=False)

      # Add rest of variables to graph.
      logging.info('Adding RL graph variables')
      self.build_graph()

      self.genre_classifier = genre_classifier.GenreClassifier(
          note_input_length=self.note_input_length,
          num_genres=self.num_genres,
          scope='genre_classifier',
          graph=self.graph,
          start_fresh=False)

      # Prepare saver and session.
      self.saver = tf.train.Saver()
      self.session = tf.Session(graph=self.graph)
      self.session.run(tf.initialize_all_variables())

      # Initialize internal networks.
      self.q_network.initialize_and_restore(self.session)
      self.target_q_network.initialize_and_restore(self.session)
      self.reward_rnn.initialize_and_restore(self.session)
      self.genre_classifier.load_from_checkpoint_into_existing_graph(
          self.session, self.genre_classifier_checkpoint_dir,
          self.genre_classifier_checkpoint_scope,
          backup_checkpoint_file=self.genre_classifier_backup_ckpt)

    if self.priming_mode == 'random_midi':
      self.get_priming_melodies()

  def generate_genre_music_sequence(self,
                                    genre,
                                    priming_mode='random',
                                    priming_note=None,
                                    visualize_probs=True,
                                    prob_image_name=None,
                                    composition_length=32,
                                    most_probable=False):
    """Generates a music sequence with the current model, and saves it to MIDI.

    Calls internal 'q_network's generate function, which deals with appending
    the desired genre to the input.

    The resulting MIDI file is saved to the model's output_dir directory.

    Args:
      genre: A string representing the desired genre of the composition. E.g.
        'classical' or 'pop.
      priming_mode: A string representing how the internal 'q_network' GenreRNN
        will be primed. If 'random', will use a random note.
      priming_note: If 'priming_mode' is set to 'specific_note', can use this
        argument to give the integer note that will be used to prime the model.
      visualize_probs: if True, the function will plot the softmax
        probabilities of the model for each note that occur throughout the
        sequence.
      prob_image_name: The name of a file in which to save the softmax
        probability image. If None, the image will simply be displayed.
      composition_length: The length of the sequence to be generated. If None,
        defaults to the num_notes_in_melody parameter of the model.
      most_probable: If True, instead of sampling each note in the sequence,
        the model will always choose the argmax, most probable note.
    """
    if composition_length is None:
      composition_length = self.num_notes_in_melody

    self.q_network.generate_genre_music_sequence(genre, priming_mode,
                                                 priming_note, visualize_probs,
                                                 prob_image_name,
                                                 composition_length,
                                                 most_probable)

  def action(self,
             observation,
             exploration_period,
             enable_random=True,
             sample_next_obs=False):
    """Given an observation, runs the q_network to choose the current action.

    Appends genre to the note produced by the parent class's action function.

    Args:
      observation: A one-hot encoding of a single observation (note).
      exploration_period: The total length of the period the network will
        spend exploring, as set in the train function.
      enable_random: If False, the network cannot act randomly.
      sample_next_obs: If True, the next observation will be sampled from
        the softmax probabilities produced by the model, and passed back
        along with the action. If False, only the action is passed back.

    Returns:
      The action chosen, and if sample_next_obs is True, also returns the next
      observation.
    """
    if sample_next_obs:
      action_note, obs = super(GenreDQN, self).action(
          observation,
          exploration_period,
          enable_random=enable_random,
          sample_next_obs=sample_next_obs)
    else:
      action_note = super(GenreDQN, self).action(
          observation,
          exploration_period,
          enable_random=enable_random,
          sample_next_obs=sample_next_obs)

    # add genre bits
    genre_action = self.q_network.add_genre_to_note(
        action_note[:self.note_input_length], self.composition_genre)

    if sample_next_obs:
      return genre_action, obs
    else:
      return genre_action

  def reset_composition(self):
    """Resets variables related to the model's current composition.

    Calls the parent class to reset the composition list and beat, and resets
    the composition genre.
    """
    self.composition_genre = np.random.randint(0, self.num_genres)
    super(GenreDQN, self).reset_composition()

  def get_random_note(self):
    """Sample a note uniformly at random. Adds genre bits to the end.

    Returns:
      random note
    """
    note_idx = np.random.randint(0, self.note_input_length)
    note_vector = np.array(
        rl_rnn_ops.make_onehot([note_idx], self.note_input_length)).flatten()
    return self.q_network.add_genre_to_note(note_vector, self.composition_genre)

  def collect_reward(self, obs, action, state):
    """Calls whatever reward function is indicated in the reward_mode field.

    If 'reward_mode' is 'genre_classifier' (the default) will call this model's
    'reward_genre_classifier' function. Otherwise will call out to the parent
    class.

    Args:
      obs: A one-hot encoding of the observed note.
      action: A one-hot encoding of the chosen action.
      state: The internal state of the LSTM q_network.
    Returns:
      Float reward value.
    """
    if self.reward_mode == 'genre_classifier':
      # Reward the model based on the decisions of the genre_classifier.
      reward = self.reward_genre_classifier(action)
    else:
      # Call the parent class to deal with other, legacy reward modes.
      return super(GenreDQN, self).collect_reward(obs, action, state)

    return reward * self.reward_scaler

  def reward_genre_classifier(self, action):
    """Uses the internal GenreClassifier to provide reward to the model.

    Note that the composition does not yet include the current action.

    Args:
      action: A one-hot encoding of the chosen action.
    Returns:
      Float return value.
    """
    song_length = len(self.composition) + 1
    if song_length < self.genre_classifier.hparams.skip_first_n_losses:
      return 0.0

    if song_length % self.reward_every_n_notes != 0:
      return 0.0

    action_note = np.argmax(action)
    composition = np.array(self.composition)
    composition = np.append(composition, action_note)

    # convert composition to sequence
    inputs = rl_rnn_ops.make_onehot(composition, self.note_input_length)
    inputs = np.reshape(inputs, (1, -1, self.note_input_length))
    lengths = [song_length]

    # feed sequence into genre classifier
    genre_probs = self.session.run(
        [self.genre_classifier.genre_probs],
        {self.genre_classifier.melody_sequence: inputs,
         self.genre_classifier.lengths: lengths})

    # select the appropriate genre output logit
    reward = genre_probs[0][self.composition_genre]

    if song_length == self.num_notes_in_melody:
      reward *= self.final_composition_bonus

    return reward
