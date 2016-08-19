"""Defines a class and operations for the GenreRNN model.

GenreRNN inherits from MelodyRNN. Like the parent class, it is designed to load
a trained note prediction RNN from a checkpoint file. This model can be placed
into the graph of another tensorflow model by simply calling it. E.g.:
  genre_rnn = GenreRNN()
  with tf.Graph().as_default:
    note = genre_rnn()

GenreRNN is distinct from MelodyRNN in that it can handle data that has extra
genre bits appended to the end of each note input vector. It can also generate
melodies of a desired genre by adding bits representing the desired genre to
each note that is generated during sampling.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import midi_io

from ...genre_rnn import genre_ops
from ...rl_rnn import melody_rnn
from ...rl_rnn import melody_rnn_encoder_decoder
from ...rl_rnn import rl_rnn_ops

NOTE_ONE_HOT_LENGTH = 38
DEFAULT_BPM = 120.0
DEFAULT_STEPS_PER_BEAT = 4
DEFAULT_STEPS_PER_BAR = 16
DEFAULT_GENRES = ['classical', 'pop']
DEFAULT_MIDI_PRIMER = 'primer.mid'


def default_hparams():
  """Generates the default hparams used to train a large basic rnn."""
  return tf.HParams(
      batch_size=128,
      lr=0.0002,
      l2_reg=2.5e-5,
      clip_norm=5,
      initial_learning_rate=0.5,
      decay_steps=1000,
      decay_rate=0.85,
      rnn_layer_sizes=[256, 256],
      skip_first_n_losses=8,
      one_hot_length=NOTE_ONE_HOT_LENGTH + len(DEFAULT_GENRES),
      exponentially_decay_learning_rate=True)


class GenreRNN(melody_rnn.MelodyRNN):
  """Implements the GenreRNN class as described above."""

  def __init__(
      self,
      hparams=None,
      output_dir='/tmp/genre_rnn',
      checkpoint_dir='mydir',
      checkpoint_scope='rnn_model',
      backup_checkpoint_file=None,
      training_data_path='myfile.tfrecord',
      midi_primer=None,
      note_input_length=NOTE_ONE_HOT_LENGTH,
      genres=None,
      scope='genre_rnn',
      graph=None,
      softmax_within_graph=True,
      bpm=DEFAULT_BPM,
      steps_per_bar=DEFAULT_STEPS_PER_BAR,
      steps_per_beat=DEFAULT_STEPS_PER_BEAT):
    """Initializes the GenreRNN.

    Args:
      hparams: Hyperparameters of the model. If None, defaults to using the
        parameters defined in the default_hparams() function of this file.
      output_dir: Path to a directory where the model will save generated MIDI
        files.
      checkpoint_dir: Path to a directory containing a checkpoint of a note
        prediction RNN trained on genre data.
      checkpoint_scope: Tensorflow scope that the variables in the checkpoint
        file will have.
      backup_checkpoint_file: Path to a checkpoint file of a trained genre note
        prediction RNN to be used if a checkpoint cannot be found in
        'checkpoint_dir'.
      training_data_path: Path to a tfrecord file containing genre sequence
        training data.
      midi_primer: Path to a MIDI file that can be used to prime the model if
        priming with a random note is not sufficient.
      note_input_length: The number of bits in an input note vector devoted to
        representing the note.
      genres: A list of string names of genres that the model has been trained
        to try to produce. Defaults to 'classical' and 'pop'.
      scope: The tensorflow graph scope in which the model will be created.
      graph: A tensorflow graph object. If None the model will create its own
        graph.
      softmax_within_graph: If True, the model will contain a softmax layer,
        such that when it is called it will output softmax probabilities.
      bpm: A number storing the default beats per minute that the model will use
        when generating melodies.
      steps_per_bar: The steps per bar the model will use when generating
        melodies.
      steps_per_beat: The steps per beat the model will use when generating
        melodies.
    """
    self.num_genres = len(genres)
    self.genres = genres
    self.note_input_length = note_input_length
    self.one_hot_length = note_input_length + self.num_genres
    self.output_dir = output_dir
    self.priming_input = None

    if midi_primer is None:
      midi_primer = DEFAULT_MIDI_PRIMER
    logging.info('MIDI primer is %s', midi_primer)

    if genres is None:
      genres = DEFAULT_GENRES

    if hparams is not None:
      logging.info('Using custom hparams')
      hparams = hparams
    else:
      logging.info('Empty hparams string. Using defaults')
      hparams = default_hparams()

    assert hparams.one_hot_length == self.one_hot_length

    if graph is None:
      graph = tf.Graph()

    super(GenreRNN, self).__init__(
        graph,
        scope,
        checkpoint_dir,
        midi_primer,
        training_data_path=training_data_path,
        hparams=hparams,
        backup_checkpoint_file=backup_checkpoint_file,
        softmax_within_graph=softmax_within_graph,
        checkpoint_scope=checkpoint_scope)

    # music-related settings
    self.transpose_amount = 0
    self.bpm = bpm
    self.steps_per_bar = steps_per_bar
    self.steps_per_beat = steps_per_beat

  def initialize_new_session(self):
    """Creates a new tensorflow session for the model and initializes variables.
    """
    with self.graph.as_default():
      self.session = tf.Session(graph=self.graph)
      self.session.run(tf.initialize_all_variables())

  def prime_with_midi_append_genre(self, genre, suppress_output=False):
    """Primes the model with its default midi primer. Appends genre to input.

    Args:
      genre: Integer ID of the desired genre.
      suppress_output: If True, output notifications about priming the model
        will be suppressed.

    Returns:
      The next note produced after priming the model.
    """
    with self.graph.as_default():
      if not suppress_output:
        logging.info('Priming the model with MIDI file %s', self.midi_primer)

      # Convert primer Melody to model inputs.
      encoder = melody_rnn_encoder_decoder.MelodyEncoderDecoder()
      seq = encoder.encode(self.primer)
      features = seq.feature_lists.feature_list['inputs'].feature
      primer_input = [list(i.float_list.value) for i in features]

      # the genre part
      primer_input = self.add_genre_to_note(primer_input, genre)

      # Run model over primer sequence.
      primer_input_batch = np.tile([primer_input], (self.batch_size, 1, 1))
      self.state_value, softmax = self.session.run(
          [self.state_tensor, self.softmax],
          feed_dict={self.initial_state: self.state_value,
                     self.melody_sequence: primer_input_batch,
                     self.lengths: np.full(
                         self.batch_size, len(self.primer), dtype=int)})
      priming_output = softmax[-1, :]
      self.priming_note = self.get_note_from_softmax(priming_output)
      return self.priming_note

  def get_random_note(self):
    """Get a randomly selected note.

    Returns:
      random note
    """
    note_idx = np.random.randint(0, self.note_input_length - 1)
    return np.array(rl_rnn_ops.make_onehot([note_idx],
                                           self.note_input_length)).flatten()

  def get_initial_observation(self,
                              genre,
                              priming_mode='random',
                              priming_note=None,
                              suppress_output=True,
                              disable_genre_bits=False):
    """Get the first note by generating one randomly or priming the model.

    Args:
      genre: Integer ID of the desired genre.
      priming_mode: If 'random', use a random note to prime the model. Can also
        be 'specific_note'.
      priming_note: If 'priming_mode' is 'specific_note', use this argument to
        pass the note used to prime the model.
      suppress_output: If False, statements about how the network is being
        primed will be printed out.
      disable_genre_bits: If True, the model will not append genre bits as it is
        composing.

    Returns:
      The first observation to feed into the model.
    """

    self.state_value = self.get_zero_state()

    if priming_mode == 'midi_file' and disable_genre_bits:
      if not suppress_output:
        logging.info('Priming with midi file')
      self.prime_model(suppress_output=suppress_output)
      next_obs = self.priming_note
    elif priming_mode == 'midi_file':
      if not suppress_output:
        logging.info('priming with midi file and appending genre')
      next_obs = self.prime_with_midi_append_genre(genre)
    elif priming_mode == 'specific_note':
      if not suppress_output:
        logging.info('priming with chosen note %s', priming_note)
      next_obs = np.array(
          rl_rnn_ops.make_onehot([priming_note],
                                 self.note_input_length)).flatten()
    else:
      if not suppress_output:
        logging.info('starting with random note')
      next_obs = self.get_random_note()
    # TODO(natashajaques): add ability to prime with a random training batch

    return next_obs

  def add_genre_to_note(self, note, genre_id):
    """Appends genre bits to an input note vector.

    Args:
      note: Either a 1D or 2D bit vector representing the input note(s).
      genre_id: An integer representing the ID of the desired genre.

    Returns:
      Modified note input vector containing appending genre bits.
    """
    if len(np.shape(note)) > 1:  # check if note input is 2D
      assert np.shape(note)[1] == self.note_input_length

      genre_bits = np.zeros((np.shape(note)[0], self.num_genres))
      genre_bits[:, genre_id] = 1.0

      return np.concatenate([note, genre_bits], axis=1)
    else:  # note input is 1D
      assert len(note) == self.note_input_length

      genre_bits = np.zeros(self.num_genres)
      genre_bits[genre_id] = 1.0

      return np.concatenate([note, genre_bits])

  def generate_genre_music_sequence(self,
                                    genre,
                                    priming_mode='random',
                                    priming_note=None,
                                    visualize_probs=False,
                                    prob_image_name=None,
                                    composition_length=32,
                                    most_probable=False,
                                    disable_genre_bits=False):
    """Generates a music sequence with the current model, and saves it to MIDI.

    The resulting MIDI file is saved to the model's output_dir directory. The
    sequence is generated by sampling from the output probabilities at each
    timestep, and feeding the resulting note back in as input to the model.

    Appends desired genre bits to each note generated by the model to ensure
    model continues generating according to the right genre.

    Args:
      genre: String name of the desired genre, e.g. 'classical'.
      priming_mode: If 'random', use a random note to prime the model. Can also
        be 'specific_note'.
      priming_note: If 'priming_mode' is 'specific_note', use this argument to
        pass the note used to prime the model.
      visualize_probs: If True, the function will plot the softmax
        probabilities of the model for each note that occur throughout the
        sequence. Useful for debugging.
      prob_image_name: The name of a file in which to save the softmax
        probability image. If None, the image will simply be displayed
      composition_length: The length of the sequence to be generated.
      most_probable: If True, instead of sampling each note in the sequence,
        the model will always choose the argmax, most probable note.
      disable_genre_bits: If True, the model will not append genre bits as it is
        composing.
    """
    if not disable_genre_bits:
      genre_id = self.genres.index(genre)
    else:
      genre_id = 0

    next_obs = self.get_initial_observation(
        genre_id,
        priming_mode=priming_mode,
        priming_note=priming_note,
        suppress_output=False,
        disable_genre_bits=disable_genre_bits)
    logging.info('Priming with note %s', np.argmax(next_obs))

    if not disable_genre_bits:
      next_obs = self.add_genre_to_note(next_obs, genre_id)

    lengths = np.full(1, 1, dtype=int)

    if visualize_probs:
      prob_image = np.zeros((self.one_hot_length, composition_length))

    generated_seq = [0] * composition_length
    for i in range(composition_length):
      input_batch = np.reshape(next_obs, (1, 1, self.one_hot_length))

      softmax, self.state_value = self.session.run(
          [self.softmax, self.state_tensor],
          {self.melody_sequence: input_batch,
           self.initial_state: self.state_value,
           self.lengths: lengths})
      softmax = np.reshape(softmax, (self.one_hot_length))

      if visualize_probs:
        prob_image[:, i] = softmax

      if most_probable:
        note_idx = np.argmax(softmax)
      else:
        note_idx = rl_rnn_ops.sample_softmax(softmax)

      generated_seq[i] = note_idx
      next_obs = np.array(
          rl_rnn_ops.make_onehot([note_idx], self.note_input_length)).flatten()
      if not disable_genre_bits:
        next_obs = self.add_genre_to_note(next_obs, genre_id)

    logging.info('Generated sequence: %s', generated_seq)

    melody = melodies_lib.MonophonicMelody()
    melody.steps_per_bar = self.steps_per_bar
    melody.from_event_list(
        rl_rnn_ops.decoder(generated_seq, self.transpose_amount))

    sequence = melody.to_sequence(bpm=self.bpm)
    filename = genre_ops.get_next_file_name(self.output_dir, genre + '_sample',
                                            'mid')
    midi_io.sequence_proto_to_midi_file(sequence, filename)

    logging.info('Wrote a melody to %s', filename)

    if visualize_probs:
      logging.info('Visualizing note selection probabilities:')
      plt.figure()
      plt.imshow(prob_image)
      plt.ylabel('Note probability')
      plt.xlabel('Time')
      if prob_image_name is not None:
        plt.savefig(self.output_dir + '/' + prob_image_name)
      else:
        plt.show()
