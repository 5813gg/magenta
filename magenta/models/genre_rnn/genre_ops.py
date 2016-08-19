"""Encoder functons for converting a melody list to SequenceExample.

Each encoder takes a melodies_lib.Melody object, and outputs a SequenceExample
proto for use in TensorFlow models.
"""

import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib
from magenta.protobuf import music_pb2

from ....genre_rnn import genre_encoder_decoder


MAX_SEQUENCE_LENGTH = 192
NUM_SPECIAL_EVENTS = 2


def get_record_writer(output_tfrecord):
  """Preps a tfrecord.RecordWriter instance to use an output directory.

  Args:
    output_tfrecord: A directory where the tfrecord file will be saved.

  Returns:
    A tfrecord.RecordWriter instance pointing to the output directory.
  """
  tfrecord_dir = strip_base_name(output_tfrecord)
  if exists(output_tfrecord):
    logging.info('Removing old tfrecord %s.', output_tfrecord)
    remove(output_tfrecord)
  elif not exists(tfrecord_dir):
    make_dirs(tfrecord_dir)
  logging.info('Opening tfrecord %s for write.', output_tfrecord)
  return tfrecord.RecordWriter(output_tfrecord)


def get_genre_dict():
  """Makes a dictionary mapping each genre in a list of songs to files of that genre.

  Returns:
    A dict mapping string genre names to lists of the IDs of songs of
    that genre.
  """
  metadata = read_a_bunch_of_sequences()

  genre_dict = dict()

  i = 0
  for k in metadata:
    for g in metadata[k].genre:
      genre = str(g)
      if genre not in genre_dict:
        genre_dict[genre] = []
      genre_dict[genre].append(k)
      i += 1

  return genre_dict


def write_shuffle_queue(queue, record_writer):
  """Writes a list of sequence inputs and outputs to a RecordWriter instance.

  Args:
    queue: A list of tuples of (inputs, labels).
    record_writer: A tfrecord.RecordWriter instance.
  """
  random.shuffle(queue)
  for seq in queue:
    record_writer.WriteRecord(seq.SerializeToString())


def truncate_melodies_to_max_length(melodies, max_length=MAX_SEQUENCE_LENGTH):
  """Limits the length of each melody in a list of melody by splitting them.

  Args:
    melodies: A list of MonophonicMelody instances.
    max_length: The maximum length allowed for any sequence.

  Returns:
    A list of melodies that is now longer, of which none is larger than
    max)length.
  """
  new_melody_list = []

  for melody in melodies:
    while len(melody) > max_length:
      remaining_events = melody.events[max_length:]

      # truncates the melody to max_length
      melody.set_length(max_length)

      new_melody_list.append(copy.deepcopy(melody))

      melody.from_event_list(remaining_events)

    new_melody_list.append(melody)

  return new_melody_list


def save_genre_dataset(output_tfrecord,
                       desired_genres=None,
                       steps_per_bar=16,
                       min_bars=7,
                       min_unique_notes=5,
                       verbose=True,
                       melody_limit=None,
                       test=False):
  """Reads MIDI files from a magic place, makes a dataset of genre melodies.

  Extracts melodies from the MIDI files, and then encodes them using a special
  encoder that appends bits to the end of each input indicating the genre of
  the melody.

  Ensures that there are equal numbers of each genre.

  Args:
    output_tfrecord: A path to the tfrecord file where the melodies will be
      stored.
    desired_genres: The string names of the genres to keep. If None, defaults
      to classical and pop.
    steps_per_bar: The number of quantization steps in one bar of the melody.
    min_bars: The number of bars that must be in the extracted Melody in order
      to save it to the tfrecord file.
    min_unique_notes: The number of unique notes that must be in the extracted
      Melody in order to save it.
    verbose: If True, will output many logging statements.
    melody_limit: If provided, the method will stop extracting melodies after it
      has reached melody_limit.
    test: If True, will save a file consisting of only a few melodies for
      testing purposes.

  Returns:
    None unless test is True, in which case it returns the total number of
    melodies and notes found in the first 50 files.
  """
  if desired_genres is None:
    desired_genres = ['classical', 'pop']

  steps_per_beat = steps_per_bar / 4

  if verbose:
    logging.info('Setting up record writer for tfrecord file %s',
                 output_tfrecord)
  record_writer = get_record_writer(output_tfrecord)

  genre_dict = get_genre_dict()
  if verbose:
    logging.info('\nConstructed genre dict with %s genres', len(genre_dict))

  # limit the number of songs in each genre to be at most the number in the
  # genre with the fewest songs
  min_n = None
  for g in desired_genres:
    if min_n is None:
      min_n = len(genre_dict[g])
    elif len(genre_dict[g]) < min_n:
      min_n = len(genre_dict[g])
  if verbose:
    logging.info('Minimum number of songs in any genre is %s', min_n)

  sample_index_dict = dict()
  for g in desired_genres:
    if len(genre_dict[g]) == min_n:
      sample_index_dict[g] = np.arange(0, min_n)
      if verbose:
        logging.info('The genre with the fewest songs is %s', g)
    else:
      sample_index_dict[g] = np.random.choice(
          np.arange(0, len(genre_dict[g])), min_n)

  # loop through the songs in all genres and extract melodies and encode them,
  # occasionally writing to the tfrecord file
  num_genres = len(desired_genres)
  total_files = min_n * num_genres
  progress = 0
  total_melodies = 0
  total_notes = 0
  total_unreadable = 0
  shuffle_queue = []

  for i in range(min_n):
    for g in desired_genres:
      progress += 1

      genre_id = desired_genres.index(g)
      sample_id = sample_index_dict[g][i]
      table_id = genre_dict[g][sample_id]

      try:
        seq = read_sequence_magically(table_id)
      except
        total_unreadable += 1
        continue

      quantized_seq = sequences_lib.QuantizedSequence()
      try:
        quantized_seq.from_note_sequence(seq, steps_per_beat=steps_per_beat)
      except:
        total_unreadable += 1
        continue
      melodies = melodies_lib.extract_melodies(
          quantized_seq, min_bars=min_bars, min_unique_pitches=min_unique_notes)

      file_notes = 0
      file_melodies = 0
      if melodies:
        melodies = truncate_melodies_to_max_length(melodies)

        file_melodies = len(melodies)
        for melody in melodies:
          file_notes += len(melody)

          encoder = genre_encoder_decoder.GenreMelodyEncoderDecoder(genre_id,
                                                                    num_genres)
          seq = encoder.encode(melody)

          shuffle_queue.append(seq)
          if len(shuffle_queue) >= 1000:
            write_shuffle_queue(shuffle_queue, record_writer)
            shuffle_queue = []

          total_melodies += file_melodies
          total_notes += file_notes

      logging.info('Progress: %d of %d (%.1f%%) - Totals: %d melodies, '
                   '%d notes - File: %d melodies, %s notes - %s', progress,
                   total_files, 100.0 * progress / total_files, total_melodies,
                   total_notes, file_melodies, str(file_notes).rjust(4),
                   table_id)
      if verbose and i % 50 == 0:
        prog_prcnt = 100.0 * progress / total_files
        print 'Progress:', progress, '/', total_files, 'files', prog_prcnt, '%'
        print '\tTotal:', total_melodies, 'melodies,', total_notes, 'notes'
        print '\tFile:', file_melodies, 'melodies,', file_notes, 'notes'
        print '\tTotal unreadable:', total_unreadable

    if test and i >= 50:
      break

    if melody_limit is not None and total_melodies >= melody_limit:
      logging.info('Melody limit reached. Stopping.')
      break

  write_shuffle_queue(shuffle_queue, record_writer)
  notes_per_melody = total_notes / total_melodies if total_melodies else 0
  bars_per_melody = notes_per_melody / 16
  time_per_melody = '%d:%d' % (bars_per_melody * 2 / 60,
                               bars_per_melody * 2 % 60)
  logging.info('Totals: %d melodies, %d notes - '
               'Average Melody: %d notes, %d bars, %s time at 120 bpm.',
               total_melodies, total_notes, notes_per_melody, bars_per_melody,
               time_per_melody)
  if verbose:
    print 'DONE! :D'
    print '\tTotal melodies:', total_melodies
    print '\tTotal notes:', total_notes
    print '\tTotal unread:', total_unreadable, '-', total_unreadable / float(
        total_files), '%'
    print '\tAverage notes per melody:', notes_per_melody
    print '\tAverage bars per melody:', bars_per_melody
    print '\tAverage time per melody:', time_per_melody

  if test:
    return total_melodies, total_notes


def get_existing_runs(base_dir):
  """Gets a list of numerical indices of run directories for training a model.

  Args:
    base_dir: A path to the parent directory containing run folders.

  Returns:
    A list integer run numbers.
  """
  try:
    subdirs = get_all_the_subdirs()
  except StopIteration:
    return []
  return [int(s[3:]) for s in subdirs if s.startswith('run')]


def get_next_run_dir(base_dir):
  """Given a parent directory, finds the index of the next run folder.

  Args:
    base_dir: A path to the parent directory containing run folders.

  Returns:
    A string path to the next run directory.
  """
  runs = get_existing_runs(base_dir)
  return base_dir + ('/run' + str(max(runs) + 1) if runs else '/run1')


def get_next_file_name(directory, prefix, extension):
  """Finds the next sequential file name given a directory and name prefix.

  Eg. If prefix is 'song', extension is '.mid', and 'directory' already contains
  'song.mid' and 'song1.mid', function will return song2.mid'.

  Args:
    directory: A path to the parent directory in which to store the file.
    prefix: A filename prefix.
    extension: A file extension, e.g. '.mid'.

  Returns:
    A string file name.
  """
  prefix_name = os.path.join(directory, prefix)
  name = prefix_name + '.' + extension
  i = 0
  while os.path.isfile(name):
    i += 1
    name = prefix_name + str(i) + '.' + extension
  return name


def plot_composition_with_genre(seq, genre, total_bits, note_bits):
  """Takes in a sequence of notes and a genre and plots an image of it.

  Args:
    seq: A list of integer notes.
    genre: An integer representing the genre ID.
    total_bits: The total number of bits used to represent the note and genre.
    note_bits: The number of bits representing the note and not the genre.
  """
  comp_image = np.zeros((total_bits, len(seq)))

  genre_bit = note_bits + genre

  comp_image[genre_bit, :] = 1.0

  for i in range(len(seq)):
    comp_image[seq(i), i] = 1.0

  plt.figure()
  plt.imshow(comp_image)
  plt.ylabel('Note input')
  plt.xlabel('Time')
  plt.show()


def plot_note_inputs(note_inputs):
  """Plots a matrix of binary note input vectors as an image.

  Args:
    note_inputs: A matrix representing a series of binary note input vectors
      stored as columns.
  """
  plt.figure()
  plt.imshow(note_inputs)
  plt.ylabel('Note input')
  plt.xlabel('Time')
  plt.show()
