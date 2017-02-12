"""Code to evaluate how well an RL Tuner conforms to music theory rules."""

import numpy as np
import tensorflow as tf

import rl_tutor_ops

def compute_composition_stats(rl_tuner,
                              num_compositions=10000,
                              composition_length=32,
                              key=None,
                              tonic_note=rl_tutor_ops.C_MAJOR_TONIC):
  """Uses the model to create many compositions, stores statistics about them.

  Args:
    num_compositions: The number of compositions to create.
    composition_length: The number of beats in each composition.
    key: The numeric values of notes belonging to this key. Defaults to
      C-major if not provided.
    tonic_note: The tonic/1st note of the desired key.
  Returns:
    A dictionary containing the computed statistics about the compositions.
  """
  stat_dict = initialize_stat_dict()

  for i in range(num_compositions):
    stat_dict = compose_and_evaluate_piece(
        rl_tuner,
        stat_dict,
        composition_length=composition_length,
        key=key,
        tonic_note=tonic_note)
    if i % (num_compositions / 10) == 0:
      stat_dict['num_compositions'] = i
      stat_dict['total_notes'] = i * composition_length

  stat_dict['num_compositions'] = num_compositions
  stat_dict['total_notes'] = num_compositions * composition_length

  tf.logging.info(get_stat_dict_string(stat_dict))

  return stat_dict

# The following functions compute evaluation metrics to test whether the model
# trained successfully.
def get_stat_dict_string(stat_dict, print_interval_stats=True):
  """Makes string of interesting statistics from a composition stat_dict.

  Args:
    stat_dict: A dictionary storing statistics about a series of compositions.
    print_interval_stats: If True, print additional stats about the number of
      different intervals types.
  Returns:
    String containing several lines of formatted stats.
  """
  tot_notes = float(stat_dict['total_notes'])
  tot_comps = float(stat_dict['num_compositions'])

  return_str = 'Total compositions: ' + str(tot_comps) + '\n'
  return_str += 'Total notes:' + str(tot_notes) + '\n'

  return_str += '\tCompositions starting with tonic: '
  return_str += str(float(stat_dict['num_starting_tonic'])) + '\n'
  return_str += '\tCompositions with unique highest note:'
  return_str += str(float(stat_dict['num_high_unique'])) + '\n'
  return_str += '\tCompositions with unique lowest note:'
  return_str += str(float(stat_dict['num_low_unique'])) + '\n'
  return_str += '\tNumber of resolved leaps:'
  return_str += str(float(stat_dict['num_resolved_leaps'])) + '\n'
  return_str += '\tNumber of double leaps:'
  return_str += str(float(stat_dict['num_leap_twice'])) + '\n'
  return_str += '\tNotes not in key:' + str(float(
      stat_dict['notes_not_in_key'])) + '\n'
  return_str += '\tNotes in motif:' + str(float(
      stat_dict['notes_in_motif'])) + '\n'
  return_str += '\tNotes in repeated motif:'
  return_str += str(float(stat_dict['notes_in_repeated_motif'])) + '\n'
  return_str += '\tNotes excessively repeated:'
  return_str += str(float(stat_dict['num_repeated_notes'])) + '\n'
  return_str += '\n'

  num_resolved = float(stat_dict['num_resolved_leaps'])
  total_leaps = (float(stat_dict['num_leap_twice']) + num_resolved)
  if total_leaps > 0:
    percent_leaps_resolved = num_resolved / total_leaps
  else:
    percent_leaps_resolved = np.nan
  return_str += '\tPercent compositions starting with tonic:'
  return_str += str(stat_dict['num_starting_tonic'] / tot_comps) + '\n'
  return_str += '\tPercent compositions with unique highest note:'
  return_str += str(float(stat_dict['num_high_unique']) / tot_comps) + '\n'
  return_str += '\tPercent compositions with unique lowest note:'
  return_str += str(float(stat_dict['num_low_unique']) / tot_comps) + '\n'
  return_str += '\tPercent of leaps resolved:'
  return_str += str(percent_leaps_resolved) + '\n'
  return_str += '\tPercent notes not in key:'
  return_str += str(float(stat_dict['notes_not_in_key']) / tot_notes) + '\n'
  return_str += '\tPercent notes in motif:'
  return_str += str(float(stat_dict['notes_in_motif']) / tot_notes) + '\n'
  return_str += '\tPercent notes in repeated motif:'
  return_str += str(stat_dict['notes_in_repeated_motif'] / tot_notes) + '\n'
  return_str += '\tPercent notes excessively repeated:'
  return_str += str(stat_dict['num_repeated_notes'] / tot_notes) + '\n'
  return_str += '\n'

  for lag in [1, 2, 3]:
    avg_autocorr = np.nanmean(stat_dict['autocorrelation' + str(lag)])
    return_str += '\tAverage autocorrelation of lag' + str(lag) + ':'
    return_str += str(avg_autocorr) + '\n'

  if print_interval_stats:
    return_str += '\n'
    return_str += '\tAvg. num octave jumps per composition:'
    return_str += str(float(stat_dict['num_octave_jumps']) / tot_comps) + '\n'
    return_str += '\tAvg. num sevenths per composition:'
    return_str += str(float(stat_dict['num_sevenths']) / tot_comps) + '\n'
    return_str += '\tAvg. num fifths per composition:'
    return_str += str(float(stat_dict['num_fifths']) / tot_comps) + '\n'
    return_str += '\tAvg. num sixths per composition:'
    return_str += str(float(stat_dict['num_sixths']) / tot_comps) + '\n'
    return_str += '\tAvg. num fourths per composition:'
    return_str += str(float(stat_dict['num_fourths']) / tot_comps) + '\n'
    return_str += '\tAvg. num rest intervals per composition:'
    return_str += str(float(stat_dict['num_rest_intervals']) / tot_comps)
    return_str += '\n'
    return_str += '\tAvg. num seconds per composition:'
    return_str += str(float(stat_dict['num_seconds']) / tot_comps) + '\n'
    return_str += '\tAvg. num thirds per composition:'
    return_str += str(float(stat_dict['num_thirds']) / tot_comps) + '\n'
    return_str += '\tAvg. num in key preferred intervals per composition:'
    return_str += str(
        float(stat_dict['num_in_key_preferred_intervals']) / tot_comps) + '\n'
    return_str += '\tAvg. num special rest intervals per composition:'
    return_str += str(
        float(stat_dict['num_special_rest_intervals']) / tot_comps) + '\n'
  return_str += '\n'

  return return_str


def compose_and_evaluate_piece(rl_tuner,
                               stat_dict,
                               composition_length=32,
                               key=None,
                               tonic_note=rl_tutor_ops.C_MAJOR_TONIC,
                               sample_next_obs=True):
  """Composes a piece using the model, stores statistics about it in a dict.

  Args:
    stat_dict: A dictionary storing statistics about a series of compositions.
    composition_length: The number of beats in the composition.
    key: The numeric values of notes belonging to this key. Defaults to
      C-major if not provided.
    tonic_note: The tonic/1st note of the desired key.
    sample_next_obs: If True, each note will be sampled from the model's
      output distribution. If False, each note will be the one with maximum
      value according to the model.
  Returns:
    A dictionary updated to include statistics about the composition just
    created.
  """
  last_observation = rl_tuner.prime_internal_models()
  rl_tuner.reset_for_new_sequence()

  for _ in range(composition_length):
    if sample_next_obs:
      action, new_observation, reward_scores = rl_tuner.action(
          last_observation,
          0,
          enable_random=False,
          sample_next_obs=sample_next_obs)
    else:
      action, reward_scores = rl_tuner.action(
          last_observation,
          0,
          enable_random=False,
          sample_next_obs=sample_next_obs)
      new_observation = action

    action_note = np.argmax(action)
    obs_note = np.argmax(new_observation)

    # Compute note by note stats as it composes.
    stat_dict = add_interval_stat(rl_tuner, new_observation, stat_dict, key=key)
    stat_dict = add_in_key_stat(rl_tuner, obs_note, stat_dict, key=key)
    stat_dict = add_tonic_start_stat(
        rl_tuner, obs_note, stat_dict, tonic_note=tonic_note)
    stat_dict = add_repeating_note_stat(rl_tuner, obs_note, stat_dict)
    stat_dict = add_motif_stat(rl_tuner, new_observation, stat_dict)
    stat_dict = add_repeated_motif_stat(rl_tuner, new_observation, stat_dict)
    stat_dict = add_leap_stats(rl_tuner, new_observation, stat_dict)

    rl_tuner.generated_seq.append(np.argmax(new_observation))
    rl_tuner.generated_seq_step += 1
    last_observation = new_observation

  for lag in [1, 2, 3]:
    stat_dict['autocorrelation' + str(lag)].append(
        rl_tutor_ops.autocorrelate(rl_tuner.generated_seq, lag))

  add_high_low_unique_stats(rl_tuner, stat_dict)

  return stat_dict

def initialize_stat_dict():
  """Initializes a dictionary which will hold statistics about compositions.

  Returns:
    A dictionary containing the appropriate fields initialized to 0 or an
    empty list.
  """
  stat_dict = dict()

  for lag in [1, 2, 3]:
    stat_dict['autocorrelation' + str(lag)] = []

  stat_dict['notes_not_in_key'] = 0
  stat_dict['notes_in_motif'] = 0
  stat_dict['notes_in_repeated_motif'] = 0
  stat_dict['num_starting_tonic'] = 0
  stat_dict['num_repeated_notes'] = 0
  stat_dict['num_octave_jumps'] = 0
  stat_dict['num_fifths'] = 0
  stat_dict['num_thirds'] = 0
  stat_dict['num_sixths'] = 0
  stat_dict['num_seconds'] = 0
  stat_dict['num_fourths'] = 0
  stat_dict['num_sevenths'] = 0
  stat_dict['num_rest_intervals'] = 0
  stat_dict['num_special_rest_intervals'] = 0
  stat_dict['num_in_key_preferred_intervals'] = 0
  stat_dict['num_resolved_leaps'] = 0
  stat_dict['num_leap_twice'] = 0
  stat_dict['num_high_unique'] = 0
  stat_dict['num_low_unique'] = 0

  return stat_dict

def add_interval_stat(rl_tuner, action, stat_dict, key=None):
  """Computes the melodic interval just played and adds it to a stat dict.

  Args:
    action: One-hot encoding of the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
    key: The numeric values of notes belonging to this key. Defaults to
      C-major if not provided.
  Returns:
    A dictionary of composition statistics with fields updated to include new
    intervals.
  """
  interval, action_note, prev_note = rl_tuner.detect_sequential_interval(action, 
                                                                         key)

  if interval == 0:
    return stat_dict

  if interval == rl_tutor_ops.REST_INTERVAL:
    stat_dict['num_rest_intervals'] += 1
  elif interval == rl_tutor_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
    stat_dict['num_special_rest_intervals'] += 1
  elif interval > rl_tutor_ops.OCTAVE:
    stat_dict['num_octave_jumps'] += 1
  elif interval == (rl_tutor_ops.IN_KEY_FIFTH or 
                    interval == rl_tutor_ops.IN_KEY_THIRD):
    stat_dict['num_in_key_preferred_intervals'] += 1
  elif interval == rl_tutor_ops.FIFTH:
    stat_dict['num_fifths'] += 1
  elif interval == rl_tutor_ops.THIRD:
    stat_dict['num_thirds'] += 1
  elif interval == rl_tutor_ops.SIXTH:
    stat_dict['num_sixths'] += 1
  elif interval == rl_tutor_ops.SECOND:
    stat_dict['num_seconds'] += 1
  elif interval == rl_tutor_ops.FOURTH:
    stat_dict['num_fourths'] += 1
  elif interval == rl_tutor_ops.SEVENTH:
    stat_dict['num_sevenths'] += 1

  return stat_dict

def add_in_key_stat(rl_tuner, action_note, stat_dict, key=None):
  """Determines whether the note played was in key, and updates a stat dict.

  Args:
    action_note: An integer representing the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
    key: The numeric values of notes belonging to this key. Defaults to
      C-major if not provided.
  Returns:
    A dictionary of composition statistics with 'notes_not_in_key' field
    updated.
  """
  if key is None:
    key = rl_tutor_ops.C_MAJOR_KEY

  if action_note not in key:
    stat_dict['notes_not_in_key'] += 1

  return stat_dict

def add_tonic_start_stat(rl_tuner,
                         action_note,
                         stat_dict,
                         tonic_note=rl_tutor_ops.C_MAJOR_TONIC):
  """Updates stat dict based on whether composition started with the tonic.

  Args:
    action_note: An integer representing the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
    tonic_note: The tonic/1st note of the desired key.
  Returns:
    A dictionary of composition statistics with 'num_starting_tonic' field
    updated.
  """
  if rl_tuner.generated_seq_step == 0 and action_note == tonic_note:
    stat_dict['num_starting_tonic'] += 1
  return stat_dict

def add_repeating_note_stat(rl_tuner, action_note, stat_dict):
  """Updates stat dict if an excessively repeated note was played.

  Args:
    action_note: An integer representing the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
  Returns:
    A dictionary of composition statistics with 'num_repeated_notes' field
    updated.
  """
  if rl_tuner.detect_repeating_notes(action_note):
    stat_dict['num_repeated_notes'] += 1
  return stat_dict

def add_motif_stat(rl_tuner, action, stat_dict):
  """Updates stat dict if a motif was just played.

  Args:
    action: One-hot encoding of the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
  Returns:
    A dictionary of composition statistics with 'notes_in_motif' field
    updated.
  """
  composition = rl_tuner.generated_seq + [np.argmax(action)]
  motif, _ = rl_tuner.detect_last_motif(composition=composition)
  if motif is not None:
    stat_dict['notes_in_motif'] += 1
  return stat_dict

def add_repeated_motif_stat(rl_tuner, action, stat_dict):
  """Updates stat dict if a repeated motif was just played.

  Args:
    action: One-hot encoding of the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
  Returns:
    A dictionary of composition statistics with 'notes_in_repeated_motif'
    field updated.
  """
  is_repeated, _ = rl_tuner.detect_repeated_motif(action)
  if is_repeated:
    stat_dict['notes_in_repeated_motif'] += 1
  return stat_dict

def add_leap_stats(rl_tuner, action, stat_dict):
  """Updates stat dict if a melodic leap was just made or resolved.

  Args:
    action: One-hot encoding of the chosen action.
    stat_dict: A dictionary containing fields for statistics about
      compositions.
  Returns:
    A dictionary of composition statistics with leap-related fields updated.
  """
  leap_outcome = rl_tuner.detect_leap_up_back(action)
  if leap_outcome == rl_tutor_ops.LEAP_RESOLVED:
    stat_dict['num_resolved_leaps'] += 1
  elif leap_outcome == rl_tutor_ops.LEAP_DOUBLED:
    stat_dict['num_leap_twice'] += 1
  return stat_dict

def add_high_low_unique_stats(rl_tuner, stat_dict):
  """Updates stat dict if rl_tuner.generated_seq has unique extrema notes.

  Args:
    stat_dict: A dictionary containing fields for statistics about
      compositions.
  Returns:
    A dictionary of composition statistics with 'notes_in_repeated_motif'
    field updated.
  """
  if rl_tuner.detect_high_unique(rl_tuner.generated_seq):
    stat_dict['num_high_unique'] += 1
  if rl_tuner.detect_low_unique(rl_tuner.generated_seq):
    stat_dict['num_low_unique'] += 1

  return stat_dict

def debug_music_theory_reward(rl_tuner,
                              composition_length=32,
                              key=None,
                              tonic_note=rl_tutor_ops.C_MAJOR_TONIC,
                              sample_next_obs=True,
                              test_composition=None):
  """Composes a piece and prints rewards from music theory functions.

  Args:
    composition_length: Desired number of notes int he piece.
    key: The numeric values of notes belonging to this key. Defaults to C
      Major if not provided.    
    tonic_note: The tonic/1st note of the desired key.
    sample_next_obs: If True, the next observation will be sampled from 
      the model's output distribution. Otherwise the action with 
      maximum value will be chosen deterministically.
    test_composition: A composition (list of note values) can be passed 
      in, in which case the function will evaluate the rewards that would
      be received from playing this composition.
  """
  last_observation = rl_tuner.prime_internal_models()
  rl_tuner.reset_for_new_sequence()

  for i in range(composition_length):
    print "Last observation", np.argmax(last_observation)
    state = rl_tuner.q_network.state_value
    
    if sample_next_obs:
      action, new_observation, reward_scores = rl_tuner.action(
          last_observation,
          0,
          enable_random=False,
          sample_next_obs=sample_next_obs)
    else:
      action, reward_scores = rl_tuner.action(
          last_observation,
          0,
          enable_random=False,
          sample_next_obs=sample_next_obs)
      new_observation = action
    action_note = np.argmax(action)
    obs_note = np.argmax(new_observation)
    
    if test_composition is not None:
      obs_note = test_composition[i]
      new_observation = np.array(rl_tutor_ops.make_onehot(
        [obs_note],rl_tuner.num_actions)).flatten()
    
    composition = rl_tuner.generated_seq + [obs_note]
    
    print "New_observation", np.argmax(new_observation)
    print "Composition was", rl_tuner.generated_seq
    print "Action was", action_note
    print "New composition", composition
    print ""

    # note to rl_tuner: using new_observation rather than action because we want
    # stats about compositions, not what the model chooses to do
    note_rnn_reward = rl_tuner.reward_from_reward_rnn_scores(new_observation, 
                                                             reward_scores)
    print "Note RNN reward:", note_rnn_reward, "\n"

    print "Key, tonic, and non-repeating rewards:"
    key_reward = rl_tuner.reward_key(new_observation)
    if key_reward != 0.0:
      print "Key reward:", key_reward
    tonic_reward = rl_tuner.reward_tonic(new_observation)
    if tonic_reward != 0.0:
      print "Tonic note reward:", tonic_reward
    if rl_tuner.detect_repeating_notes(obs_note):
      print "Repeating notes detected!"
    print ""

    print "Interval reward:"
    rl_tuner.reward_preferred_intervals(new_observation, verbose=True)
    print ""

    print "Leap & motif rewards:"
    leap_reward = rl_tuner.reward_leap_up_back(new_observation, verbose=True)
    if leap_reward != 0.0:
      print "Leap reward is:", leap_reward
    last_motif, num_unique_notes = rl_tuner.detect_last_motif(composition)
    if last_motif is not None:
      print "Motif detected with", num_unique_notes, "notes"
    repeated_motif_exists, repeated_motif = rl_tuner.detect_repeated_motif(
      new_observation)
    if repeated_motif_exists:
      print "Repeated motif detected! it is:", repeated_motif  
    print ""

    for lag in [1, 2, 3]:
      print "Autocorr at lag", lag, rl_tutor_ops.autocorrelate(composition, lag)
    print ""

    rl_tuner.generated_seq.append(np.argmax(new_observation))
    rl_tuner.generated_seq_step += 1
    last_observation = new_observation

    print "-----------------------------"

  if rl_tuner.detect_high_unique(rl_tuner.generated_seq):
    print "Highest note is unique!"
  else:
    print "No unique highest note :("
  if rl_tuner.detect_low_unique(rl_tuner.generated_seq):
    print "Lowest note is unique!"
  else:
    print "No unique lowest note :("
