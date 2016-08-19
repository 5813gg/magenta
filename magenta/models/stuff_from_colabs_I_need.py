"""Melody Q stuff"""

# run rl-rnn
MELODY_CHECKPOINT_DIR = ''
hparams = rops.small_model_hparams()
MIDI_PRIMER = ''
PRIME_WITH_MIDI = False
OUTPUT_DIR = '/tmp/melodyq'
LOG_DIR = '/tmp/melodyq/melody_q_model'
TRAINING_DATA_PATH = ''
CHECKPOINT_PATH = ''

dqn_hparams = tf.HParams(random_action_probability=0.1,
                          store_every_nth=1,
                          train_every_nth=5,
                          minibatch_size=32,
                          discount_rate=0.5,
                          max_experience=100000,
                          target_network_update_rate=0.01)

reload(mq)
mq.reload_files()
reload(rops)

melody_q = mq.MelodyQNetwork(OUTPUT_DIR, LOG_DIR, MELODY_CHECKPOINT_DIR,
                             MIDI_PRIMER, dqn_hparams=dqn_hparams, 
                             reward_softmax=False, reward_scaler=2.0,
                             output_every_nth=1000, 
                             training_data_path=TRAINING_DATA_PATH,
                             custom_hparams=hparams,
                             num_notes_in_melody=64,
                             stochastic_observations=False)

reward_vars = melody_q.reward_rnn.variables()
q_vars = melody_q.q_network.variables()

reward1 = melody_q.session.run(reward_vars[0])
q1 = melody_q.session.run(q_vars[0])

print "diff:", np.sum((q1 - reward1)**2)

melody_q.generate_music_sequence(visualize_probs=True, title='rlrnn-pre-training', length=32)

melody_q.train(num_steps=2000000, exploration_period=1000000)

melody_q.plot_rewards()

melody_q.generate_music_sequence(visualize_probs=True, title='fjord_reward_rescaled')

melody_q.saver.save(melody_q.session, CHECKPOINT_PATH + 'fjord_reward_rescaled', global_step=len(melody_q.rewards_batched)*melody_q.output_every_nth)

# restore a checkpointed model
melody_q.saver.restore(melody_q.session,  CHECKPOINT_PATH + 'new_reward_weighting-4000000')

# test composition stats
stat_dict = model.compute_composition_stats(num_compositions=100000)

# debugging reward function
reload(mq)
mq.reload_files()

new_reward = mq.MelodyQNetwork(OUTPUT_DIR, LOG_DIR, MELODY_CHECKPOINT_DIR,
                               MIDI_PRIMER, reward_softmax=False, custom_hparams=hparams)

new_reward.reset_composition()
obs = new_reward.prime_q_model()
seq = [26, 26, 26, 0, 26, 0, 26, 0, 0, 27, 0, 22, 0, 0, 17, 0, 0, 10, 0, 0, 7, 0, 0, 12, 0, 0, 17, 0, 0, 10, 0, 0, 7, 0, 0, 15, 0, 0, 10, 0, 0, 7, 0, 0, 15, 0, 0, 10, 0, 0, 7, 0, 0, 15, 0, 0, 10, 0, 0, 7, 0, 0, 15, 0]
for action in seq:
  print "composition", new_reward.composition
  print "action:", action
  action_vector = rops.make_onehot([action], 38)
  reward = new_reward.collect_reward(obs, action_vector, new_reward.q_network.state_value, verbose=True)
  print "reward:", reward
  new_reward.composition.append(action)
  new_reward.beat += 1
  obs = action_vector
  print ""

seq = [15, 0, 5, 5, 11, 0, 16, 0, 9, 0, 16, 0, 0, 16, 0, 0, 24, 0, 0, 27, 0, 0, 20, 0, 0, 12, 0, 0, 16, 0, 0, 24, 0, 0, 5, 0, 0, 16, 0, 0, 24, 0, 0, 20, 0, 0, 10, 0, 0, 24, 0, 0, 20, 0, 0, 17, 0, 0, 20, 0, 0, 12, 0, 0]
for lag in [1,2,3]:
  print "lag", lag, "autocorrelation:", rops.autocorrelate(seq,lag)

# make a tensorboard event for looking at the graph
session = tf.Session(graph=melody_q.graph)
journalist = tf.train.SummaryWriter(LOG_DIR, graph=session.graph)
journalist.flush()

# other debugging stuff
def calc_diff_between_successive_states(melq):
  state_diffs = [0] * (len(melq.experience)-1)
  prev_state = melq.experience[0][1]
  for e in range(1,len(melq.experience)):
    next_state = melq.experience[e][1]
    diff = np.sum(abs(next_state - prev_state))
    print "Diff between state", e, "and state", e-1, "is", diff
    state_diffs[e-1]=diff
    prev_state = next_state
    
  print "Average difference:", np.mean(state_diffs)
    
  plt.figure()
  plt.plot(state_diffs)
  plt.show()
  
  return state_diffs

def calc_diff_between_paired_states(melq):
  diffs = [0] * len(melq.experience)
  for e in range(len(melq.experience)):
    next_state = melq.experience[e][5]
    prev_state = melq.experience[e][1]
    diff = np.sum(abs(next_state - prev_state))
    print "Diff at", e, "is", diff
    diffs[e]=diff
    
  plt.figure()
  plt.plot(diffs)
  plt.show()

def calc_diff_between_identical_states(melq):
  state_diffs = [0] * (len(melq.experience)-1)

  for e in range(1,len(melq.experience)):
    prev_next_state = melq.experience[e-1][5]
    current_state = melq.experience[e][1]
    diff = np.sum(abs(current_state - prev_next_state))
    print "Diff between state", e, "and next state at", e-1, "is", diff
    state_diffs[e-1]=diff
    
  plt.figure()
  plt.plot(state_diffs)
  plt.show()

# black magic for seeing what loaded from an import
dir(mq)
open('melody_q.py').read()

# read contents of checkpoint file
checkpoint_file = tf.train.latest_checkpoint(genre_classifier_checkpoint_dir)
reader = tf.train.NewCheckpointReader('model.ckpt-0')
print(reader.debug_string().decode("utf-8"))
type(reader.get_tensor('rnn_model/RNN/MultiRNNCell/Cell0/LSTMCell/B'))
[str(x) for x in reader.debug_string().decode("utf-8").split('\n')]


"""Genre Stuff"""

# train genre dqn
genre_classifier_checkpoint_dir = '/tmp/genre_classifier'
output_dir = '/tmp/genre_dqn'
genre_rnn_checkpoint_dir = ''
backup_checkpoint_file = 'model.ckpt-30000'
CHECKPOINT_PATH = ''

dqn_hparams = tf.HParams(random_action_probability=0.1,
                        store_every_nth=1,
                        train_every_nth=5,
                        minibatch_size=32,
                        discount_rate=0.95,
                        max_experience=1000000,
                        target_network_update_rate=0.01)

reload(genre_dqn)
genre_dqn.reload_files()

gdqn = genre_dqn.GenreDQN(genre_classifier_checkpoint_dir,
                          genre_rnn_checkpoint_dir, output_dir,
                          genre_rnn_backup_checkpoint_file=backup_checkpoint_file,
                          dqn_hparams=dqn_hparams, reward_scaler=5.0)

gdqn.generate_genre_music_sequence('classical', composition_length=192, visualize_probs=True)

gdqn.generate_genre_music_sequence('pop', composition_length=192, visualize_probs=True)

gdqn.train(num_steps=4000000, exploration_period=1000000)

gdqn.plot_rewards()

gdqn.saver.save(gdqn.session, CHECKPOINT_PATH + 'genre_dqn/debugged', global_step=len(gdqn.rewards_batched)*gdqn.output_every_nth)

# restore from checkpoint
gdqn.saver.restore(gdqn.session, 'debugged-4000000')

# debugging
def debug_genre_dqn(gdqn, num_steps=400):
  last_observation = gdqn.prime_q_model(suppress_output=False)
  gdqn.reset_composition()
  note_inputs = np.zeros((gdqn.input_size,gdqn.num_notes_in_melody))
  print "Desired genre is:", gdqn.composition_genre
  sys.stdout.flush()
  
  for i in range(num_steps):
    action = gdqn.action(last_observation,
                         num_steps/2,
                         enable_random=False,
                         sample_next_obs=False)
    note_inputs[:,gdqn.beat] = action
    #print "note_inputs[-2:,-1]", note_inputs[-2:,gdqn.beat]
    
    new_observation = action
    state = np.array(gdqn.q_network.state_value).flatten()
    reward = gdqn.collect_reward(last_observation, new_observation, state)
    gdqn.composition.append(np.argmax(new_observation))
    gdqn.beat += 1
    last_observation = new_observation

    #if i % gdqn.reward_every_n_notes == 0:
    if reward > 0:
      print "Received reward of", reward, "at iteration", i
      print "Desired genre is:", gdqn.composition_genre
      print "Melody is:", gdqn.composition
      
      genre_ops.plot_note_inputs(note_inputs)

    # Reset the state after each composition is complete.
    if gdqn.beat % gdqn.num_notes_in_melody == 0:
      last_observation = gdqn.prime_q_model()
      gdqn.reset_composition()
      print "resetting composition at step", i
      note_inputs = np.zeros((gdqn.input_size,gdqn.num_notes_in_melody))

def debug_genre_sequence_rewards(gdqn, seq, genre):
  gdqn.beat = 0
  gdqn.composition = [seq[0]]
  gdqn.composition_genre = genre
  last_observation = rl_rnn_ops.make_onehot([seq[0]], gdqn.note_input_length)
  last_observation = gdqn.q_network.add_genre_to_note(last_observation, genre)
  state = None # state is not actually used for anything
  
  for i in range(1,len(seq)):
    new_observation = rl_rnn_ops.make_onehot([seq[i]], gdqn.note_input_length)
    new_observation = gdqn.q_network.add_genre_to_note(new_observation, genre)
    reward = gdqn.collect_reward(last_observation, new_observation, state)
    
    gdqn.composition.append(seq[i])
    gdqn.beat += 1
    last_observation = new_observation
    
    if reward > 0:
      print "Received reward of", reward, "at iteration", i
      print "Desired genre is:", gdqn.composition_genre
      print "Melody is:", gdqn.composition

# test probabilities assigned by genre classifier to a classical generated melody (output both probs)

def get_genre_classifier_probs_for_seq(gdqn, seq, genre):
  # convert composition to sequence
  inputs = rl_rnn_ops.make_onehot(seq, gdqn.note_input_length)
  inputs = np.reshape(inputs, (1, -1, gdqn.note_input_length))
  lengths = [len(seq)]

  # feed sequence into genre classifier
  genre_probs = gdqn.session.run(
      [gdqn.genre_classifier.genre_probs],
      {gdqn.genre_classifier.melody_sequence: inputs,
       gdqn.genre_classifier.lengths: lengths})
  
  print "reward would be:", genre_probs[0][genre]

  return genre_probs

def pull_several_training_batches(genre_classifier, num_batches=50):
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=genre_classifier.session, coord=coord)
  
  sequences = []
  labels = []
  lengths = []
  
  num_classical = 0
  num_pop = 0
  
  for i in range(num_batches):
    
    seq, labs, lens = genre_classifier.session.run([genre_classifier.genre_sequence, 
                                                    genre_classifier.labels, 
                                                    genre_classifier.lengths])
    genre_bits = seq[0,0,-2:]
    print "genre_bits", genre_bits
    genre_id = np.argmax(genre_bits)
    if genre_id == 0:
      num_classical += 1
    else:
      num_pop += 1
    
    sequences.append(seq)
    labels.append(labs)
    lengths.append(lens)

  coord.request_stop()
  coord.join(threads)
  
  print "total classical:", num_classical
  print "total pop:", num_pop
  
  return sequences, labels, lengths


def plot_training_sequence(gdqn, seqs, i):
  genres = ['classical', 'pop']
  
  inputs = np.array(seqs[i])  
  one_seq = np.reshape(inputs, (-1, 40))
  one_seq = np.transpose(one_seq)
  
  genre_bits = one_seq[-2:,0]
  genre_id = np.argmax(genre_bits)
  print "genre_bits", genre_bits
  print "genre:", genres[genre_id]
  
  get_training_sequence_prob(gdqn, inputs, genre_id)
  
  genre_ops.plot_note_inputs(one_seq)

def get_training_sequence_prob(gdqn, inputs, genre):
  # convert composition to sequence
  lengths = [np.shape(inputs)[1]]
  print "input shape", np.shape(inputs)
  print "lengths:", lengths
  inputs = inputs[:,:,:38]
  
  # feed sequence into genre classifier
  genre_probs = gdqn.session.run(
      [gdqn.genre_classifier.genre_probs],
      {gdqn.genre_classifier.melody_sequence: inputs,
       gdqn.genre_classifier.lengths: lengths})
  
  print "genre_probs:", genre_probs
  print "reward would be:", genre_probs[0][genre]

# debugging loading from a checkpoint
reload(genre_dqn)
genre_dqn.reload_files()

gdqn = genre_dqn.GenreDQN(genre_classifier_checkpoint_dir, 
                          genre_rnn_checkpoint_dir, output_dir,
                          backup_checkpoint_file=backup_checkpoint_file)

# comment out the restoration code and store variables
q_vars = gdqn.q_network.variables()
classifier_vars = gdqn.genre_classifier.variables()
target_vars = gdqn.target_q_network.variables()

for var in q_vars:
  print var.name, var.get_shape()

gdqn.q_network.initialize_new_session()
gdqn.target_q_network.initialize_new_session()

q1 = gdqn.session.run(q_vars[0])
class1= gdqn.session.run(classifier_vars[0])
target1 = gdqn.session.run(target_vars[0])

print "diff", np.sum((q1 - target1)**2)
print "mean: q", np.mean(q1), "target", np.mean(target1), "classifier", np.mean(class1)
print "stdev: q", np.std(q1), "target", np.std(target1), "classifier", np.std(class1)

# restore variables
gdqn.q_network.initialize_and_restore(gdqn.session)
gdqn.target_q_network.initialize_and_restore(gdqn.session)
gdqn.reward_rnn.initialize_and_restore(gdqn.session)
gdqn.genre_classifier.load_from_checkpoint_into_existing_graph(gdqn.session, gdqn.genre_classifier_checkpoint_dir, 
                                                               gdqn.genre_classifier_checkpoint_scope)

q2 = gdqn.session.run(q_vars[0])
class2 = gdqn.session.run(classifier_vars[0])
reward2 = gdqn.session.run(gdqn.reward_rnn.variables()[0])
target2 = gdqn.session.run(target_vars[0])

# compare
print "diff between q and target", np.sum((q2 - target2)**2)
print "diff between q and reward", np.sum((q2 - reward2)**2)
print "mean: q", np.mean(q2), "target", np.mean(target2), "reward", np.mean(reward2), "classifier", np.mean(class2)
print "stdev: q", np.std(q2), "target", np.std(target2), "reward", np.std(reward2), "classifier", np.std(class2)

# train genre classifier
reload(gclass)
hparams = gclass.default_hparams()
training_path =  ''
output_dir = '/tmp/genre_classifier'

gclassifier = gclass.GenreClassifier(
        hparams=hparams, training_data_path=training_path,
        output_dir=output_dir, num_training_steps=10)

i=0
for step in gclassifier.training_loop():
  print step
  i += 1
  sys.stdout.flush()
  if i > 10:
    break

# test the encoder
for k in metadata:
  if not 'classical' in metadata[k].genre:
    seq = SOMEHOW GET SEQUENCE MAGICALLY
    quantized_seq = sequences_lib.QuantizedSequence()
    try:
      quantized_seq.from_note_sequence(seq, steps_per_beat=4)
    except:
      continue
    melodies = melodies_lib.extract_melodies(
          quantized_seq, min_bars=7, min_unique_pitches=5)
    if melodies:
      print k
      for g in metadata[k].genre:
        print g
      melody=melodies[0]
      encoder = genre_encoder_decoder.GenreMelodyEncoderDecoder(1,2)
      encoded_seq = encoder.encode(melody)

      break

for feat in encoded_seq.feature_lists.feature_list['inputs'].feature:
  bits = list(feat.float_list.value)
  print bits
  print sum(bits)
  print bits[-1]
  break

# debugging when gradients don't work
with gclassifier.graph.as_default():
  with tf.device(lambda op: ""):
    with tf.variable_scope(gclassifier.scope):
      trainable_variables = tf.trainable_variables()
      gradients = []
      for var in trainable_variables:
        print var.name
        grad = tf.gradients(gclassifier.cost, var)
        gradients.append(grad)
      #gradients, _ = tf.clip_by_global_norm(gradients, gclassifier.max_gradient_norm)
      #optimizer = tf.train.AdamOptimizer(gclassifier.learning_rate)
      #train_op = optimizer.apply_gradients(zip(gradients, trainable_variables))
