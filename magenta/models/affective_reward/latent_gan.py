import random
import os
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sonnet as snt

import sketch_rnn_interface

MODEL_NAMES = ['cat'] #'duck', 'crab', 'rhinoceros', 'penguin', 'rabbit',

LABEL_NAMES = ['user_normalized_avg_contentment',
               'user_normalized_avg_amusement',
               'user_normalized_max_contentment',
               'user_normalized_avg_concentration',
               'user_normalized_max_concentration',
               'user_normalized_avg_sadness']
FRIENDLY_LABEL_NAMES = ['avg_content', 'avg_amuse', 'max_content', 'avg_conc', 'max_conc', 'avg_sad']
LABEL_WEIGHTS = [0.8, 1.0, 0.7, -1.0, -1.0, -0.8]

def get_default_hparams():
  """Return default HParams for latent gan."""
  hparams = tf.contrib.training.HParams(
    batch_size=16,
    lr=3e-4,
    linear_size=1024,
    n_per_update_g=10,
    actor_loss_thresh = 1e-2,
    critic_loss_thresh = 1e-2,
    n_start = 5,
    n_avg=100,
    exp_uid='test',
    discriminator_pretraining_iters=100,
    n_iters=50,
  )
  return hparams

def t_avg(key):
  return np.mean(traces[key][-N_avg:])

def normalize01(data):
  maxd = max(data)
  mind = min(data)
  return (np.array(data) - mind) / (maxd-mind)


class Actor(snt.AbstractModule):
  '''Actor'''
  def __init__(self, n_latent, size=1024, name='actor'):
    super(Actor, self).__init__(name=name)
    self.n_latent = n_latent
    self.size = size

  def _build(self, z):
    x = z
    x = tf.nn.relu(snt.Linear(self.size)(x))
    x = tf.nn.relu(snt.Linear(self.size)(x))
    x = tf.nn.relu(snt.Linear(self.size)(x))
    x = snt.Linear(2*self.n_latent)(x)
    gates = tf.nn.sigmoid(x[:, :self.n_latent])
    dz = gates * x[:, self.n_latent:]
    z_prime = (1-gates)*z + dz
    return z_prime


class Critic(snt.AbstractModule):
  '''Critic'''
  def __init__(self, size=1024, n_outputs=1, name='critic'):
    super(Critic, self).__init__(name=name)
    self.size = size
    self.n_outputs = n_outputs

  def _build(self, z):
    x = z
    x = tf.nn.relu(snt.Linear(self.size)(x))
    x = tf.nn.relu(snt.Linear(self.size)(x))
    x = tf.nn.relu(snt.Linear(self.size)(x))
    v = tf.nn.sigmoid(snt.Linear(self.n_outputs)(x))
    return v

class LatentSketchGAN:
  def __init__(self, models=MODEL_NAMES,
               model_dir='TODO/path/sketch_rnn/models/continuous_gmm/',
               affect_data_file='TODO/path/affective_reward/csv_data/summarized_emotions_and_sketches-clean.csv',
               save_dir='TODO/path/affective_reward/model_checkpoints/',
               remove_experimenter_data=False,
               sketch_class='cat',
               sketch_temp=0.5,
               label_names=LABEL_NAMES,
               label_weights=LABEL_WEIGHTS,
               friendly_label_names=FRIENDLY_LABEL_NAMES,
               hparams=None):
    self.models = [{'name':n, 'path':n+'/train/'} for n in models]
    self.model_dir = model_dir
    print "Model dir is:", model_dir
    self.affect_data_file = affect_data_file
    self.sketch_class = sketch_class
    self.save_dir = os.path.join(save_dir, self.sketch_class)
    self.sketch_temp = sketch_temp
    self.remove_experimenter_data = remove_experimenter_data
    self.label_names = label_names
    self.label_weights = np.array(label_weights)
    self.friendly_label_names = friendly_label_names

    if hparams is None:
      self.hparams = get_default_hparams()
    else:
      self.hparams = hparams

    print "Loading Sketch RNN models..."
    self.sketch_obj = sketch_rnn_interface.SketchRNNInterface(
        models=self.models, model_dir=self.model_dir)
    self.n_latent = self.sketch_obj.models[0]['eval_model'].hps.z_size

    # Process collected emotion data
    self.load_emotion_data()

    print "\nBuilding graph..."
    self.penalty_weight_ = np.linspace(0.1, 1.0, self.hparams.n_iters)
    self.build_graph()
    self.traces = {"actor_loss":[],
                  "critic_loss":[],
                  "value":[],
                  "value_prime":[],
                  "value_prior":[],
                  "zmse":[],
                  "time":[],}

  def load_emotion_data(self):
    self.emo_df = pd.DataFrame.from_csv(self.affect_data_file)
    print "\nLoaded affect data from file. Found", len(self.emo_df), "rows."
    if self.remove_experimenter_data:
      # TODO: FIX: once UIDs were hashed they can be < 1
      self.emo_df = self.emo_df[self.emo_df['user_id'] >= 0]
      print "Removed experimenter data. There are now", len(self.emo_df), "rows."
    self.sketch_df = self.emo_df[self.emo_df['image_class'] == self.sketch_class]
    self.sketch_df = self.sketch_df.dropna(subset=self.label_names)

    print "There are", len(self.sketch_df), "samples for sketch class", self.sketch_class

    self.embeddings = self.sketch_df['embedding'].tolist()
    self.embeddings = [np.fromstring(x, sep=',') for x in self.embeddings]
    self.embeddings = np.array(self.embeddings)

    labels = []
    for label in self.label_names:
      labels.append(normalize01(self.sketch_df[label].tolist()))
    self.labels = np.array(labels).T

    self.n_labels = len(self.label_names)
    self.n_data = len(self.embeddings)

  def reset_session_graph(self):
    sess = tf.get_default_session()
    if sess:
      sess.close()
    tf.reset_default_graph()
    self.build_graph()
    self.traces = {"actor_loss":[],
                  "critic_loss":[],
                  "value":[],
                  "value_prime":[],
                  "value_prior":[],
                  "zmse":[],
                  "time":[],}

  def build_graph(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.sess = tf.Session()

      # Modules with Parameters
      self.actor = Actor(size=self.hparams.linear_size, n_latent=self.n_latent)
      self.critic = Critic(size=self.hparams.linear_size, n_outputs=self.n_labels)

      self.z = tf.placeholder(tf.float32,
                              shape=(self.hparams.batch_size, self.n_latent))
      self.z_prior = tf.random_normal([self.hparams.batch_size, self.n_latent])
      self.z_prime = self.actor(self.z_prior)

      self.v = self.critic(self.z)
      self.v_prior = self.critic(self.z_prior)
      self.v_prime = self.critic(self.z_prime)

      # Loss functions
      # clip values
      #v = tf.clip_by_value(v, 1e-5, 1-1e-5)
      #v_prime = tf.clip_by_value(v_prime, 1e-5, 1-1e-5)
      self.actor_loss_log = tf.log(self.v_prime)
      self.actor_loss = - self.actor_loss_log * self.label_weights

      # Reward labels (1 = good, 0 = bad)
      self.affects = tf.placeholder(tf.float32, shape=(self.hparams.batch_size, self.n_labels))
      self.critic_loss = - (self.affects * tf.log(self.v) + (1.0-self.affects) * tf.log(1.0 - self.v))
      #self.critic_loss = tf.losses.mean_squared_error(self.affects, self.v)

      self.critic_loss = tf.reduce_mean(self.critic_loss)
      self.actor_loss = tf.reduce_mean(self.actor_loss)

      # Distance Penalty
      # penalty = tf.reduce_mean((z_prime - z)**2, axis=-1)
      penalty = tf.log(1 + tf.reduce_mean((self.z_prime - self.z_prior)**2, axis=-1))
      self.penalty_weight = tf.constant(0.0)
      self.actor_loss += penalty * self.penalty_weight

      actor_vars = self.actor.get_variables()
      critic_vars = self.critic.get_variables()
      all_vars = list(actor_vars)
      all_vars.extend(critic_vars)

      with tf.variable_scope('optimizers'):
        self.global_step_critic = tf.Variable(0,name='global_step_critic',trainable=False)
        self.global_step_actor = tf.Variable(0,name='global_step_actor',trainable=False)
        self.train_critic = tf.train.AdamOptimizer(learning_rate=self.hparams.lr).minimize(self.critic_loss, var_list=critic_vars, global_step=self.global_step_critic)
        self.train_actor = tf.train.AdamOptimizer(learning_rate=self.hparams.lr).minimize(self.actor_loss, var_list=actor_vars, global_step=self.global_step_actor)

        self.saver = tf.train.Saver(all_vars, max_to_keep=100)

      opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizers')

      with tf.variable_scope('input_gradients'):
        self.z_transform = tf.get_variable('z_transform',
                                           shape=(self.hparams.batch_size,
                                                  self.n_latent),
                                           dtype=tf.float32)
        self.v_transform = self.critic(self.z_transform)
        self.v_transform = tf.clip_by_value(self.v_transform, 1e-15, 1-1e-15)
        self.transform_loss = tf.reduce_mean(- tf.log(self.v_transform) * self.label_weights)
        self.transform_lr = tf.constant(1e-2)
        self.train_transform = tf.train.AdamOptimizer(
            self.transform_lr).minimize(self.transform_loss,
                                        var_list=[self.z_transform])

      transform_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='input_gradients')

      self.sess.run(tf.initialize_variables(opt_vars))
      self.sess.run(tf.initialize_variables(transform_vars))
      self.sess.run(tf.initialize_variables(all_vars))

  def get_batch_data(self):
    mb_idx = np.random.choice(self.n_data, size=self.hparams.batch_size)
    return np.vstack(self.embeddings)[mb_idx], np.vstack(self.labels)[mb_idx]

  def train_discriminator(self, n_iters=100, plot=False, output_on=False):
    for i in range(n_iters):
      z_sample, label_sample = self.get_batch_data()

      feed_dict = {
          self.z: z_sample,
          self.affects: label_sample
      }
      (_,
       global_step_critic_,
       critic_loss_,
       v_)= self.sess.run([self.train_critic,
                           self.global_step_critic,
                           self.critic_loss,
                           self.v],
                          feed_dict)

      self.traces['critic_loss'].append(critic_loss_)
      self.traces['value'].append(np.mean(v_))

      if output_on and i % 10 == 0:
        print "Mean discriminator loss at iteration", i, np.mean(self.traces["critic_loss"][-10:])

    if plot:
      plt.figure()
      plt.plot(self.traces['critic_loss'])
      plt.show()

  def train_generator(self, n_iters=1, try_plotting=True, plot_every=1):
    for i in range(n_iters):
      z_ = np.random.randn(self.hparams.batch_size, self.n_latent)
      feed_dict = {
          self.z:z_,
          self.penalty_weight:self.penalty_weight_[len(self.traces['time'])],
      }
      (_,
       global_step_actor_,
       actor_loss_,
       z_prior_,
       z_prime_,
       v_prime_,
       v_prior_ )= self.sess.run([self.train_actor,
                                 self.global_step_actor,
                                 self.actor_loss,
                                 self.z_prior,
                                 self.z_prime,
                                 self.v_prime,
                                 self.v_prior],
                                feed_dict)


      self.traces['actor_loss'].append(actor_loss_)
      self.traces['value_prime'].append(np.mean(v_prime_))
      self.traces['value_prior'].append(np.mean(v_prior_))
      self.traces['zmse'].append(np.mean((z_prior_ - z_prime_)**2))

      if try_plotting and i % plot_every == 0:
        print "Drawings with average values:",
        for i in range(self.n_labels):
          print self.friendly_label_names[i], np.mean(v_prime_[:,i]),
        print "\n"
        self.sketch_obj.draw_embeddings(self.sketch_class, z_prime_,
                                        temperature=self.sketch_temp)

  def sample_from_generator(self):
    z_ = np.random.randn(self.hparams.batch_size, self.n_latent)
    feed_dict = {
          self.z:z_,
          self.penalty_weight:self.penalty_weight_[len(self.traces['time'])],
    }

    (z_prime_, v_prime_)= self.sess.run([self.z_prime, self.v_prime],
                                        feed_dict)
    return z_prime_, v_prime_

  def plot_from_generator(self):
    z_prime_, v_prime_ = self.sample_from_generator()

    self.plot_embeddings_and_avg_values(z_prime_, v_prime_)

  def plot_embeddings_and_avg_values(self, embeddings, values):
    print "Drawings with average values:",
    for i in range(self.n_labels):
      print self.friendly_label_names[i], np.mean(values[:,i]),
    print "\n"
    self.sketch_obj.draw_embeddings(self.sketch_class, embeddings,
                                    temperature=self.sketch_temp)

  def compare_generator_with_prior(self):
    print "Original Sketch RNN:"
    self.sketch_obj.sample_and_draw(self.sketch_class,
                                    n=self.hparams.batch_size,
                                    temperature=self.sketch_temp)

    print "\nGenerator:"
    self.plot_from_generator()

  def plot_traces(self):
    plt.figure(figsize=(32, 12))
    plt.subplot(711)
    plt.plot(self.traces['critic_loss'])
    plt.ylabel('critc_loss')
    plt.subplot(712)
    plt.plot(self.traces['actor_loss'])
    plt.ylabel('actor_loss')
    plt.subplot(713)
    plt.plot(self.traces['value'])
    plt.ylabel('value')
    plt.subplot(714)
    plt.plot(self.traces['value_prime'])
    plt.ylabel('value_prime')
    plt.subplot(715)
    plt.plot(self.traces['value_prior'])
    plt.ylabel('value_prior')
    plt.subplot(716)
    plt.plot(self.traces['zmse'])
    plt.ylabel('zmse')
    plt.subplot(717)
    plt.plot(self.traces['time'])
    plt.ylabel('time')

  def save_model(self):
    step = self.sess.run(self.global_step_critic)
    save_name = os.path.join(self.save_dir, 'latent_constraints_%s_%d.ckpt' % (self.sketch_class, step))
    self.saver.save(self.sess, save_name)

  def main_training_loop(self, output_every=1, plot_every=10, save_every=100):
    print "Pretraining samples:"
    self.sketch_obj.sample_and_draw(self.sketch_class,
                                    n=self.hparams.batch_size,
                                    temperature=self.sketch_temp)

    print "Pretraining discriminator..."
    self.train_discriminator(n_iters=self.hparams.discriminator_pretraining_iters, output_on=True)

    start = time.time()
    self.traces['time'].append(time.time() - start)

    for i in range(self.hparams.n_iters):

      if i % self.hparams.n_per_update_g == 0:
        self.train_generator(try_plotting=i%plot_every==0)

      self.train_discriminator(n_iters=1)

      if i % output_every == 0:
        print 'Step %4d, GLoss:%.2f, DLoss:%.2f, (v:%.2f, v_prime:%.2f, v_prior:%.2f) ZMSE:%.3f, Time:%.2f' % (
              i,
              np.mean(self.traces["actor_loss"][-self.hparams.n_avg:]),
              np.mean(self.traces["critic_loss"][-self.hparams.n_avg:]),
              np.mean(self.traces["value"][-self.hparams.n_avg:]),
              np.mean(self.traces["value_prime"][-self.hparams.n_avg:]),
              np.mean(self.traces["value_prior"][-self.hparams.n_avg:]),
              np.mean(self.traces["zmse"][-self.hparams.n_avg:]),
              self.traces['time'][-1])


      if i > 0 and i % save_every == 0:
        self.save_model()

    print "Final samples:"
    self.plot_from_generator()

    self.plot_traces()

  def test_discriminator_quality(self):
    embeddings = np.zeros((self.hparams.batch_size, self.n_latent))
    strokes = []
    for i in range(self.hparams.batch_size):
      curr_strokes, z = self.sketch_obj.generate_sample_strokes(
          self.sketch_class, temperature=self.sketch_temp)
      embeddings[i,:] = z
      strokes.append(curr_strokes)

    values = self.sess.run(self.v, feed_dict={self.z:embeddings})

    for i, v in enumerate(values):
      print "Predicted value:",
      sum_score = 0
      for l in range(self.n_labels):
        print self.friendly_label_names[l], v[l],
        sum_score += v[l] * self.label_weights[l]
      print "\tTotal score:", sum_score
      sketch_rnn_interface.draw_strokes(strokes[i])

  def test_data_quality(self):
    zs, labels = self.get_batch_data()
    for i, label in enumerate(labels):
      print "Real label:",
      sum_score = 0
      for l in range(self.n_labels):
        print self.friendly_label_names[l], label[l],
        sum_score += label[l] * self.label_weights[l]
      print "\tTotal score:", sum_score
      self.sketch_obj.draw_embedding(self.sketch_class, zs[i,:],
                                     temperature=self.sketch_temp)

  def descend_input_gradient(self, z_original_=None, lr=1e-1, n_opt=100,
                             v_threshold = 0.9):
    if z_original_ is None:
      z_original_ = np.random.randn(self.hparams.batch_size, self.n_latent)
    _ = self.sess.run(tf.assign(self.z_transform, z_original_))
    z_new = np.zeros([self.hparams.batch_size, self.n_latent])
    i_threshold = np.zeros([self.hparams.batch_size])
    z_trace = []
    for i in range(n_opt):
      (_, transform_loss_,
       z_transform_, v_transform_) = self.sess.run([self.train_transform,
                                                    self.transform_loss,
                                                    self.z_transform,
                                                    self.v_transform],
                                                  {self.transform_lr:lr})

      # Ensuring each z in the batch is classified as higher than a threshold
      z_trace.append(z_transform_)
      check_idx = np.where(i_threshold == 0)[0]
      if len(check_idx) == 0:
        break
      # need to fix for bad labels
      for idx in check_idx:
        count_good = 0
        for l in range(self.n_labels):
          if self.label_weights[l] > 0 and v_transform_[idx,l] > v_threshold*self.label_weights[l]:
            count_good += 1
          elif v_transform_[idx,l] < 1.0 - v_threshold:
            count_good += 1

          if count_good == self.n_labels:
            z_new[idx] = z_transform_[idx]
            i_threshold[idx] = i

      if i % 100 == 1:
        print( 'Step %d, '
              'NotConverged: %d, '
              'Loss: %0.3e, '
              'v: %0.3f, '
              'v_min: %0.3f, '
              'ZMSE: %0.3f'
              % (
                  i,
                  len(check_idx),
                  transform_loss_,
                  np.mean(v_transform_),
                  np.min(v_transform_),
                  np.mean((z_new - z_original_)**2),
             ))
        print "Average values:",
        for vi, name in enumerate(self.friendly_label_names):
          print name, np.mean(v_transform_[:,vi]),
        print "\n"


    check_idx = np.where(i_threshold == 0)[0]
    print('%d did not converge' % len(check_idx))
    for idx in check_idx:
      z_new[idx] = z_transform_[idx]

    values = self.sess.run(self.v, feed_dict={self.z:z_new})
    self.plot_embeddings_and_avg_values(z_new, values)

    return z_new, i_threshold, z_trace

  def load_checkpoint(self, checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    tf.logging.info('Successfully restored checkpoint')

  def generate_sample_strokes_from_generator(self):
    z_prime, v_prime = self.sample_from_generator()

    svgs = []
    for i in range(len(z_prime)):
      strokes = self.sketch_obj.generate_strokes_from_embedding(self.sketch_class,
                                                                z_prime[i,:],
                                                                self.sketch_temp)
      svgs.append(sketch_rnn_interface.draw_strokes(strokes,
                                                    show_drawing=False))
    return z_prime, svgs

  def generate_sample_strokes_from_prior(self):
    z_grid = np.zeros((self.hparams.batch_size, self.n_latent))
    svgs = []
    for i in range(self.hparams.batch_size):
      strokes, z = self.sketch_obj.generate_sample_strokes(self.sketch_class,
                                                           self.sketch_temp)
      z_grid[i,:] = z
      svgs.append(sketch_rnn_interface.draw_strokes(strokes,
                                                    show_drawing=False))
    return z_grid, svgs


def main(argv):
  del argv  # Unused.


if __name__ == '__main__':
  app.run(main)
