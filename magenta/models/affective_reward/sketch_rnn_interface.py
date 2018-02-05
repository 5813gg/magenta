"""TODO(natashajaques): DO NOT SUBMIT without one-line documentation for sketch_rnn_interface.

TODO(natashajaques): DO NOT SUBMIT without a detailed description of sketch_rnn_interface.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import svgwrite
from IPython.display import SVG, display

import sketch_rnn_utils
import sketch_rnn_train
import sketch_rnn_model

from absl import flags, app

FLAGS = flags.FLAGS

def reload_dependencies():
  reload(sketch_rnn_utils)
  reload(sketch_rnn_train)
  reload(sketch_rnn_model)

flags.DEFINE_string(
    'models_location', 'local',
    'Location of the sketch RNN models to use. Can be "local" or other.')
flags.DEFINE_string(
    'model_description', 'vanilla_sketch_rnn_local',
    'String description of the model to save in the filename.')
flags.DEFINE_integer(
    'image_batch_id', 1,
    'ID used in labeling batches of images corresponding to embedding csv')
flags.DEFINE_string(
    'svg_path', 'TODO/path/sketch_rnn/unprocessed_samples/',
    'Location where rendered svg files are saved for later upload to datastore.')
flags.DEFINE_float(
    'temperature', 0.5,
    'Temperature used when creating drawings. Lower is ~less entropy')
flags.DEFINE_integer(
    'num_repeats', 100,
    'Number of times to run the script')

FLAGS = tf.flags.FLAGS


LOCAL_MODELS = [{'name':'aaron_sheep', 'path':'aaron_sheep/layer_norm/'},
          {'name':'catbus', 'path':'catbus/lstm/'},
          {'name':'elephantpig', 'path':'elephantpig/lstm/'},
          {'name':'flamingo', 'path':'flamingo/lstm_uncond/'},
          {'name':'owl', 'path':'owl/lstm/'}]
LOCAL_MODELS_DIR = 'TODO/sketch_rnn/models/'

OTHER_MODEL_NAMES = ['cat', 'bird', 'bicycle', 'octopus', 'face', 'flamingo', 'cruise_ship', 'truck', 'pineapple', 'spider', 'mosquito', 'angel', 'butterfly', 'pig', 'garden', 'The_Mona_Lisa', 'crab', 'windmill', 'yoga', 'hedgehog', 'castle', 'ant', 'basket', 'chair', 'bridge', 'diving_board', 'firetruck', 'flower', 'owl', 'palm_tree', 'rain', 'skull', 'duck', 'snowflake', 'speedboat', 'sheep', 'scorpion', 'sea_turtle', 'pool', 'paintbrush', 'bee', 'backpack', 'ambulance', 'barn', 'bus', 'cactus', 'calendar', 'couch', 'hand', 'helicopter', 'lighthouse', 'lion', 'parrot', 'passport', 'peas', 'postcard', 'power_outlet', 'radio', 'snail', 'stove', 'strawberry', 'swan', 'swing_set', 'tiger', 'toothpaste', 'toothbrush', 'trombone', 'whale', 'tractor', 'squirrel', 'alarm_clock', 'bear', 'book', 'brain', 'bulldozer', 'dog', 'dolphin', 'elephant', 'eye', 'fan', 'fire_hydrant', 'frog', 'kangaroo', 'key', 'lantern', 'lobster', 'map', 'mermaid', 'monkey', 'penguin', 'rabbit', 'rhinoceros', 'rifle', 'roller_coaster', 'sandwich', 'steak']
OTHER_MODEL_NAMES = ['duck', 'crab', 'rhinoceros', 'penguin', 'rabbit', 'cat']
OTHER_MODELS_DIR = 'TODO/path/sketch_rnn/models_v2/'
OTHER_MODELS = [{'name':n, 'path':n+'/train/'} for n in OTHER_MODEL_NAMES]

def draw_strokes(data, factor=0.2,
                 svg_filename = '/tmp/sketch_rnn/svg/sample.svg',
                 show_drawing=True):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = sketch_rnn_utils.get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  svg_str = dwg.tostring()
  if show_drawing:
    display(SVG(svg_str))
  return svg_str


class SketchRNNInterface:
  def __init__(self, graph=None, models=LOCAL_MODELS, model_dir=LOCAL_MODELS_DIR,
               model_description='vanilla_sketch_rnn_local', initialize_immediately=True):
    if graph is None:
      self.graph = tf.Graph()
    else:
      self.graph = graph
    self.session = None
    self.models = models
    self.model_dir = model_dir
    self.model_names = []
    self.model_description = model_description

    if initialize_immediately:
      self.initialize_from_scratch()

  def find_model_num_by_name(self, model_name):
    return self.model_names.index(model_name)

  def initialize_from_scratch(self):
    self.add_models_to_graph()
    self.initialize_session()
    self.restore_models()

  def add_models_to_graph(self):
    self.model_names = [''] * len(self.models)
    for i, model in enumerate(self.models):
      self.model_names[i] = model['name']
      model_path = self.model_dir + model['path']
      print("Adding model", model['name'], "to the graph")
      [train_hps, eval_hps, sample_hps] = sketch_rnn_train.load_model(model_path)

      with self.graph.as_default():
        with tf.variable_scope(model['name']):
          # construct the sketch-rnn model here:
          self.models[i]['train_model'] = sketch_rnn_model.Model(train_hps)
          self.models[i]['eval_model'] = sketch_rnn_model.Model(eval_hps, reuse=True)
          self.models[i]['sample_model'] = sketch_rnn_model.Model(sample_hps, reuse=True)

  def initialize_session(self, session=None):
    with self.graph.as_default():
      if session is None:
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.initialize_all_variables())
      else:
        self.session = session

  def make_model_checkpoint_dict(self, vars, model_name):
    """Constructs a dict mapping the checkpoint variables to those in new graph.

    Returns:
      A dict mapping variable names in the checkpoint to variables in the graph.
    """
    var_dict = dict()
    for var in vars:
      inner_name = var.name.replace(model_name + '/', '')

      # Trim unnecessary postfixes
      idx = inner_name.find(':')
      inner_name = inner_name[:idx]

      var_dict[inner_name] = var
    return var_dict

  def restore_models(self):
    for model in self.models:
      self.load_model_into_subgraph(model)

  def load_model_into_subgraph(self, model_dict):
    checkpoint_path = self.model_dir + model_dict['path']
    print("Sketch RNN Interface looking for checkpoints at path", checkpoint_path)

    with self.graph.as_default():
      model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=model_dict['name'])
      checkpoint_dict = self.make_model_checkpoint_dict(model_vars,
                                                        model_dict['name'])
      saver = tf.train.Saver(var_list=checkpoint_dict)
      ckpt = tf.train.get_checkpoint_state(checkpoint_path)

      # Deal with crazy stupid bug where tf keeps a hardcoded checkpoint path
      # in a file so if you move the checkpoint it no longer works.
      if checkpoint_path not in ckpt.model_checkpoint_path:
        dirs = ckpt.model_checkpoint_path.split('/')
        model_checkpoint_path = checkpoint_path + dirs[-1]
      else:
        model_checkpoint_path = ckpt.model_checkpoint_path

      tf.logging.info('Loading model %s.', model_checkpoint_path)
      saver.restore(self.session, model_checkpoint_path)
      print('Successfully restored checkpoint ', ckpt)

  def generate_strokes_from_embedding(self, model, embedding, temperature=0.1):
    return self._generate_sample(model, temperature, z=[embedding])

  def generate_sample_strokes(self, model, temperature=0.1):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    z = np.random.randn(1, self.models[model]['sample_model'].hps.z_size)
    return self._generate_sample(model, temperature, z=z), z

  def _generate_sample(self, model, temperature=0.1, z=None):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    sample_model = self.models[model]['sample_model']
    eval_model = self.models[model]['eval_model']
    sample_strokes, _ = sketch_rnn_model.sample(
        self.session, sample_model, seq_len=eval_model.hps.max_seq_len,
        temperature=temperature, z=z)
    return sketch_rnn_utils.to_normal_strokes(sample_strokes)

  def encode_decode(self, model, input_strokes, draw_debug=False,
                    temperature=0.1):
    z = self.encode_strokes(model, input_strokes, draw_debug)
    return self.generate_strokes_from_embedding(model, z, temperature)

  def encode_strokes(self, model, input_strokes,
                                    draw_debug=False):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    eval_model = self.models[model]['eval_model']
    strokes = sketch_rnn_utils.to_big_strokes(input_strokes).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    if draw_debug:
      draw_strokes(sketch_rnn_utils.to_normal_strokes(np.array(strokes)))
    return self.session.run(eval_model.batch_z,
                            feed_dict={eval_model.input_data: [strokes],
                                       eval_model.sequence_lengths: seq_len})[0]

  def sample_and_draw(self, model, n=10, temperature=0.5, factor=0.2):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    reconstructions = []
    for i in range(n):
      strokes, z = self.generate_sample_strokes(model, temperature)
      reconstructions.append([strokes, [0, i]])

    stroke_grid = sketch_rnn_utils.make_grid_svg(reconstructions)
    draw_strokes(stroke_grid, factor)

  def draw_embeddings(self, model, embeddings, temperature=0.5, factor=0.2):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    reconstructions = []
    for i,z in enumerate(embeddings):
      strokes = self.generate_strokes_from_embedding(model, z, temperature)
      reconstructions.append([strokes, [0, i]])

    stroke_grid = sketch_rnn_utils.make_grid_svg(reconstructions)
    draw_strokes(stroke_grid, factor)

  def draw_embedding(self, model, embedding, temperature=0.5, factor=0.2):
    if type(model) != int:
      model = self.find_model_num_by_name(model)
    strokes = self.generate_strokes_from_embedding(model, embedding, temperature)
    draw_strokes(strokes, factor)

  def save_sample_to_file(self, model_class, svg_id=None, temperature=.1,
                          batch_id=0, show_drawing=False):
    strokes, z = self.generate_sample_strokes(model_class, temperature)
    if svg_id is None:
      svg_id = np.random.randint(0,1024)
    filename = model_class + '-' + str(batch_id) + '-' + self.model_description + '-' + str(svg_id) + '.svg'
    svg_str = draw_strokes(strokes,
                           svg_filename='/tmp/sketch_rnn/samples/'+filename,
                           show_drawing=show_drawing)

    open(FLAGS.svg_path + filename, 'w').write(svg_str) #TODO: check
    print('Saved image to path:', filename)
    return z

def initialize_model_serve_samples(models, model_dir, batch_id=0, start_id=0,
                                   num_samples_per_model=10,
                                   show_drawing=False, temperature=0.5):
  model = SketchRNNInterface(models=models, model_dir=model_dir,
                             model_description=FLAGS.model_description)

  df = pd.DataFrame()

  for j in range(num_samples_per_model):
    for i, name in enumerate(model.model_names):
      sample_num = j + start_id
      print('Generating a sample from model', name, 'and saving to file')
      z = model.save_sample_to_file(name, temperature=FLAGS.temperature,
                                    svg_id=sample_num, show_drawing=show_drawing,
                                    batch_id=batch_id)
      z_string = ','.join(['%.5f' % num for num in z.ravel()])
      df = df.append({'image_class': name, 'z_embedding': z_string,
                      'sample_id': sample_num}, ignore_index=True)
  print('Returning a df of length', len(df))
  return df


def serve_samples_model_batches(models, model_dir, current_batch_id=0,
                                batch_size=3):
  remaining_models = models
  big_df = None

  while remaining_models:
    models_batch = remaining_models[:batch_size]

    batch_df = initialize_model_serve_samples(models_batch, model_dir,
                                              current_batch_id, start_id=0)
    if big_df is None:
      big_df = batch_df
    else:
      big_df = pd.concat([big_df, batch_df])
    print('Total df length is now', len(big_df))
    remaining_models = remaining_models[batch_size:]

  csv_name = 'embeddings-' + FLAGS.model_description + '-' + str(current_batch_id) + '.csv'

  # Save df to csv..
  df_fname = os.path.join(FLAGS.svg_path, csv_name)
  with open(df_fname, 'w') as f: #TODO: check
    big_df.to_csv(f)

def get_model_info(models_location):
  if models_location == 'local':
    print('Using local models')
    model_dir = LOCAL_MODELS_DIR
    models = LOCAL_MODELS
  else:
    print('Using other models')
    model_dir = OTHER_MODELS_DIR
    models = OTHER_MODELS

  return models, model_dir

def main(unused_argv):
  models, model_dir = get_model_info(FLAGS.models_location)
  for i in range(FLAGS.num_repeats):
    current_batch_id = FLAGS.image_batch_id + i
    serve_samples_model_batches(models, model_dir, current_batch_id)
    num_left = FLAGS.num_repeats - i
    print('The script has run', i, 'times. Only', num_left, 'left to go')

if __name__ == '__main__':
  app.run(main)
