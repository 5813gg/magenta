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

import discrete_sketch_decoder as dsd

FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string(
    'models_dir', 'TODO/path/discrete_decoder/',
    'Location of the models to use.')
flags.DEFINE_string(
    'sketch_class', 'cat',
    'String description of the image class.')
flags.DEFINE_string(
    'base_model_description', 'discrete_decoder',
    'String description of the model to save in the filename.')
flags.DEFINE_integer(
    'initial_batch_id', 1,
    'ID used in labeling batches of images corresponding to embedding csv')
flags.DEFINE_string(
    'sample_save_path', 'TODO/path/unprocessed_samples/',
    'Location where rendered svg files are saved for later upload to datastore.')
flags.DEFINE_integer(
    'num_runs', 2,
    'Number of times to run the script')
tf.app.flags.DEFINE_boolean(
    'store_raw_data', False,
    'Set to true to store raw data in discrete format, rather than data from '
    'the model')
tf.app.flags.DEFINE_string(
    'data_dir',
    'TODO/path/sketch_rnn/data/',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_integer(
    'vocab_size', 256,
    'The number of outputs that will be mapped to the 256 image pixels. If less'
    'than 256, will get a lower resolution version.')
tf.app.flags.DEFINE_boolean(
    'relative_coords', True,
    'Set to true if using relative coordinates rather than absolute.')


MODEL_NAMES = ['cat', 'face', 'owl', 'duck'] #['crab', 'cat', 'rhinoceros', 'penguin', 'face', 'owl', 'dolphin', 'duck', 'frog', 'rabbit']

def save_samples_to_files(svgs, model_class, model_description, svg_id_start=0, batch_id=0):
  for i in range(len(svgs)):
    svg_id = svg_id_start + i
    filename = model_class + '-' + str(batch_id) + '-' + model_description + '-' + str(svg_id) + '.svg'

    open(FLAGS.sample_save_path + filename, 'w').write(svgs[i]) #TODO: check
    print('Saved image to path:', filename)

def save_zs_to_df(zs, df, sketch_class, start_sample_id=0):
  for i in range(len(zs)):
    sample_num = start_sample_id + i
    df = df.append({'image_class': sketch_class, 'z_embedding': zs[i],
                    'sample_id': sample_num}, ignore_index=True)
  return df

def initialize_model_serve_samples(sketch_class, model_dir, batch_id=0,
                                   start_id=0, num_sketches_per_model=20):
  # Initialize model graph.
  hps_model, hps_eval, hps_sample = dsd.get_hps_set(FLAGS.vocab_size,
                                                    FLAGS.relative_coords)
  dsd.reset_graph()
  model = dsd.DiscreteSketchRNNDecoder(hps_model)
  eval_model = dsd.DiscreteSketchRNNDecoder(hps_eval, reuse=True)
  sample_model = dsd.DiscreteSketchRNNDecoder(hps_sample, reuse=True)

  # Load checkpointed model.
  checkpoint_path = model_dir +sketch_class+'/' + sketch_class + '/train/'
  model.load_checkpoint(checkpoint_path)
  eval_model.load_checkpoint(checkpoint_path)
  sample_model.load_checkpoint(checkpoint_path)

  df = pd.DataFrame()

  print('Generating a batch of samples from model', sketch_class,
        'and saving to file')
  svgs = []
  for j in range(num_sketches_per_model):
    strokes = dsd.sample_sketch(sample_model, bias=0.2,
                                relative=FLAGS.relative_coords)
    if FLAGS.relative_coords:
      strokes = dsd.decode_relative_strokes(strokes, hps_model)
      svgs.append(dsd.draw_strokes_relative(strokes, show_drawing=False))
    else:
      svgs.append(dsd.draw_lines(strokes, show_drawing=False,
                               vocab_size=hps_model.vocab_size))

  save_samples_to_files(svgs, sketch_class, FLAGS.base_model_description,
                        svg_id_start=start_id, batch_id=batch_id)

  zs = [''] * len(svgs)
  df = save_zs_to_df(zs, df, sketch_class, start_id)

  print('Returning a df of length', len(df))
  return df

def serve_samples_raw_data(sketch_class, data_dir, batch_id=0, start_id=0):
  data_file = data_dir + sketch_class + '.simple_line.npz'
  hps_model, hps_eval, hps_sample = dsd.get_hps_set()
  loader = dsd.DiscreteDataLoader(data_file, hps_model)

  batch = loader.random_batch()
  svgs = []
  for i, strokes in enumerate(batch):
    svgs.append(dsd.draw_lines(strokes, show_drawing=False,
                              vocab_size=FLAGS.vocab_size))

  save_samples_to_files(svgs, sketch_class, 'discrete_data',
                        svg_id_start=start_id, batch_id=batch_id)

  zs = [''] * len(svgs)
  df = pd.DataFrame()
  df = save_zs_to_df(zs, df, sketch_class, start_id)

  print('Returning a df of length', len(df))
  return df

def save_df_to_csv(df, csv_name):
  # Save df to csv in directory.
  df_fname = os.path.join(FLAGS.sample_save_path, csv_name)
  with open(df_fname, 'w') as f: #TODO: check
    df.to_csv(f)

def serve_samples_model_batches(models, model_dir, current_batch_id=0,
                                use_raw_data=False, data_dir=None):
  big_df = None

  for model in models:
    if use_raw_data:
      batch_df = serve_samples_raw_data(model, data_dir, current_batch_id,
                                        start_id=0)
    else:
      batch_df = initialize_model_serve_samples(
          model, model_dir, current_batch_id, start_id=0)

    if big_df is None:
      big_df = batch_df
    else:
      big_df = pd.concat([big_df, batch_df])

    print('Total df length is now', len(big_df))

  if use_raw_data:
    model_descrip = 'discrete_data'
  else:
    model_descrip = FLAGS.base_model_description
  csv_name = 'embeddings-' + model_descrip + '-' + str(current_batch_id) + '.csv'
  save_df_to_csv(big_df, csv_name)


def main(unused_argv):
  if FLAGS.store_raw_data:
    print('Okay, will store samples from the data distribution instead of the '
          'model.')

  for i in range(FLAGS.num_runs):
    current_batch_id = FLAGS.initial_batch_id + i
    serve_samples_model_batches(MODEL_NAMES, FLAGS.models_dir, current_batch_id,
                                FLAGS.store_raw_data, FLAGS.data_dir)
    num_left = FLAGS.num_runs - i
    print('The script has run', i, 'times. Only', num_left, 'left to go')

if __name__ == '__main__':
  app.run(main)
