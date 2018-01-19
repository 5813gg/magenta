"""
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

import latent_gan

def reload_dependencies():
  reload(sketch_rnn_utils)
  reload(sketch_rnn_train)
  reload(sketch_rnn_model)

flags.DEFINE_string(
    'model_location', 'local',
    'Location of the sketch RNN models to use. Can be "local" or other.')
flags.DEFINE_string(
    'sketch_class', 'cat',
    'String description of the image class.')
flags.DEFINE_string(
    'base_model_description', 'latent_constraints',
    'String description of the model to save in the filename.')
flags.DEFINE_integer(
    'initial_batch_id', 1,
    'ID used in labeling batches of images corresponding to embedding csv')
flags.DEFINE_string(
    'sample_save_path', 'TODO/path/sketch_rnn/unprocessed_samples/',
    'Location where rendered svg files are saved for later upload to datastore.')
flags.DEFINE_integer(
    'num_runs', 5,
    'Number of times to run the script')

FLAGS = flags.FLAGS


MODEL_NAMES = ['crab', 'cat', 'rhinoceros', 'penguin']
LOCAL_MODELS_DIR = 'TODO/path/affective_reward/model_checkpoints/'
OTHER_MODELS_DIR = '/TODO/path/sketch_rnn/models_v2/'

def save_samples_to_files(svgs, model_class, model_description, svg_id_start=0, batch_id=0):
  for i in range(len(svgs)):
    svg_id = svg_id_start + i
    filename = model_class + '-' + str(batch_id) + '-' + model_description + '-' + str(svg_id) + '.svg'

    open(FLAGS.sample_save_path + filename, 'w').write(svgs[i]) #TODO: check
    print('Saved image to path:', filename)

def save_zs_to_df(zs, df, sketch_class, start_sample_id=0):
  for i in range(len(zs)):
    sample_num = start_sample_id + i
    z_string = ','.join(['%.5f' % num for num in zs[i,:].ravel()])
    df = df.append({'image_class': sketch_class, 'z_embedding': z_string,
                    'sample_id': sample_num}, ignore_index=True)
  return df

def initialize_model_serve_samples(sketch_class, model_dir, batch_id=0,
                                   start_id=0, num_batches_per_model=1):
  lgan = latent_gan.LatentSketchGAN(sketch_class=sketch_class,
                                    models=[sketch_class])

  checkpoint_path = model_dir + sketch_class + '/'
  lgan.load_checkpoint(checkpoint_path)

  df = pd.DataFrame()
  prior_df = pd.DataFrame()

  for j in range(num_batches_per_model):

      sample_num_start = start_id + j*lgan.hparams.batch_size

      print('Generating a batch of samples from model', sketch_class,
            'and saving to file')

      # Saving samples from trained LGAN
      zs, svgs = lgan.generate_sample_strokes_from_generator()
      save_samples_to_files(svgs, sketch_class, FLAGS.base_model_description,
                            svg_id_start=sample_num_start, batch_id=batch_id)
      df = save_zs_to_df(zs, df, sketch_class, sample_num_start)

      # Saving sample from prior
      zs, svgs = lgan.generate_sample_strokes_from_prior()
      save_samples_to_files(svgs, sketch_class, FLAGS.base_model_description+'_prior',
                            svg_id_start=sample_num_start, batch_id=batch_id)
      prior_df = save_zs_to_df(zs, prior_df, sketch_class, sample_num_start)

  print('Returning a df of length', len(df))
  return df, prior_df

def save_df_to_csv(df, csv_name):
  # Save df to csv.
  df_fname = os.path.join(FLAGS.sample_save_path, csv_name)
  with open(df_fname, 'w') as f: #TODO: check
    df.to_csv(f)

def serve_samples_model_batches(models, model_dir, current_batch_id=0,
                                batch_size=4):
  remaining_models = models
  big_df = None
  big_df_prior = None

  while remaining_models:
    models_batch = remaining_models[:batch_size]

    for model in models_batch:
      batch_df, batch_df_prior = initialize_model_serve_samples(
          model, model_dir, current_batch_id, start_id=0)

      if big_df is None:
        big_df = batch_df
      else:
        big_df = pd.concat([big_df, batch_df])

      if big_df_prior is None:
        big_df_prior = batch_df_prior
      else:
        big_df_prior = pd.concat([big_df_prior, batch_df_prior])

      print('Total df length is now', len(big_df))
    remaining_models = remaining_models[batch_size:]

  csv_name = 'embeddings-' + FLAGS.base_model_description + '-' + str(current_batch_id) + '.csv'
  save_df_to_csv(big_df, csv_name)

  csv_name_prior = 'embeddings-' + FLAGS.base_model_description + '_prior-' + str(current_batch_id) + '.csv'
  save_df_to_csv(big_df_prior, csv_name_prior)


def get_model_info(models_location):
  if models_location == 'local':
    print('Using local models')
    return LOCAL_MODELS_DIR
  else:
    print('Using other models')
    return OTHER_MODELS_DIR

def main(unused_argv):
  model_dir = get_model_info(FLAGS.model_location)
  for i in range(FLAGS.num_runs):
    current_batch_id = FLAGS.initial_batch_id + i
    serve_samples_model_batches(MODEL_NAMES, model_dir, current_batch_id)
    num_left = FLAGS.num_runs - i
    print('The script has run', i, 'times. Only', num_left, 'left to go')

if __name__ == '__main__':
  app.run(main)
