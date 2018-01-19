"""Moves data between tensorflow/directory world and Google App Engine datastore.

Can access the GAE datastore, read the data collected there, and put it
into CSV files in a directory. Can also sketch image files / samples from the model
from a directory and stores them into the datastore.

Can also be used to find out how much data has been collected from the app.

If the store_data flag is set to true, it will look for data in a directory to save to
the datastore. Otherwise, it will pull data from GAE and save it to csv.
"""

import datetime
import json
import logging
import os
import time
import tempfile
import numpy as np
import pandas as pd

from google.cloud import datastore # This is available externally: https://cloud.google.com/datastore/docs/reference/libraries#client-libraries-usage-python

import process_object_response

# TODO either replace with tf.logging or change logger name
_LOG = logging.getLogger(
    'learning.brain.research.affective_reward.models.datastore_interface')
_DATASTORE_SCOPE = 'https://www.googleapis.com/auth/datastore'

flags = tf.app.flags.FLAGS

flags.DEFINE_string(
    'creds_file', 'TODO/path/affective_reward_webapp_credentials.json',
    'Location of a JSON credentials file used to access the datastore. If '
    'empty, will use the default (defined below).')
flags.DEFINE_string(
    'sample_read_path', 'TODO/path/unprocessed_samples/',
    'Location where rendered svg files are saved for later upload to DB.')
flags.DEFINE_string(
    'sample_write_path', 'TODO/path/saved_samples/',
    'Location where svg files are stored once uploaded to datastore.')
flags.DEFINE_string(
    'csv_write_path', 'TODO/path/csv_data/',
    'Location where csv files are stored after extracting datastore emotions.')
flags.DEFINE_bool('store_data', False, 'Use this flag to write samples to the '
                  'datastore rather than reading it and creating csv files.')
flags.DEFINE_integer(
    'pandas_row_limit', 1000000,
    'When this row limit is reached, output the current dataframe and start a '
    'new one.')

FLAGS = flags.FLAGS


def get_datastore_client(creds=None):
  """Initializes a client that can talk to the datastore using a creds file.

  Args:
    creds: String path location of the JSON credentials file.
  Returns:
    Datastore client object.
  """
  input_file = # TODO .Open(creds, 'r')
  read_json = json.load(input_file)

  temp_path = tempfile.mkdtemp()
  tmp_cred_file = os.path.join(temp_path, 'cred.json')
  with open(tmp_cred_file, 'wb') as cred_json:
    json.dump(read_json, cred_json, ensure_ascii=False)

  client = datastore.Client.from_service_account_json(tmp_cred_file)
  _LOG.info('Successfully connected to datastore')
  return client


def query_db(client=None,
             kind='EmotionMetrics',
             timestr='timestamp',
             limit=None):
  """Gets most recent db entries of the stated kind and return them as a list.

  Each element of the list will be an Entity object. entity['amusement'] will
  give the value of a specific named property. entity.key will give the key,
  which is itself an object with values like id. So entity.key.id gives the ID

  Args:
    client: Datastore client object.
    kind: The type of datastore model to query.
    timestr: String name of the timestamp field to order by.
    limit: The maximum number of entities to return.
  Returns:
    A list of datastore entity objects.
  """
  if client is None:
    client = get_datastore_client(FLAGS.creds_file)

  query = client.query(kind=kind)
  query.order = ['-' + timestr]
  if limit:
    results = list(query.fetch(limit=limit))
  else:
    results = list(query.fetch())

  return results


def print_all_emotion_metrics(client):
  results = query_db(client, kind='EmotionMetrics')
  for e in results:
    _LOG.info('Key: %s, timestamp: %s, session ID: %s, amusement: %s', e.key.id,
              e['timestamp'], e['user_id'], e['amusement'])


def remove_emotion_metrics_with_no_experiment_name(client):
  _LOG.info('Removing all emotion metrics with no experiment name')
  num_removed = 0
  results = query_db(client)
  for e in results:
    if 'experiment_name' not in e.keys():
      e.key.delete()
      num_removed += 1
  _LOG.info('Removed a total of %s rows', num_removed)


def get_csv_files_from_emotion_metrics(client, relevant_experiments='default'):
  """Retrieves emotion metrics in the DB and makes a csv for each experiment.

  Args:
    client: A datastore client that was previously initialized.
    relevant_experiments: A list of experiments for which to actually create
      csvs and print statistics. Can be None to get all the experiments, or
      'default' to get a list that are likely to be relevant.
  """
  if relevant_experiments == 'default':
    relevant_experiments = [
        u'initial_scribble', u'demo', u'scribble', u'latent_constraints',
        u'normal', u'pre_launch_testing', u'initial_launch', u'testing'
    ]

  _LOG.info('\n\nGetting all emotion metrics from db')
  query = client.query(kind='EmotionMetrics')
  query.order = ['timestamp']
  results = list(query.fetch())
  _LOG.info('Found %s results from querying the db', len(results))

  experiment_df_dict = {}
  experiment_batch_dict = {}

  num_processed = 0
  num_batches = 0
  while results:
    e = results.pop()
    try:
      exp = e['experiment_name']
      if relevant_experiments and exp not in relevant_experiments:
        # Skip data from irrelevant experiments.
        continue
      elif exp not in experiment_df_dict:
        experiment_df_dict[exp] = df = pd.DataFrame()
        experiment_batch_dict[exp] = 0
        _LOG.info('Found another experiment: %s', exp)

      df = experiment_df_dict[exp]
      e_dict = convert_entity_to_dict(e)
      df = df.append(e_dict, ignore_index=True)
      num_processed += 1
      if num_processed % 100 == 0:
        _LOG.info('Have now processed %s rows', num_processed)
      if num_processed >= FLAGS.pandas_row_limit:
        batch_str = str(experiment_batch_dict[exp])
        _LOG.info('Reached the max number of rows. Outputting interim df '
                  'for batch %s.', batch_str)
        filename = 'emotion_metrics-' + exp + '-' + batch_str + '.csv'
        store_csv(df, filename)
        experiment_batch_dict[exp] += 1
        extract_summary_emotion_statistics(df, num_batches)
        df = pd.DataFrame()
      experiment_df_dict[exp] = df
    except:
      _LOG.info('Error! Could not process emotion entity %s %s', num_processed,
                e)

  _LOG.info('Processed a total of %s records', num_processed)
  _LOG.info('Found the following experiments: %s', experiment_df_dict.keys())

  for experiment_name in experiment_df_dict:
    num_batches = experiment_batch_dict[experiment_name]
    filename = (
        'emotion_metrics-' + experiment_name + '-' + str(num_batches) + '.csv')
    store_csv(experiment_df_dict[experiment_name], filename)
    print_emotion_df_stats(experiment_df_dict[experiment_name], experiment_name)
    extract_summary_emotion_statistics(experiment_df_dict[experiment_name],
                                       experiment_name, num_batches)


def store_csv(df, filename):
  df_fname = os.path.join(FLAGS.csv_write_path, filename)
  with open(df_fname, 'w') as f:
    df.to_csv(f)
  _LOG.info('Stored csv file %s into directory %s', filename,
            FLAGS.csv_write_path)


def print_emotion_df_stats(df, experiment_name):
  """Prints info about the number of clean records in an emotion recording df.

  Args:
    df: A pandas dataframe containing recordings of emotions.
    experiment_name: A string name describing the experiment in which the
      recordings were made.
  """
  _LOG.info('Experiment %s has:', experiment_name)
  _LOG.info('\t %s total records', len(df))
  num_users = len(df['user_id'].unique())
  _LOG.info('\t %s total users', num_users)
  _LOG.info('\t %s unique episodes', compute_unique_episodes_in_emotion_df(df))

  testing_users = [x for x in df['user_id'].unique() if x > -5 and x < 0]
  _LOG.info('\t %s testing users, which are: %s', len(testing_users),
            testing_users)
  _LOG.info('\t %s clean users', num_users - len(testing_users))

  strip_df = df.copy()
  for user in testing_users:
    strip_df = strip_df[strip_df['user_id'] != user]
  _LOG.info('\t %s clean rows', len(strip_df))
  _LOG.info('\t %s unique episodes',
            compute_unique_episodes_in_emotion_df(strip_df))


def compute_unique_episodes_in_emotion_df(df):
  unique_episodes = 0
  for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user]
    unique_episodes += len(user_df['episode_id'].unique())
  return unique_episodes


def extract_summary_emotion_statistics(df, experiment_name, num_batches=0):
  """Summarizes a df of emotion recordings by episode and user id.

  Given a df with 100ms emotion recordings over different episodes, computes
  statistics about the mean and max emotions per each episode, normalized
  by all recordings for that user.

  Args:
    df: A pandas dataframe object.
    experiment_name: A string description of the experiment name.
    num_batches: An integer number of batches that have come before this one.
      Used in saving the filename.
  """
  emotions = [
      'amusement', 'anger', 'concentration', 'contentment', 'desire',
      'disappointment', 'disgust', 'elation', 'embarrassment', 'interest',
      'pride', 'sadness', 'surprise'
  ]
  additional_columns = [
      'experiment_description', 'experiment_name', 'image_id', 'sample_id',
      'time_since_sample_displayed', 'timestamp'
  ]

  # Check that this experiment has all the necessary field for the below
  # computation.
  set_df = set(df.columns.values)
  set_required = set(emotions + additional_columns)
  if len(set_df.intersection(set_required)) != len(set_required):
    _LOG.info('Uh oh, experiment %s does not have all of the required columns.'
              'Missing: %s.', experiment_name, set_required.difference(set_df))
    _LOG.info('No file will be saved for this experiment')
    return

  sum_df = pd.DataFrame()
  for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user]

    user_avg_dict = dict()
    user_std_dict = dict()
    for emotion in emotions:
      user_avg_dict[emotion] = np.nanmean(user_df[emotion])
      user_std_dict[emotion] = np.nanstd(user_df[emotion])

    for ep in user_df['episode_id'].unique():
      ep_df = user_df[user_df['episode_id'] == ep]
      sum_dict = {'user_id': user, 'episode_id': ep}
      for col in additional_columns:
        sum_dict[col] = ep_df[col].tolist()[0]

      for emotion in emotions:
        sum_dict['average_' + emotion] = np.nanmean(ep_df[emotion])
        sum_dict['max_' + emotion] = max(ep_df[emotion])
        user_normalized_scores = ((
            ep_df[emotion] - user_avg_dict[emotion]) / user_std_dict[emotion])
        mean = np.nanmean(user_normalized_scores)
        sum_dict['user_normalized_avg_' + emotion] = mean
        sum_dict['user_normalized_max_' + emotion] = max(user_normalized_scores)

      sum_df = sum_df.append(sum_dict, ignore_index=True)

  filename = ('summarized_emotion_metrics-' + experiment_name + '-' +
              str(num_batches) + '.csv')
  store_csv(sum_df, filename)


def add_object_proto_to_df(e, df):
  """Parses an ObjectImage db entity and appends it to a pandas dataframe.

  Args:
    e: An ObjectImage datastore entity.
    df: A pandas dataframe.
  Returns:
    The pandas dataframe with additional rows corresponding to the objects
    detected in the entity's proto.
  """
  proto = process_object_response.parse_proto_from_binary_string(
      e['object_detection_proto'])
  key = e.key.id
  objects = process_object_response.get_detected_objects(proto)
  for obj in objects:
    obj_dict = {
        'id': key,
        'user_id': e['user_id'],
        'time_created': str(pd.to_datetime(e['time_created'])),
        'name': obj.name,
        'score': obj.score,
        'x1': obj.x1,
        'x2': obj.x2,
        'y1': obj.y1,
        'y2': obj.y2,
        'area': obj.area
    }
  df = df.append(obj_dict, ignore_index=True)
  return df


def get_csv_files_from_object_protos(client):
  """Retrieves ObjectImages in the DB and makes a csv after parsing the protos.

  Args:
    client: A datastore client that was previously initialized.
  """
  _LOG.info('\n\nGetting all objects from db')
  results = query_db(client, kind='ObjectImage', timestr='time_created')
  _LOG.info('Found %s results from querying the db', len(results))

  df = pd.DataFrame()
  num_processed = 0
  num_batches = 0
  while results:
    e = results.pop()
    try:
      df = add_object_proto_to_df(e, df)
      num_processed += 1
      if num_processed % 100 == 0:
        _LOG.info('Have now processed %s rows', num_processed)
      if num_processed >= FLAGS.pandas_row_limit:
        _LOG.info('Reached the max number of rows. Outputting interim df')
        filename = 'objects-' + str(num_batches) + '.csv'
        store_csv(df, filename)
        num_batches += 1
        df = pd.DataFrame()
    except:
      _LOG.info('Error! Could not process object %s %s', num_processed, e)

  filename = 'objects-' + str(num_batches) + '.csv'
  store_csv(df, filename)


def convert_entity_to_dict(e_metric, timestr='timestamp'):
  """Convert a datastore EmotionMetric entity into a dictionary.

  Args:
    e_metric: An EmotionMetric datastory entity.
    timestr: The string description of the timestamp field for this datastore
      model.

  Returns:
    A dictionary with keys as emotions and values are confidences
  """
  keys = e_metric.keys()
  e_dict = {}
  for key in keys:
    e_dict[key] = e_metric[key]
  e_dict['id'] = e_metric.key.id
  e_dict[timestr] = str(pd.to_datetime(e_metric[timestr]))
  return e_dict


def print_all_images(client):
  _LOG.info('PRINTING ALL SKETCH IMAGES')
  results = query_db(client, kind='SketchImage', timestr='time_created')
  for e in results:
    if 'times_viewed' not in e.keys():
      e['times_viewed'] = 0
      client.put(e)
    _LOG.info(e.key.id, e['time_created'], e['image_class'],
              e['model_description'], e['times_viewed'])


def summarize_all_images(client, model_description='vanilla_sketch_rnn'):
  """Queries all sketch images for a model and displays the # of each sketch.

  Args:
    client: A datastore client that has previously been created.
    model_description: The model for which to summarize the available sketches.
  """
  _LOG.info('\nSummarizing sketch images')
  results = query_db(client, kind='SketchImage', timestr='time_created')

  totals_dict = {}
  totals_dict['total'] = 0

  for e in results:
    if e['model_description'] == model_description:
      totals_dict['total'] += 1
      if e['image_class'] in totals_dict:
        totals_dict[e['image_class']] += 1
      else:
        totals_dict[e['image_class']] = 1

  _LOG.info('TOTAL SKETCHES FROM %s = %s', model_description,
            totals_dict['total'])
  for key in totals_dict:
    if key != 'total':
      _LOG.info('Total %s = %s', key, totals_dict[key])


def get_csv_files_from_images(client):
  """Retrieves all sketches currently in the DB and makes a csv file of them.

  Args:
    client: A datastore client that was previously initialized.
  """
  _LOG.info('\n\nGetting all sketches from db')
  results = query_db(client, kind='SketchImage', timestr='time_created')
  _LOG.info('Found %s results from querying the db', len(results))

  df = pd.DataFrame()
  num_processed = 0
  num_batches = 0
  while results:
    e = results.pop()
    try:
      e_dict = convert_entity_to_dict(e, timestr='time_created')
      df = df.append(e_dict, ignore_index=True)
      num_processed += 1
      if num_processed % 100 == 0:
        _LOG.info('Have now processed %s rows', num_processed)
      if num_processed >= FLAGS.pandas_row_limit:
        _LOG.info('Reached the max number of rows. Outputting interim df')
        filename = 'sketches-' + str(num_batches) + '.csv'
        store_csv(df, filename)
        num_batches += 1
        df = pd.DataFrame()
    except:
      _LOG.info('Error! Could not process sketch %s %s', num_processed, e)

  filename = 'sketches-' + str(num_batches) + '.csv'
  store_csv(df, filename)


def save_image(svg_image,
               image_class,
               model_description=None,
               z=None,
               client=None):
  """Stores an SVG image into the database.

  Args:
    svg_image: A string representing an SVG image file.
    image_class: The sketch class of the image, like 'duck'.
    model_description: String description of the model that created the sketch,
      like 'latent_constraints'.
    z: The embedding vector if the sketch was generated with a VAE.
    client: The datastore client if one has already been created.
  Returns:
    The ID of the sketch after it has been stored.
  """
  if client is None:
    client = get_datastore_client(FLAGS.creds_file)
  k = client.key('SketchImage')
  e = datastore.Entity(k, exclude_from_indexes=('svg_image',))
  e['svg_image'] = svg_image
  e['image_class'] = image_class
  e['model_description'] = model_description
  e['time_created'] = datetime.datetime.now()
  e['times_viewed'] = 0
  e['embedding'] = z
  client.put(e)
  return e.key.id


def parse_filename_svg(fname):
  """Parses an SVG filename to find information about the model and sample.

  The format for the SVG filenames is:
    image_class-batch_id-model_description-sketch_id.svg
    e.g. cat-0-discrete_decoder-0.svg

  Args:
    fname: A string name of an svg file.
  Returns:
    Four strings, the model class (like 'duck'), model description (like
      'latent_constraints', the batch ID, and the sketch or sample ID.
  """
  di1 = fname.find('-')
  model_class = fname[:di1]
  rest = fname[di1 + 1:]

  di2 = rest.find('-')
  batch_id = rest[:di2]
  rest = rest[di2 + 1:]

  di3 = rest.find('-')
  model_descrip = rest[:di3]

  di4 = rest.find('.svg')
  sample_id = rest[di3 + 1:di4]

  return model_class, batch_id, model_descrip, sample_id


def parse_filename_csv(fname):
  """Parses a csv filename to find the model description and batch ID.

  The format for the CSV filenames is:
    embeddings-model_description-batch_id.csv
    e.g. embeddings-discrete_decoder-0.csv

  Args:
    fname: A string name of a csv file.
  Returns:
    Two strings, the model description and the batch ID.
  """
  di1 = fname.find('-')
  rest = fname[di1 + 1:]

  di2 = rest.find('-')
  model_descrip = rest[:di2]

  di3 = rest.find('.csv')
  batch_id = rest[di2 + 1:di3]

  return model_descrip, batch_id


def save_all_images_sample_dir(client, sample_read_path, sample_write_path):
  """Processes model samples in a directory and saves them to the datastore.

  The samples should be stored as SVG files, but also with associated .csv files
  which store the embeddings / latent vectors used to generate each sample. The
  .csv files are linked to samples by the model description and batch number.

  The format for the SVG filenames is:
    image_class-batch_id-model_description-sketch_id.svg
    e.g. cat-0-discrete_decoder-0.svg

  The format for the CSV filenames is:
    embeddings-model_description-batch_id.csv
    e.g. embeddings-discrete_decoder-0.csv

  Args:
    client: A datastore client instance that has previously been initialized.
    sample_read_path: The location where the samples are saved.
    sample_write_path: A location to move the samples to after they have been
      written to the datastore.
  """
  #TODO: could be bugs with listdir and open
  files = os.listdir(sample_read_path)
  _LOG.info('Found the following files in directory: %s', files)

  csv_files = [f for f in files if '.csv' in f and 'error' not in f]
  _LOG.info('Found the following csv files %s', csv_files)
  while csv_files:
    csv = csv_files[0]
    model_descrip, batch_id = parse_filename_csv(csv)
    _LOG.info('Parsed csv file %s and found batch ID %s and model '
              'description %s', csv, batch_id, model_descrip)

    infile = open(sample_read_path + csv, 'r')
    df = pd.read_csv(infile)
    num_df = len(df)  # Saving this so I have it after I close the file.
    _LOG.info('Dataframe had %s rows', num_df)
    num_processed_this_batch = 0

    for f in files:
      _LOG.info('Reading file %s', f)
      if batch_id not in f or model_descrip not in f:
        continue
      (model_class, svg_batch_id, svg_model_descrip,
       sample_id) = parse_filename_svg(f)
      _LOG.info('Model class: %s, model_descrip: %s, batch: %s, sample_id: %s',
                model_class, svg_model_descrip, svg_batch_id, sample_id)
      if batch_id != svg_batch_id:
        _LOG.info('CSV batch %s not equal to sketch batch %s. Skipping.',
                  batch_id, svg_batch_id)
        continue
      if model_descrip != svg_model_descrip:
        _LOG.info('CSV model %s not equal to sketch model %s. Skipping.',
                  model_descrip, svg_model_descrip)
        continue

      svg = open(sample_read_path + f, 'r').read() #TODO could be bug

      # Find this sample in pandas dataframe.
      sample_df = df[(df['image_class'] == model_class) &
                     (df['sample_id'] == int(sample_id))]

      # The sample ID should be a unique identifier so we're sure we have the
      # right z.
      if len(sample_df) != 1:
        _LOG.info('Uh oh, expected to find a single z value in the df '
                  'matching this sample, but instead found %s. Skipping.',
                  len(sample_df))
        continue
      z_string = sample_df['z_embedding'].tolist()[0]

      key_id = save_image(svg, model_class, model_descrip, z_string, client)

      new_filename = (
          model_class + '-' + model_descrip + '-' + str(key_id) + '.svg')
      _LOG.info('Saving new file: %s', sample_write_path + new_filename)
      open(sample_write_path + new_filename, 'w').write(svg) #TODO could be bug
      Remove(sample_read_path + f) # TODO find real command
      _LOG.info('Removed old file.\n\n')

      num_processed_this_batch += 1

    infile.close()
    if num_processed_this_batch == num_df:
      Remove(sample_read_path + csv) #TODO find real command
      _LOG.info('CSV completely processed. Removing it.\n\n')
    else:
      _LOG.info('ERROR! There were %s entries in the csv but % files '
                'were read\n\n', num_df, num_processed_this_batch)
      Rename(sample_read_path + csv, sample_read_path + 'error-' + csv) #TODO find real command

    files = os.listdir(sample_read_path) #TODO could be bug
    csv_files = [f for f in files if '.csv' in f and 'error' not in f]


def main(unused_argv):
  client = get_datastore_client(FLAGS.creds_file)

  if FLAGS.store_data:
    while True:
      save_all_images_in_cns_dir(client, FLAGS.cns_read_path,
                                 FLAGS.cns_write_path)

      summarize_all_images(client, model_description='discrete_decoder')
      summarize_all_images(client, model_description='discrete_decoder_256')
      summarize_all_images(client, model_description='discrete_data')
      summarize_all_images(client, model_description='latent_constraints')
      summarize_all_images(client, model_description='latent_constraints_prior')

      _LOG.info('Sleeeeeping...')
      time.sleep(60*5)  # Sleep for 5 minutes.
  else:
    get_csv_files_from_images(client)
    get_csv_files_from_emotion_metrics(client)
    get_csv_files_from_object_protos(client)


if __name__ == '__main__':
  app.run(main)
