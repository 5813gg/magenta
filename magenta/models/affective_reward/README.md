# Affective Reward Models

Code to support models that learn from Affective Reward.

## Reading and writing affective data from/to the datastore

In order to train the models with affective reward, emotional expression data
must first be read from the Google App Engine (GAE) datastore. The file
datastore_interface.py allows for this to happen. It depends on the
gae_service_account_creds.json, which allows it to log into the GAE datastore
and read the data there. 

Other files (ending in store_samples) are necessary to write samples from the
models to the datastore.

## Models

### Discrete Decoder

The discrete decoder is an LSTM trained on sequences of absolute x and y pixel
values, as well as pen up/down events (from the Quickdraw data).
