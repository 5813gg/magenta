import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import json
import pickle
import sys

class SmilesLoader():
    def __init__(self, vocab_file, input_file=None, pickle_file='smiles.p', batch_size=128, max_seq_len=120):
        self.input_file = input_file
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.pickle_file = pickle_file
        self.max_seq_len = max_seq_len

        self.create_char_conversions()

        self.batches = {}
        if os.path.exists(pickle_file):
            self.load_preprocessed()
        else:
            if self.input_file is None:
                print "Error! Pre-processed pickle file does not exist and input file not provided"
                sys.exit()
            self.preprocess()

    def create_char_conversions(self):
        print "Loading vocabulary from file", self.vocab_file
        if not os.path.exists(self.vocab_file):
            print "ERROR! Vocab file", self.vocab_file, "does not exist!"
        self.char_list = json.load(open(self.vocab_file))
        self.vocab_size = len(self.char_list)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))

    def load_preprocessed(self):
        self.batches = pickle.load(open(self.pickle_file,"rb"))

    def preprocess(self):
        f = open(self.input_file, 'r')
        lines = f.readlines()
        lines = sorted(lines, key=len) # sort sequences by length for efficient processing
        num_seqs = len(lines)

        for dataset in ['train','val','test']:
            self.batches[dataset] = []

        i = 0
        while(i < num_seqs):
            smiles = lines[i:i+self.batch_size]
            smiles = [self.clean_smile(x) for x in smiles]
            lens = [len(x) +1 for x in smiles] # adding 1 extra space on all sequences to indicate EOS
            max_len = max(lens)

            X, Y = self.smiles_batch_to_matrices(smiles, max_len)
            batch_dict = self.make_batch_dict(X,Y,lens)
            dataset = self.roll_for_dataset()
            self.batches[dataset].append(batch_dict)

            i += self.batch_size
        
        pickle.dump(self.batches, open(self.pickle_file,"wb"))

    def roll_for_dataset(self):
        r = np.random.rand()
        if r <= 0.6:
            return 'train'
        elif r <= 0.8:
            return 'val'
        else:
            return 'test'

    def make_batch_dict(self, X, Y, lens):
        bdict = {}
        bdict['X'] = X
        bdict['Y'] = Y
        bdict['lengths'] = lens
        return bdict

    def get_next_step_smile(self, smile):
        ydata = [''] * len(smile)
        ydata[:-1] = smile[1:]
        ydata[-1] = ' '
        return ydata

    def smiles_batch_to_matrices(self, smiles_list, max_len):    
        y_smiles = [self.get_next_step_smile(x) for x in smiles_list]
                
        X = np.zeros((len(smiles_list),
                      max_len, self.vocab_size),
                      dtype=np.bool)
        for i, smile in enumerate(smiles_list):
            for t, char in enumerate(smile):
                X[i, t, self.char_to_index[char]] = 1

        Y = np.zeros((len(smiles_list), max_len), dtype=np.int32)
        for i, smile in enumerate(y_smiles):
            for t, char in enumerate(smile):
                Y[i, t] = self.char_to_index[char]
        return X, Y

    def clean_smile(self, string):
        return string.rstrip('\n')

    def next_batch(self, dataset='train'):
        num_batches = len(self.batches[dataset])
        i = np.random.randint(0,num_batches)

        X = self.batches[dataset][i]['X']
        Y = self.batches[dataset][i]['Y']
        lens = self.batches[dataset][i]['lengths']
        while len(X) != self.batch_size:
            X,Y,lens = self.next_batch(dataset=dataset)

        return X, Y, lens

