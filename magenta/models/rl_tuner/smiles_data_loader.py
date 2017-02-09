import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import json

class SmilesLoader():
    def __init__(self, input_file, vocab_file, numpy_file, batch_size, max_seq_length=120):
        self.input_file = input_file
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.numpy_file = numpy_file
        self.max_seq_length = max_seq_length

        self.create_char_conversions()

        if os.path.exists(self.numpy_file):
            self.load_preprocessed()
        else:
            self.preprocess()

    def create_char_conversions(self):
        self.char_list = json.load(open(self.vocab_file))
        self.vocab_size = len(self.char_list)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))

    def load_preprocessed(self):
        self.batch_array = np.load(self.numpy_file)
        self.num_batches = len(self.batch_array)

    def preprocess(self):
        f = open(self.input_file, 'r')
        lines = f.readlines()
        lines = sorted(lines, key=len) # sort sequences by length for efficient processing
        num_seqs = len(lines)

        batches = []
        i = 0
        while(i < num_seqs):
            smiles = lines[i:i+self.batch_size]
            lens = [len(x) for x in smiles]
            max_len = max(lens)

            i += self.batch_size
        
        np.save(self.numpy_file, self.batch_array)

    def smiles_batch_to_one_hot(self, smiles_list, max_len):
        smiles_list = [self.clean_and_pad_smile(x) for x in smiles_list if self.check_smile_len(x)]
        
        Z = np.zeros((len(smiles_list),
                      max_len, self.vocab_size),
                      dtype=np.bool)
        for i, smile in enumerate(smiles_list):
            for t, char in enumerate(smile):
                Z[i, t, self.char_to_index[char]] = 1
        return Z

    def check_smile_len(self, string):
        if len(string) <= self.max_seq_len:
            return True
        else:
            return False

    def clean_and_pad_smile(self, string):
        string = string.rstrip('\n')
        return string + " " * (self.max_seq_len - len(string))

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

