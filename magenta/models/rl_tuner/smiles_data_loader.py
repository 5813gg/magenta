import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import json

class SmilesLoader():
    def __init__(self, input_file, vocab_file, batch_size, numpy_file=None, max_seq_length=120):
        self.input_file = input_file
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.numpy_file = numpy_file
        self.max_seq_length = max_seq_length

        self.create_char_conversions()
        
        if self.numpy_file is not None:
            self.load_preprocessed
        else:
            self.preprocess()

    def create_char_conversions(self):
        self.char_list = json.load(open(vocab_file))
        self.n_chars = len(self.char_list)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))

    def load_preprocessed(self):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def preprocess(self):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)




    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

