"""
This represents the utility functions of the neuralRisk project for TUM
Copyright  2016 Charis - Nicolas Georgiou

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import gzip
import cPickle as pickle
import numpy as np
import theano
from theano import function
import theano.tensor as T
import csv
import sys

__author__ = 'c.n.georgiou'


def load_data(dataset, train, valid, test, targetCol, targets=4, borrow=True):

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in data directory
        new_path = os.path.join(
                os.path.split(__file__)[0],
                "data",
                dataset
                )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    print('loading data into model')
    if(data_file == 'mnist.pkl.gz'):
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f,
                        encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
    else:
        if '.csv' in data_file:
            data =  np.loadtxt(data_file, delimiter=',')
           #train_set, valid_set, test_set = 
        else:
            print('Data file does not have a valid format please either \
                   run demo or use a CSV as input\n')
            print('The program will exit NOW!')
            sys.exit()

    def shared_dataset(data_xy):
        """ Function that loads the datasets into shared variables
            Take from tutorial in here:
            http://deeplearning.net/tutorial/gettingstarted.html#gettingstarted
            """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # GPU shared data has to be float since GPU is good at floats ;)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
