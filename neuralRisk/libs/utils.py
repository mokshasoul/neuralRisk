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
from datetime import date
import theano
import pandas as pd
import config
import theano.tensor as T
import csv
# from theano import pydotprint as pdp
import sys

__author__ = 'c.n.georgiou'


def shared_dataset(data_xy, borrow=True):
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


def load_data(dataset, delimiter=',', borrow=True):
    dataset = os.path.abspath(dataset)
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in data directory
        dirname = os.path.dirname
        new_path = os.path.join(
                dirname(dirname(__file__)),
                "testData",
                dataset
                )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    print('Loading data into model')
    if(data_file == 'mnist.pkl.gz'):
        print("USING MNIST PICKLE")
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f,
                                                             encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
    else:
        if (".csv" in data_file):
                print("CREATING CSV FILE NAME SETS")
                dataset_name = data_file.split('.')[0]
                train_set = os.path.join(data_dir, dataset_name + "_train.csv")
                valid_set = os.path.join(data_dir, dataset_name + "_valid.csv")
                test_set = os.path.join(data_dir,  dataset_name + "_test.csv")
                if(file_exists(train_set) and file_exists(train_set) and
                   file_exists(test_set)):
                    rval = load_data_csv(train_set, valid_set, test_set,
                                         delimiter)
                else:
                    print('It appears you have a valid dataset but forgot to \
                          split it or one of them is missing, \
                          the programm will exit NOW!')
                    sys.exit(2)
        else:
            print('Data file either does not have a valid format please either \
                   run demo or use a CSV as input\n')
            print('The program will exit NOW!')
            sys.exit(2)

    return rval


def load_data_csv(train_set, valid_set, test_set, delimiter):

    train_xy = load_csv(train_set, delimiter)
    valid_xy = load_csv(valid_set, delimiter)
    test_xy = load_csv(test_set, delimiter)

    test_set_x, test_set_y = shared_dataset(test_xy)
    valid_set_x, valid_set_y = shared_dataset(valid_xy)
    train_set_x, train_set_y = shared_dataset(train_xy)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


def load_data_prediction(dataset, delimiter=',', borrow=True):
    print('Loading prediction dataset into shared memory')
    prediction_xy = load_csv(dataset, delimiter, isprediction=True)
    prediction_set_x, prediction_set_y = shared_dataset(prediction_xy)
    rval = [(prediction_set_x, prediction_set_y)]
    return rval


def load_csv(path, delimiter, isprediction=False):
    try:
        f_csv_in = open(path)
    except:
        print('File given in' + path + ' does not exist')
        sys.exit(2)

    print('File given in ' + path + ' successfully loaded')
    csv_data = pd.read_csv(f_csv_in, delimiter=delimiter)
    y_label = csv_data.axes[1][-1:].values[0]
    j = 0
    for i in csv_data.axes[1]:
        if y_label in i:
            j = j-1

    if isprediction:
        write_headers(csv_data.axes[1])
    # csv_data = pd.get_dummies(csv_data)
    # plot_input(csv_data, path)
    data = csv_data.values
    if(j == -1):
        data = (data[:, 0:-1], data[:, -1:].flatten())
    else:
        data = (data[:, 0:j], data[:, j:])
    return data


def write_headers(headers):
    with open(config.prediction_file, 'ab') as csvfile:
        print('Writing headers to prediction file')
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow('\n')
        writer.writerow(headers)


def append_prediction(X, y):
    with open(config.prediction_file, 'ab') as csvfile:
        print('Writing prediction data to prediction file')
        writer = csv.writer(csvfile, delimiter=';')
        i = 0
        for pred in y:
            writer.writerow([X[i, :].tolist(), pred])
            i = i + 1


def file_exists(data_file):
    return os.path.isfile(data_file)


def is_valid_file(parser, arg):
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exists!" % arg)
    else:
        return arg


def data_file_name(dataset):
    data_file = os.path.split(dataset)[1]
    return data_file.split('.')[0]


def find_two_closest_factors(n):
    deltas = []
    factors = {}
    if n <= 1:
        return (1, 1)
    for i in range(1, n):
        if n % i == 0:
            delta = abs(i - n/i)
            deltas.append(delta)
            factors[delta] = (i, n/i)
    return factors[min(deltas)]


def write_results(task_num, dataset_name,
                  learning_rate, epochs, batch_size,
                  n_in, n_out, n_hidden,
                  logfile, best_validation_loss, test_score,
                  model_name, runtime):
    with open(config.log_file, 'ab') as csvfile:
        print('Writing trainings results to logfile')
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([task_num, date.today(), dataset_name,
                         learning_rate,
                         epochs,
                         batch_size,
                         n_in,
                         n_out,
                         n_hidden,
                         best_validation_loss,
                         test_score, model_name,
                         runtime])
        writer.writerow('\n')


def load_svm_dataset(dataset):
    data_dir, data_file = os.path.split(dataset)
    dataset_name = data_file.split('.')[0]
    train_set = os.path.join(data_dir, dataset_name + "_train.csv")
    test_set = os.path.join(data_dir,  dataset_name + "_test.csv")
    train_X, train_y = load_csv(train_set)
    test_X, test_y = load_csv(test_set)
    rval = [(train_X, train_y, test_X, test_y)]
    return rval

# def print_theano_graph(function, out_file):
#     pdp(function, outfile=out_file)
