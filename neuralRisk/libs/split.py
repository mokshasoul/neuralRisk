#!/usr/bin/python
# coding=utf-8
# vim: set fileencoding=utf-8
"""
split.py:
    This program is used to split CSV or XLSX data into multiple files for
    neural network purposes. Pieces of this code where taken from:
    https://stackoverflow.com/questions/31112689/shuffle-and-split-a-data-file-into-training-and-test-set
    Usage is split.py <input file> the output will be 3 files
    input_file_name_train,_valid,_test.{csv || xlsx}
Copyright © 2016 Charis - Nicolas Georgiou

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
import sys
import os
import pandas as pd
import numpy as np

# input_file = sys.argv[1]
# delimiter = sys.argv[2]
# filepath, filename = os.path.split(input_file)
# filename = filename[:-4]


def main(input_file, delimiter=";", split_percent=80,
         cross_validation=False):
    input_file = os.path.abspath(input_file)
    if ".csv" in input_file:
        transform_set = pd.read_csv(input_file, sep=delimiter)
    else:
        transform_set = pd.read_excel(input_file)
    print(transform_set)

    train_shape = transform_set.shape

    df = pd.DataFrame(transform_set)

    train_set = df.reindex(np.random.permutation(df.index))

    indice_percent = int((train_shape[0]/100.0)*split_percent)
    indice_percent_valid = int((indice_percent/100.0) * split_percent)

    print(indice_percent)
    filepath, filename = os.path.split(input_file)
    filename = filename[:-4]
    output_train = filename + "_train.csv"
    output_valid = filename + "_valid.csv"
    output_test = filename + "_test.csv"

    train_set[:indice_percent][:indice_percent_valid].to_csv(
        os.path.join(filepath, output_train), header=True, index=False)
    train_set[:indice_percent:][indice_percent_valid:].to_csv(
        os.path.join(filepath, output_valid), header=True, index=False)
    train_set[indice_percent:].to_csv(
        os.path.join(filepath, output_test), header=True, index=False)
    print(output_train + " " + output_valid + " " + output_test)

# main(input_file, delimiter)
