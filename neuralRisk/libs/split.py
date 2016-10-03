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
    main usage is included in run.py
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
from sklearn.feature_extraction import DictVectorizer
import utils
import pandas as pd
import numpy as np


def main(input_file, delimiter=";", split_percent=80,
         normalization=True, ordinal=False):
    input_file = os.path.abspath(input_file)
    print(input_file)
    try:
        if ".csv" in input_file:
            transform_set = pd.read_csv(input_file, sep=delimiter)
        else:
            transform_set = pd.read_excel(input_file)
    except:
        print("Hey Jim something happened while reading the file")
        sys.exit(2)

    # Here the algorithms performs the standarization
    # and shuffling of the data
    # y_label_set = transform_set.ix[:, -1:] <-- accessing pandas
    # Dataframes through ix
    proper_shit(transform_set)
    y_label_name = transform_set.axes[1][-1:].values[0] + '_'
    transform_set = pd.get_dummies(transform_set)
    train_shape = transform_set.shape
    new_index = np.random.permutation(transform_set.index)
    train_set = transform_set.reindex(new_index)
    # Here we perform normalization on all columns that do not contain
    # categorical data
    # print(train_set.ix[:, -2:]) <-- Sample to extract Columns
    """
        This hack removes the Label column since this program does
        primarily classification and avoids normalizing it instead
        only normalizing the X values.
    """
    if normalization:
        i = 0
        for j in train_set.axes[1].values:
            if y_label_name in j:
                i = i - 1

        train_set_x = train_set.ix[:, :i]
        train_set_y = train_set.ix[:, i:]
    # if ordinal:
    #     for j in xrange(i):
    #         for k in xrange(train_set_y.shape[0]):

        train_set_norm = ((train_set_x -
                           train_set_x.mean()) /
                          (train_set_x.max()-train_set_x.min()))
        # Merge the set again
        train_set_norm = train_set_norm.join(train_set_y)
        train_set = train_set_norm

    # Here we do the splitting
    indice_percent = int((train_shape[0]/100.0)*split_percent)
    indice_percent_valid = int((indice_percent/100.0) * split_percent)
    filepath, filename = os.path.split(input_file)
    filename = utils.data_file_name(input_file)
    # generate the output sets
    output_train = filename + "_train.csv"
    output_valid = filename + "_valid.csv"
    output_test = filename + "_test.csv"

    train_set[:indice_percent][:indice_percent_valid].to_csv(
        os.path.join(filepath, output_train), header=True, index=False)
    train_set[:indice_percent:][indice_percent_valid:].to_csv(
        os.path.join(filepath, output_valid), header=True, index=False)
    train_set[indice_percent:].to_csv(
        os.path.join(filepath, output_test), header=True, index=False)
    print("Files outputed as:" + output_train + " " + output_valid + " "
          + output_test + "in folder" + filepath)


def charis_normalizer(transform_set):
    """
        Creates integer list, which is a column wise identifier of columsn
        which only contain integers and thus are excempt from standarization
        per se
    """
    y_label_name = transform_set.axes[1][-1:].values
    integer_list = []
    x_first_row = transform_set.iloc[0].values.tolist()
    for i,val in enumerate(x_first_row):
        print('We have : ' + str(val))
        if str(val).isdigit():
            integer_list.append(i)
        else:
            pass
    create_normalized_df(transform_set, integer_list)


def create_normalized_df(transform_set, integer_list):
     dv = DictVectorizer(sparse=False) 
     df = pd.to_numeric(transform_set)
     dv.fit_transform(df.to_dict(orient='records'))

     print(df)


def proper_shit(transform_set):
    transform_set_x = pd.get_dummies(transform_set.ix[:,:-1])
    transform_set_y = transform_set.ix[:,-1:]
    category_list = []
    for i in transform_set_y.iterrows():
        category = str(i[1].values[0])
        if not(category in category_list):
            category_list.append(category)

def categorical_to_numerical_y(transform_set_y, category_list):
    pass        

