#!/usr/bin/python
# coding=utf8
"""
This is a class to plot the graphs of the neuralRisk Application
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


Code inspiration from here:
https://github.com/crmne/Genretron-Theano/blob/master/plot.py
and
https://github.com/twuilliam/ift6266h14_wt/blob/master/post_01/mlp.py

"""

import os
import sys
import utils
import numpy as np
import matplotlib as mpl
from datetime import date


class Plot(object):
    def __init__(self, *what_to_plot):
        """
            init function for our Plot class

        :*what_to_plot:  Strings of what to
        plot
        """
        for x in what_to_plot:
            assert isinstance(x, str)
        self.data = {key: [] for key in what_to_plot}
        self.length = {key: [] for key in what_to_plot}
        # self.fig = mpl .pyplot.figure()
        # mpl.pyplot.ion()
        # mpl.pyplot.show()

    def append(self, key, value, epoch):
        if key not in self.data:
            raise StandardError("Plot: Key %s not found." % key)
        self.data[key].append(value)
        self.length[key].append(epoch)


    def update_plot(self):
        dim_x, dim_y = utils.find_two_closest_factors(len(self.data))
        legend = [] 

        for k, v in self.data.iteritems():
            if len(v) != 0:
                y=np.array(v)
                mpl.pyplot.plot(self.length[k], y, label=k)

        mpl.pyplot.legend()
        mpl.pyplot.ylabel('MSE')
        mpl.pyplot.xlabel('epoch')


    def save_plot(self, epoch=0, task_n=1, file_format='PDF'):
        output_folder = os.path.split(__file__)[0]
        output_file = \
        os.path.join(output_folder,'Plot_{:d}_{:%Y_%m_%d_%H_%M}.{}'.format(task_n,date.today(),
                                                                          file_format))
        self.update_plot()
        print("saving_plot")
        mpl.pyplot.savefig(output_file, format=file_format)
