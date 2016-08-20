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

"""

import os
import utils
import matplotlib as mpl


class Plot(object):
    def __init__(self, *what_to_plot):
        """init function for our Plot class

        :*what_to_plot:  Strings of what to
        plot
        """
        for x in what_to_plot:
            assert isinstance(x, str)
        self.data = {key: [] for key in what_to_plot}
        self.length = 0
        self.fig = mpl .pyplot.figure()
        mpl.pyplot.ion()
        mpl.pyplot.show()

    def append(self, key, value):
        if key not in self.data:
            raise StandardError("Plot: Key %s not found." % key)
        self.data[key].append(value)
        self.length = max([len(v) for k, v in self.data.iteritems()])

    def update_plot(self):
        x = self.length
        dim_x, dim_y = utils.find_two_closest_factors(len(self.data))

        i = 1
        for k, v in self.data.iteritems():
            if len(v) != 0:
                mpl.pyplot.subplot(dim_x, dim_y, i)
                mpl.pyplot.plot(x, v[-1], 'r.-')
                mpl.pyplot.title(k)
            i = i + 1

    def save_plot(self, format='PDF'):
