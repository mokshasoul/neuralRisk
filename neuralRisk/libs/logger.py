"""
    Based on:
        https://github.com/twuilliam/ift6266h14_wt/blob/master/post_final/multiple_mlps/deep_mlp_gen.py
"""
import sys


class logs(object):
    """ log to file and output """
    def __init__(self, logfile, mode):
        self.file = open(logfile, mode)
        self.stdout = sys.stdout

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write_log(self, data):
        self.file.write(data + '\n')
        self.stdout.write(data + '\n')
