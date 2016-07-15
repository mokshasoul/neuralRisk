import os
import sys
import re
import argparse
import neuralRisk


def init():
    """TODO: Docstring for __process_cl_args.
    :returns: Function to be use

    """
    parser = argparse.ArgumentParser(description='Loader for Risk Prediction')
    parser.add_argument('commands', nargs='*')
    parser.add_argument('--help', '-h', action='store_true')
    parser.add_argument('--version', '-v', action='store_true')
    parser.add_argument('--trainingset', dest='load_training_set',
            help='trainingset to use')
    parser.add_argument('--testingset ', dest='load_testing_set')
    parser.add_argument('--playdemo', dest='load_demo_version')
    parser.add_argument('--predictionset', dest='load_prediction_set')
    args = parser.parse_args()
