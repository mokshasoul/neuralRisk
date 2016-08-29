#!/usr/bin/python
'''
    @author Charis - Nicolas Georgiou
    inspiration is: https://github.com/tajo/deeplearning/blob/master/run.py
    main instance to run Neural Risk Network
'''
import csv
import os
import json
import sys
from libs import utils
from libs.split import main as split
from libs import riskNN

'''
    run tasks given in json format
'''


def run(tasks):
    task_num = 1
    for task_params in tasks:
        print('####### PROCESSING TASK NUMBER {} ######'.format(task_num))
        run_task(task_params)
        task_num += 1

'''
    prepare arguments to run in MLP
'''


def run_task(params):
    if(params['setting'] == 'train'):
        if not os.path.isfile(params['logfile']) or \
           not os.stat(params['logfile'])[6] == 0:
            with open(params['logfile'], "wb") as logfile:
                data = ['date_time',
                        'dataset',
                        'target_name',
                        'finetune_lr',
                        'pretraining_epochs',
                        'pretrain_lr',
                        'batch_size',
                        'n_ins',
                        'n_outs',
                        'hidden_layers_sizes',
                        'valid_perf (%)',
                        'test_perf (%)',
                        'test_recall',
                        'run_time (min)']
            writer = csv.writer(logfile, delimiter=params['delimiter'])
            writer.writerow(data)
        for dataset in params['datasets']:
            riskNN.create_NN(params['learning_rate'],
                             params['L1_reg'],
                             params['L2_reg'],
                             params['training_epochs'],
                             dataset,
                             params['n_ins'],
                             params['n_outs'],
                             params['batch_size'],
                             params['hidden_layers_sizes'],
                             params['logfile'],
                             params['activation'])

    elif(params['setting'] == 'predict'):
        for dataset in params['datasets']:
            riskNN.predict(dataset,
                           params['best_model'],
                           params['batch_size'],
                           params['n_ins'],
                           params['hidden_layers_sizes'],
                           params['n_outs'],
                           params['activation'])


'''
    Parser
'''


def get_parser():
    """ Get the parser fo any script """
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--demo",
                        action='store_true',
                        dest="runDemo",
                        default=False,
                        help="runs mnist demo")
    parser.add_argument("-pd", '--predict-demo',
                        action='store_true',
                        dest="predictDemo",
                        default=False,
                        help="runs a demo prediction")
    parser.add_argument("-j", "--json",
                        dest="json_data",
                        type=lambda x: utils.is_valid_file(parser, x),
                        default=None,
                        help="input json when not running demo")
    parser.add_argument("-s", "--split",
                        dest='filetosplit',
                        type=lambda x: utils.is_valid_file(parser, x),
                        default=None,
                        help="input CSV file to split")
    parser.add_argument("-o", "--output",
                        dest="j")
    return parser

'''
    Program invocations
'''

if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.runDemo:
        riskNN.create_NN()
    if args.predictDemo:
        riskNN.predict()
    if args.json_data is not None:
        try:
            json_data = open(args.json_data)
        except:
            print('You have to specify a valid JSON file to start computing')
            sys.exit()
        data = json.load(json_data)
        run(data['tasks'])
    if args.filetosplit is not None:
        try:
            split(args.filetosplit)
        except:
            print('Oops something died Jim')
            sys.exit()
