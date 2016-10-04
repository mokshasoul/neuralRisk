#!/usr/bin/python
'''
    @author Charis - Nicolas Georgiou
    inspiration is: https://github.com/tajo/deeplearning/blob/master/run.py
    main instance to run Neural Risk Network
'''
import csv
import os
import json
import config
from datetime import date
import sys
from neuralRisk.libs import svm
from neuralRisk.libs import utils
from neuralRisk.libs.split import main as split
from neuralRisk.libs.riskNN import riskNN
from neuralRisk.libs.riskNN import predict

'''
    run tasks given in json format
'''


def run(tasks):
    task_num = 1
    for task_params in tasks:
        print('####### PROCESSING TASK NUMBER {} ######'.format(task_num))
        run_task(task_params, task_num)
        task_num += 1

'''
    prepare arguments to run in MLP
'''


def run_task(params, task_num):
    if(params['setting'] == 'train'):
        log_file = os.path.join(os.getcwd(), 'logs', params['logfile'])
        if config.log_file is None:
            config.log_file = log_file
        if config.plot_dir is None:
            config.plot_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.isfile(config.log_file) or \
           not os.stat(config.log_file)[6] == 0:
            with open(config.log_file, "a") as logfile:
                data = ['task_number',
                        'date_run',
                        'dataset',
                        'learning_rate',
                        'epochs',
                        'batch_size',
                        'n_ins',
                        'n_outs',
                        'n_hidden',
                        'valid_perf (%)',
                        'test_perf (%)',
                        'model_name',
                        'run_time (min)']
                writer = csv.writer(logfile,
                                    delimiter=str(params['delimiter']))
                writer.writerow(data)
        for dataset in params['datasets']:
            riskNN(params['learning_rate'],
                   params['L1_reg'],
                   params['L2_reg'],
                   params['training_epochs'],
                   dataset,
                   params['n_outs'],
                   params['batch_size'],
                   params['hidden_layers_sizes'],
                   params['logfile'],
                   params['activation'],
                   task_num)

    elif(params['setting'] == 'predict'):
        for dataset in params['datasets']:
            config.prediction_file = os.path.join(os.getcwd(), 'predictions',
                                                  utils.data_file_name(dataset)
                                                  + '_' +
                                                  str(task_num)+'_' +
                                                  params['activation']+'_' +
                                                  str(date.today())+'.csv'
                                                  )

            if not(os.path.isfile(config.prediction_file)):
                open(config.prediction_file, 'a').close()
            predict(dataset,
                    params['best_model'],
                    params['batch_size'],
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
    parser.add_argument("-z", "--svm",
                        dest="svm_dataset",
                        type=lambda x: utils.is_valid_file(parser, x),
                        default=None,
                        help="The dataset to be processed by the SVM")
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
    if args.svm_dataset is not None:
            svm.svm_exp(args.svm_dataset)
