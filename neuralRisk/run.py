'''
    @author Charis - Nicolas Georgiou
    inspiration is: https://github.com/tajo/deeplearning/blob/master/run.py
    main instance to run Neural Risk Network
'''
import csv
import os
import json
import sys
from libs import riskNN

'''
    run tasks given in json format
'''


def run(tasks):
    task_num = 1
    for task_params in tasks:
        print '####### PROCESSING TASK NUMBER {} ######'.format(task_num)
        run_task(task_params)
        task_num += 1

'''
    prepare arguments to run in MLP
'''


def run_task(params):
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
    # number of target columns in dataset to they can be split
    try:
        targets = params['targets']
    except:
        targets = 3  # default for stock datasets

    for dataset in params['datasets']:
        riskNN.create_NN(params['finetune_lr'],
                         params['pretraining_epochs'],
                         params['pretrain_lr'],
                         params['training_epochs'],
                         dataset,
                         params['batch_size'],
                         params['n_ins'],
                         params['hidden_layers_sizes'],
                         params['n_outs'],
                         params['logfile']
                         )

if __name__ == '__main__':
    try:
        # first argument is a json config file as parameter
        json_data = open(sys.argv[1])
    except:
        print 'You have to specify a valid JSON file for the computing \
               tasks to start'
        sys.exit()
    data = json.load(json_data)
    run(data['tasks'])
