from __future__ import print_function
import numpy as np
np.random.seed(0)

"""
    Keras Imports
"""
from keras.models import Sequential # core model ( Always use sequential?)
from keras.layers import Dense, Dropout, Activation, Flatten # core layers
from keras.utils import np_utils
from keras.regularizers import WeightRegularizer, l2  # L2-Reg
from keras.layers.advanced_activations import SReLU

class keroRisk(object):
    
    @classmethod
    def __init__(self, learning_rate, L1_reg, L2_reg, n_epochs,
                datasets, n_in, n_out, batch_size, n_hidden, logfile,
                activation, task_num, dataset_name):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        print(datasets[0])
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, valid_set_y = datasets[2]
        input_size = train_set_x.get_value(borrow=True).shape[1]
        model = Sequential()
        # add one hidden layer
        model.add(Dense(n_hidden,input_dim=input_size))
        model.add(SReLU())
        model.add(Dense(n_hidden))
        model.add(Dense(n_out))

        model.compile(loss='categorical_crossentropy', 
                      optimizer='adadelta',
                      metrics=['accuracy'])

        X_train = train_set_x.get_value(borrow=True)
        Y_train = train_set_y.eval()
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epochs,
                   verbose=2, validation_split=0.2)


    def plot(self):
        pass

