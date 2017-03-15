"""
This is the main class for our neuralRisk application

Copyright  2016 Charis - Nicolas Georgiou

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


This was part of the tutorial that introduces MLP in the site
www.deeplearning.net
"""

from __future__ import print_function
import numpy as np
import theano.tensor as T
import theano
from riskMLP import riskMLP
import sys
import os
import timeit
from datetime import date
# from plot import Plot
from matplotlib import pyplot as plt
from ohnn import keroRisk
import utils


__docformat__ = 'restructedtext en'


class riskNN:
    def __init__(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
                 n_epochs=1000,
                 dataset='mnist.pkl.gz', n_out=10, batch_size=20,
                 n_hidden=500, logfile='test.csv', activation='tanh',
                 task_num=1,
                 load_params=False):

        datasets = utils.load_data(dataset)
        dataset_name = utils.data_file_name(dataset)

        train_set_x, train_set_y = datasets[0]
        try:
            if(train_set_y.eval().shape[1] > 1):
                print('Using Keras Classifier')
                keroRisk(learning_rate, L1_reg, L2_reg, n_epochs,
                         datasets, n_out, batch_size,
                         n_hidden, logfile, activation,
                         task_num,
                         dataset_name
                         )
            else:
                print('Using Theano Logistic Regression Classifier')
                create_NN(learning_rate, L1_reg, L2_reg, n_epochs,
                          dataset, n_out, batch_size,
                          n_hidden, logfile, activation, task_num,
                          load_params)
        except IndexError:
            print('Using Theano Logistic Regression Classifier, \
                   since program encountered an index exception')
            create_NN(learning_rate, L1_reg, L2_reg, n_epochs,
                      dataset, n_out, batch_size,
                      n_hidden, logfile, activation, task_num,
                      load_params)


def create_NN(learning_rate, L1_reg, L2_reg, n_epochs,
              dataset, n_out=10, batch_size=20,
              n_hidden=500, logfile='test.csv', activation='tanh', task_num=1,
              load_params=False):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    TODO: Add a load_params value in order to load a model
    """
    datasets = utils.load_data(dataset)
    dataset_name = utils.data_file_name(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # computation of minibatches for training, valid and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('...building the model')
    # Debugging for theano
    # theano.config.compute_test_value = 'warn'
    # theano.config.optimizer='fast_compile'
    # allocate symbolic vars for the data
    index = T.lscalar()
    x = T.matrix('x')

    y = T.ivector('y')
    n_in = train_set_x.eval().shape[1]
    rng = np.random.RandomState(1234)

    # construct MLP class
    classifier = riskMLP(
            rng=rng,
            input=x,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            activation_function=activation
            )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # symbolically (THEANO)
    negative = classifier.negative_log_likelihood(y)
    cost = (
            negative +
            L1_reg * classifier.L1 +
            L2_reg * classifier.L2_sqr
            )

    # compile a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
                }
            )

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                }
            )
    # compute the gradient of cost with respect to theta (sorted in params)
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
            ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # plot = Plot(dataset_name+"_vali_loss", 'Validation-Loss', 'Test-Loss')
    # cost_plot = Plot(dataset_name+"avg_cost", 'AVG-Batch-Cost')
    # TODO: reimplement patience
    # early-stopping parameters ( in order to stop on mistakes)
    # patience = 10000        # look as this many samples regardless
    # patience_increase = 5   # wait this much longer when a new best is found
    # a relative improvement of this much
    # is considered significant
    # improvement_threshold = 0.995
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch
    # validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    avg_cost = []
    avg_val = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        minibatch_avg_cost = [train_model(i) for i in range(n_train_batches)]
        avg_cost.append(np.mean(minibatch_avg_cost))
        # cost_plot.append('AVG-Batch-Cost', minibatch_avg_cost, epoch)
        # iteration number
        validation_losses = [validate_model(i) for i
                             in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        avg_val.append(this_validation_loss)
        print(
                'epoch %i, training error %f %%, validation error %f %%' %
                (
                    epoch,
                    np.mean(minibatch_avg_cost) * 100,
                    this_validation_loss * 100.
                    )
                )
        # plot.append('Validation-Loss', this_validation_loss, epoch)
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss

            # test it on the test set
            test_losses = [test_model(i) for i
                           in range(n_test_batches)]
            test_score = np.mean(test_losses)

            print(('epoch %i, test error of '
                   'best model %f %%') %
                  (epoch,
                   test_score * 100.))
            # plot.append('Test-Loss', test_score, epoch)

            model_name = ('best_model_'+str(task_num)
                                       + '_'
                                       + dataset_name
                                       + '_'
                                       + activation
                                       + '_'
                          + str(date.today())+'.pkl')
            classifier.save_model(filename=model_name)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)),
          file=sys.stderr)
    print('model saved under name: {}'.format(model_name))
    utils.write_results(task_num, dataset_name,
                        learning_rate, n_epochs, batch_size,
                        n_in, n_out, n_hidden,
                        logfile, best_validation_loss, test_score,
                        model_name, ' ran for %.2fm' % ((end_time - start_time)
                                                        / 60.))
    # plot.save_plot(task_n=task_num)
    # cost_plot.save_plot(task_n=task_num)
    plt.figure()
    plt.plot(range(epoch), avg_val)
    plt.plot(range(epoch), avg_cost)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(['valid', 'train'])
    plt.title('Error ' + activation)
    plt.show()


def predict(dataset='mnist.pkl.gz',
            best_model='best_model_mnist.pk_2016-08-23.pkl', batch_size=20,
            n_hidden=50, n_out=10, activation_function='tanh'):
    """
    An example of how to load a trained model and use it
    to predict labels. Modified in order to be able to pickle
    model taken from :
    https://stackoverflow.com/questions/34068922/save-theano-model-doenst-work-for-mlp-net
    Which didn't work so we implemented a save and load function in the
    classifier itself
    TODO: Make save and load be more dynamical e.g. take into account multiple
    hidden layers
    :type n_hidden: hidden units
    """
    # We can test it on some examples from test test
    if (dataset is not 'mnist.pkl.gz'):
        datasets = utils.load_data_prediction(dataset)
        test_set_x, test_set_y = datasets[0]
    else:
        datasets = utils.load_data(dataset)
        test_set_x, test_set_y = datasets[2]
    n_in = test_set_x.eval().shape[1]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    rng = np.random.RandomState(1234)
    x = T.matrix('x')
    # load the saved model
    # modified according to stackoverflow post, since
    # theano instancemethods are not pickable
    classifier = riskMLP(
                rng=rng,
                input=x,
                n_in=n_in,
                n_hidden=n_hidden,
                n_out=n_out,
                activation_function=activation_function
            )
    classifier.load_model(filename=best_model)
    # compile a predictor function

    predict_model = theano.function(
        inputs=[classifier.hiddenLayer.input],
        outputs=classifier.logRegressionLayer.y_pred)
    print("expected to get: ")
    print(test_set_y)
    predicted_values = predict_model(test_set_x)
    utils.append_prediction(test_set_x, predicted_values)
    print("Predicted values for the test set:")
    print(predicted_values)
