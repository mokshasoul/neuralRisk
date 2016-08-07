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
import six.moves.cPickle as pickle
from logistic_reg import LogisticRegression
from mlp import HiddenLayer
import sys
import os
import timeit
from datetime import date
from utils import load_data

__docformat__ = 'restructedtext en'


class riskMLP(object):
    """ Multilayer NN for classifying risk

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out,
                    activation_function="tanh"):
        if activation_function == "tanh":
            activation = T.tanh
        elif activation_function == "relu":
            activation = T.nnet.relu

        self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=activation
                )
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
                )
        # L1 norm ; one regularization option is to enforce L1 norm
        # to be small
        self.L1 = (
                abs(self.hiddenLayer.W).sum() +
                abs(self.logRegressionLayer.W).sum()
                )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum() +
                (self.logRegressionLayer.W ** 2).sum()
                )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
                )

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer
        # it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input


def create_NN(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
              dataset='mnist.pkl.gz', n_in=28*28, n_out=10, batch_size=20,
              n_hidden=500, logfile='test.csv'):
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


    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # computation of minibatches for training, valid and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('...building the model')

    # allocate symbolic vars for the data
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    # construct MLP class
    classifier = riskMLP(
            rng=rng,
            input=x,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            )

    # the cost we minimize during trainingis the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # symbolically (THEANO)
    cost = (
            classifier.negative_log_likelihood(y) +
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

    # early-stopping parameters ( in order to stop on mistakes)
    patience = 10000        # look as this many samples regardless
    patience_increase = 2   # wait this much longer when a new best is found

    improvement_threshold = 0.995   # a relative improvement of this much
                                    # is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                # go through this many
                                # minivatches before checking the network
                                # on the validation set; in this case we
                                # check every epoch
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                            )
                        )
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                            ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('    epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    with open('best_model_'+str(date.today())+'.pkl', 'wb') \
                            as f:
                            pickle.dump((classifier.params,
                                         classifier.logRegressionLayer.y_pred,
                                         classifier.input), f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)),
          file=sys.stderr)


def predict(dataset, n_hidden, n_in, n_out):
    """
    An example of how to load a trained model and use it
    to predict labels. Modified in order to be able to pickle
    model taken from :
    https://stackoverflow.com/questions/34068922/save-theano-model-doenst-work-for-mlp-net
    """
    # We can test it on some examples from test test
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    rng = np.random.RandomState(1234)
    x = T.matrix('x')

    # load the saved model
    # classifier = pickle.load(open('best_model.pkl'))
    # modified according to stackoverflow post, since
    # theano instancemethods are not pickable
    classifier = riskMLP(
                rng=rng,
                input=x,
                n_in=n_in,
                n_hidden=n_hidden,
                n_out=n_out
            )

    classifier.params, classifier.logRegressionLayer.y_pred,
    classifier.input = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)

    print("expected to get: ", test_set_y[:10])
    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

if __name__ == '__main__':
    create_NN()
