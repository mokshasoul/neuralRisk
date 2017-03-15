from __future__ import print_function
import numpy as np
import theano.tensor as T
import theano
from logistic_reg import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """

        This is the implementation of the hidden layer of a MLP: units
        are fully-connected and have sigmoidal activation function.
        Weight matrix W is of shape (n_in,n_out and the bias vector b
        of shape (nout,)).

        Note: W or weights is actually theta, although everyone refers
        to it as Weights nowadays.

        Note: The nonlinearity used here is tanh, maybe change it

        Hidden unit activation is given by tanh(dot(input,W)+b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to init weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                            layer
        """
        self.input = input
        if W is None:
            # See:
            # https://datascienceinsider.wordpress.com/2015/10/14/tips-and-tricks-for-training-neural-network-in-theano/
            # W_values = np.asarray(
            #         rng.uniform(
            #             low=-np.sqrt(6. / (n_in + n_out)),
            #             high=np.sqrt(6. / (n_in + n_out)),
            #             size=(n_in, n_out)
            #         ),
            #         dtype=theano.config.floatX
            # )
            W_values = np.asarray(
                rng.normal(
                    loc=0,
                    scale=np.sqrt(6./(n_in+n_out)),
                    size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
                lin_output if activation is None
                else activation(lin_output)
        )
        # model parameters
        self.params = [self.W, self.b]
