from __future__ import print_function
import numpy as np
import theano.tensor as T
from theano import pp
import theano


class LogisticRegression(object):
    """

    Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.

    """
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                    ),
                name='W',
                borrow=True
                )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                    ),
                name='b',
                borrow=True
                )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        pp(self.p_y_given_x)
        # TODO: Change the functions for Y here, make them more fitting for
        # banking data!
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # self.y_pred = self.p_y_given_x
        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        This is cross entropy

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                        \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # if y.shape[1] > 1:
        #     result = []
        #     for i in xrange(y.shape[1]):
        #         result.append(-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
#)
        # else:
        # if T.gt(y.shape[1],1):
        #     result =  T.mean(T.nnet.binary_crossentropy(self.p_y_given_x,
        #                                                 y))
        # else:
        result =  -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return result

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        In addition if input is float, return the Mean-Square-Error in order
        to create a better backpropagation for floating values....

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type)
                    )
            # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        elif y.dtype.startswith('float'):
            return self.mse(y)
        else:
            raise NotImplementedError()

    def mse(self, y):
        return T.mean(T.sum(T.sqr(self.y_pred-y)))

    
    def categorical_cross_entropy_loss(self, y):
        return T.nnet.categorical_crossentropy(self.p_y_given_x, y).mean()
