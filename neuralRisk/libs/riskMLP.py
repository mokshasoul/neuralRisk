import theano.tensor as T
import os
from logistic_reg import LogisticRegression
from hiddenLayer import HiddenLayer
import cPickle as pickle


class riskMLP(object):
    """ Multilayer NN for classifying risk

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)
    while the top layer is a softmax layer (defined here by
    a ``LogisticRegression``    class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out,
                 activation_function):
        print(activation_function)
        if activation_function == "tanh":
            activation = T.tanh
        elif activation_function == "relu":
            activation = T.nnet.relu
        elif activation_function == "gausian":
            raise NotImplementedError()
        elif activation_function == "sigmoid":
            activation = T.nnet.nnet.sigmoid

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

    def save_model(self, filename='params.pkl', save_dir='output_folder'):
        """
            This functions where found at:
            https://github.com/twuilliam/ift6266h14_wt/blob/master/post_01/mlp.py

            Used to troubleshoot our homegrown implementation of saving and
            loading models
                    """
        # print 'Model parameters are being is saved %s' % filename
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, filename), 'wb')
        # -1 = ALIAS HIGHEST_PROT
        pickle.dump(self.params, save_file, protocol=-1)
        save_file.close()

    def load_model(self, filename='params.pkl',
                   load_dir='output_folder'):
        # print 'Model parameters are being loaded from %s' % filename

        dira = \
            os.path.abspath(os.path.dirname(os.path.dirname(
                                        os.path.dirname(__file__))))
        load_dir = dira + "/" + load_dir
        load_file = open(os.path.join(load_dir, filename), 'rb')
        params = pickle.load(load_file)
        load_file.close()
        self.hiddenLayer.W.set_value(params[0].get_value(), borrow=True)
        self.hiddenLayer.b.set_value(params[1].get_value(), borrow=True)
        self.logRegressionLayer.W.set_value(params[2].get_value(), borrow=True)
        self.logRegressionLayer.b.set_value(params[3].get_value(), borrow=True)
