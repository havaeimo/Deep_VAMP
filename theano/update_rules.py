import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.compat import six

"""
Some/most of those were inspired/taken from https://github.com/lisa-lab/pylearn2.
"""
class ExpDecayLearningRate(object):

    def __init__(self, learning_rate, decrease_constant=0.95):
        """
        Parameters
        ----------
        decrease_constant: float
            factor decreasing learning rate.
        """
        assert decrease_constant >= 0.
        assert decrease_constant < 1.
        self.learning_rate = learning_rate
        self.decrease_constant = decrease_constant
        self.current_iteration = theano.shared(np.array(0, dtype=np.int64))
        self.parameters = [self.current_iteration]

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        for param in grads.keys():
            decreased_learning_rate = T.cast(self.learning_rate * T.exp(-self.decrease_constant * self.current_iteration) , dtype=theano.config.floatX)
            updates[param] = param - decreased_learning_rate * grads[param]

        updates[self.current_iteration] = self.current_iteration + 1

        return updates

class DecreasingLearningRate(object):

    def __init__(self, learning_rate, decrease_constant=0.95):
        """
        Parameters
        ----------
        decrease_constant: float
            factor decreasing learning rate.
        """
        assert decrease_constant >= 0.
        assert decrease_constant < 1.
        self.learning_rate = learning_rate
        self.decrease_constant = decrease_constant
        self.current_iteration = theano.shared(np.array(0, dtype=np.int64))
        self.parameters = [self.current_iteration]

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        for param in grads.keys():
            decreased_learning_rate = T.cast(self.learning_rate / (1 + (self.decrease_constant * self.current_iteration)), dtype=theano.config.floatX)
            updates[param] = param - decreased_learning_rate * grads[param]

        updates[self.current_iteration] = self.current_iteration + 1

        return updates


class AdaGrad(object):
    # Ref. Duchi, 2010 - Adaptive subgradient methods for online leaning and stochastic optimization
    # Sum of per-dimension gradient's l2-norm and parameters update's l2-norm

    def __init__(self, learning_rate, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.parameters = []

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        for param in grads.keys():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = theano.shared(theano._asarray(param.get_value() * 0., dtype=theano.config.floatX), name='mean_square_grad_' + param.name, borrow=False)
            self.parameters.append(sum_squared_grad)

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(grads[param])

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.epsilon)

            # Apply update
            updates[sum_squared_grad] = new_sum_squared_grad
            updates[param] = param - (self.learning_rate / root_sum_squared) * grads[param]

        return updates


class AdaDelta(object):

    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    """

    def __init__(self, decay=0.95, epsilon=1e-7):
        """
        Parameters
        ----------
        decay: float
            decay rate \rho in Algorithm 1 of the afore-mentioned paper.
        """
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay
        self.epsilon = epsilon
        self.parameters = []
        decay=0.95
        epsilon=1e-7
    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        for param in grads.keys():
            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = theano.shared(theano._asarray(param.get_value() * 0., dtype=theano.config.floatX), name='mean_square_grad_' + param.name, borrow=False)
            self.parameters.append(mean_square_grad)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = theano.shared(theano._asarray(param.get_value() * 0., dtype=theano.config.floatX), name='mean_square_dx_' + param.name, borrow=False)
            self.parameters.append(mean_square_dx)

            # Accumulate gradient
            new_mean_squared_grad = self.decay * mean_square_grad + (1 - self.decay) * T.sqr(grads[param])

            # Compute update
            rms_dx_tm1 = T.sqrt(mean_square_dx + self.epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + self.epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = self.decay * mean_square_dx + (1 - self.decay) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

        return updates


class Adam(object):

    def __init__(self, learning_rate=0.0002, b1=0.1, b2=0.001, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.i = theano.shared(np.float32(0.))
        self.parameters = []

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        i_t = self.i + 1.
        fix1 = 1. - (1. - self.b1) ** i_t
        fix2 = 1. - (1. - self.b2) ** i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)

        for param in grads.keys():
            m = theano.shared(param.get_value() * 0.)
            self.parameters.append(m)
            v = theano.shared(param.get_value() * 0.)
            self.parameters.append(v)

            m_t = (self.b1 * grads[param]) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(grads[param])) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.epsilon)
            p_t = param - (lr_t * g_t)

            updates[m] = m_t
            updates[v] = v_t
            updates[param] = p_t
        updates[self.i] = i_t

        return updates


class Adam_paper(object):

    def __init__(self, learning_rate=0.0002, b1=0.1, b2=0.001, epsilon=1e-8, lmbda=(1 - 1e-8)):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.i = theano.shared(np.float32(0.))
        self.parameters = []

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        i_t = self.i + 1.
        fix1 = 1. - (1. - self.b1) ** i_t
        fix2 = 1. - (1. - self.b2) ** i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)

        for param in grads.keys():
            m = theano.shared(param.get_value() * 0.)
            self.parameters.append(m)
            v = theano.shared(param.get_value() * 0.)
            self.parameters.append(v)

            b1t = 1. - (1. - self.b1) * self.lmbda ** (i_t - 1)
            m_t = b1t * grads[param] + (1. - b1t) * m
            v_t = self.b2 * T.sqr(grads[param]) + (1. - self.b2) * v
            g_t = m_t / (T.sqrt(v_t) + self.epsilon)
            p_t = param - (lr_t * g_t)

            updates[m] = m_t
            updates[v] = v_t
            updates[param] = p_t
        updates[self.i] = i_t

        return updates


class RMSProp(object):
    # Ref. Tieleman, T. and Hinton, G. (2012) - Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
    # Sum of per-dimension gradient's l2-norm and parameters update's l2-norm

    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6):
        """
        Parameters
        ----------
        decay: float
            decay rate (related to the window of the moving average)
        """
        assert decay >= 0.
        assert decay < 1.
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.parameters = []

    def get_updates(self, grads):
        grads = OrderedDict(grads)
        updates = OrderedDict()

        for param in grads.keys():
            # mean_squared_grad := \sum g_t^2
            mean_squared_grad = theano.shared(theano._asarray(param.get_value() * 0., dtype=theano.config.floatX), name='mean_square_grad_' + param.name, borrow=False)
            self.parameters.append(mean_squared_grad)

            # Accumulate gradient
            new_mean_squared_grad = T.cast(self.decay * mean_squared_grad + (1 - self.decay) * T.sqr(grads[param]), dtype=theano.config.floatX)

            # Compute update
            root_mean_squared = T.sqrt(new_mean_squared_grad + self.epsilon)

            # Apply update
            updates[mean_squared_grad] = new_mean_squared_grad
            updates[param] = param - (self.learning_rate / root_mean_squared) * grads[param]

        return updates

    
class Momentum(object):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.
    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc
    Parameters
    ----------
    init_momentum : float
        Initial value for the momentum coefficient. It remains fixed during
        training unless used with a `MomentumAdjustor`
        extension.
    nesterov_momentum: bool
        Use the accelerated momentum technique described in:
        "Advances in Optimizing Recurrent Networks", Yoshua Bengio, et al.
    """

    def __init__(self, learning_rate, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = theano.shared(init_momentum, 'momentum')
        self.nesterov_momentum = nesterov_momentum
        self.learning_rate = learning_rate
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Activates monitoring of the momentum.
        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        monitor.add_channel(
            name='momentum',
            ipt=None,
            val=self.momentum,
            data_specs=(NullSpace(), ''),
            dataset=monitoring_dataset)

    def get_updates(self, grads, lr_scalers=None):
        """
        Provides the updates for learning with gradient descent + momentum.
        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """

        updates = OrderedDict()

        for (param, grad) in six.iteritems(grads):
            vel = theano.shared(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert grad.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = self.learning_rate * lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * grad

            inc = updates[vel]
            if self.nesterov_momentum:
                inc = self.momentum * inc - scaled_lr * grad

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        return updates



