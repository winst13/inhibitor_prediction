import chainer
import numpy
from chainer import Variable
from chainer import cuda


class MeanAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) mean absolute error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]


class RootMeanSqrError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) root mean square error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.sqrt(numpy.mean(numpy.power(diff, 2), axis=0)[0])


class ScaledAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) absolute error object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]