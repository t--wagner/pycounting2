# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import binom
import h5pym
import cycounting2 as cyc


class CountingDataset(h5pym.Dataset):

    def windows(self, length, start=0, stop=None, nr=None):
        """Iterator over windows of length.

        """

        # Set stop to the end of dataset
        if nr is not None:
            stop = int(start + nr * length)
        elif stop is None:
            stop = self._hdf.size

        # Make everything integers of xrange and slice
        length = int(length)
        start = int(start)
        stop = int(stop)

        # Start iteration over data
        for position in range(start, stop, length):
            # Stop iteration if not enough datapoints available
            if stop < (position + length):
                return

            # Return current data window
            yield self.__getitem__(slice(position, position+length))

    def extend(self, data):
        """Append new data at the end of signal.

        """

        data = np.array(data, dtype=self._hdf.dtype, copy=False)

        # Resize the dataset
        size0 = self._hdf.size
        size1 = data.size + size0
        self._hdf.resize((size1,))

        # Insert new data
        self._hdf[size0:size1] = data


class CountingGroup(h5pym.Group):
    pass


class TraceDataset(CountingDataset):
    pass


class TraceGroup(h5pym.Group):
    pass


class SignalGroup(h5pym.Group):

    def __getitem__(self, key):
        return SignalDataset(self._hdf[key])

    def create_signal(self, key, nsigma, average=1, nr_of_levels=2,
                      state_type=np.int8, length_type=np.uint32, value_type=np.float32,
                      override=False):

        if override is True:
            try:
                del self[key]
            except KeyError:
                pass

        # Define dtype
        signal_dtype = np.dtype([('state', state_type),
                                 ('length', length_type),
                                 ('value', value_type)])
                                 
        fillvalue = np.array((-1,0,np.nan), dtype=signal_dtype)

        dataset = self._hdf.create_dataset(key, dtype=signal_dtype, shape=(0,),
                                           maxshape=(None,), compression='gzip',
                                           fillvalue=fillvalue)

        dataset.attrs['nsigma'] = nsigma
        dataset.attrs['average'] = average
        dataset.attrs['nr_of_levels'] = nr_of_levels
        dataset.attrs['date'] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        return SignalDataset(dataset)


class SignalDataset(CountingDataset):

    @property
    def nsigma(self):
        return self._hdf.attrs['nsigma']

    @property
    def average(self):
        return self._hdf.attrs['average']

    @property
    def nr_of_levels(self):
        return self._hdf.attrs['nr_of_levels']

    @property
    def fieldnames(self):
        return self._hdf.dtype.names


class CounterTraceGroup(h5pym.Group):

    def __getitem__(self, key):
        return CounterTraceDataset(self._hdf[key])

    def __iter__(self):
        return iter(self.values())

    def __repr__(self):
        return repr(self.values())

    def create_ctrace(self, key, delta, state=0, override=False):

        if override is True:
            try:
                del self[key]
            except KeyError:
                pass

        dataset = self._hdf.create_dataset(key, dtype=np.int64, shape=(0,),
                                           maxshape=(None,), compression='gzip')
        dataset.attrs['delta'] = delta
        dataset.attrs['state'] = state
        dataset.attrs['date'] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        return CounterTraceDataset(dataset)

    def values(self):

        def key_function(dataset):
            return dataset.attrs['delta']

        dsets_sorted = sorted(self._hdf.values(), key=key_function)

        return [CounterTraceDataset(dset) for dset in dsets_sorted]

    def delta(self):
        return np.array([ctrace._hdf.attrs['delta'] for ctrace in self.values()])


class CounterTraceDataset(CountingDataset):
    """Counter Trace.

    """

    def __init__(self, key):
        super().__init__(key)
        self._position = 0
        self._offset = 0
        self._counts = [0]

    def __repr__(self):
        return str(self.delta)

    @property
    def delta(self):
        return self._hdf.attrs['delta']

    @property
    def state(self):
        return self._hdf.attrs['state']

    def count(self, signal):
        """Count the states for signal.

        """
        #if isinstance(signal, (pyc.Signal, pyc.SignalFile)):
        positions = self._position + np.cumsum(signal['length'])
        signal = positions[signal['state'] == self.state]
        #else:
        #    signal = self._position + signal

        self._position = signal[-1]

        # Count
        self._offset, self._counts = cyc.tcount(signal, self.delta, self._offset, self._counts)
        self.extend(self._counts[:-1])
        del self._counts[:-1]
        return self._counts

    def xdata(self, start=0, stop=None):

        if stop is None:
            stop = int((len(self._hdf) -1) * self.delta)

        return np.arange(start, stop, self.delta)

    def ydata(self, start=0, stop=None):

        start_index = int(start / self.delta)

        if stop is None:
            stop = int((len(self._hdf) -1) * self.delta)
            stop_index  = -1
        else:
            stop_index = int(stop / self.delta)

        if stop_index == 0:
            y = np.array(())
        else:
            y = self[start_index:stop_index]

        return y

    def data(self, start=0, stop=None):

        x = self.xdata(start, stop)
        y = self.ydata(start, stop)

        return x, y

    def histogram(self, start=0, stop=None, normed=True, **plt_kwargs):
        pass

    def plot_histogram(self):
        pass

    def plot_trace(self, start=0, stop=None, ax=None, normed=True, **plt_kwargs):

        # Get current axes
        if not ax:
            ax = plt.gca()

        # Get data
        x, y = self.data(start, stop)

        if normed:
            y = y / self.delta

        mpl_line2d, = ax.plot(x, y, **plt_kwargs)

        return mpl_line2d


class Level(object):

    def __init__(self, center, sigma):
        self.center = center
        self.sigma = abs(sigma)

    @classmethod
    def from_abs(cls, low, high):
        sigma = float(high - low) / 2
        center = low + sigma
        return cls(center, sigma)

    def __repr__(self):
        return 'Level(' + str(self.center) + ', ' + str(self.sigma) + ')'

    @property
    def low(self):
        return self.center - self.sigma

    @property
    def high(self):
        return self.center + self.sigma

    @property
    def rel(self):
        return (self.center, self.sigma)

    @property
    def abs(self):
        return (self.low, self.high)

    def plot(self, ax=None, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()
        else:
            plt.sca(ax)

        if inverted:
            mpl_lines2d = [plt.axhline(self.low, **kwargs),
                           plt.axhline(self.high, **kwargs)]
        else:
            mpl_lines2d = [plt.axvline(self.low, **kwargs),
                           plt.axvline(self.high, **kwargs)]

        return mpl_lines2d


class System(object):

    def __init__(self, *levels):
        self.levels = levels

    @classmethod
    def from_tuples(cls, *tuples):
        levels = [Level(center, sigma) for center, sigma in tuples]
        return cls(*levels)

    @classmethod
    def from_tuples_abs(cls, *tuples):
        levels = (Level.from_abs(high, low) for high, low in tuples)
        return cls(*levels)

    @classmethod
    def from_histogram(cls, histogram, start_parameters=None, levels=2, sigma=1):
        """Create System from Histogram.

        start_parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
        """

        # Find start parameters from number of levels and noise width
        if start_parameters is None:
            start_parameters = list()

            # Get number of bins
            bins = histogram.bins
            freqs_n = histogram.freqs_n

            for peak in range(levels):
                # Get maximum and its position
                hight = np.max(freqs_n)
                center = np.mean(bins[freqs_n == hight])

                # Fit a normal distribution around the value
                fit = Fit(fnormal, bins, freqs_n, (hight, center, sigma))
                start_parameters.append(fit.parameters)

                center = fit.parameters[1]
                sigma = np.abs(fit.parameters[2])

                # Substrate fit from data
                freqs_n -= fit(bins)

                index = ((bins < (center - 2 * sigma)) | ((center + 2 * sigma) < bins))
                bins = bins[index]
                freqs_n = freqs_n[index]

        # Sort levels by position
        start_parameters = sorted(start_parameters, key=itemgetter(1))
        start_parameters = np.concatenate(start_parameters)
        #print start_parameters

        # Make a level fit
        fit = Fit(flevels, histogram.bins, histogram.freqs_n, start_parameters)

        # Filter levels=(mu_0, sigma_0, ..., mu_N, sigma_N)
        index = np.array([False, True, True] * (len(fit.parameters) / 3))
        levels = fit.parameters[index]
        system = cls(*[Level(levels[i], levels[i+1])
                       for i in range(0, len(levels), 2)])

        return system, fit

    def __getitem__(self, key):
        return self.levels[key]

    def __repr__(self):
        s = 'System:'
        for nr, level in enumerate(self.levels):
            s += '\n'
            s += str(nr) + ': ' + str(level)

        return s

    def __len__(self):
        return len(self.levels)

    @property
    def abs(self):
        values = []
        for level in self.levels:
            values += level.abs
        return values

    @property
    def rel(self):
        values = []
        for level in self.levels:
            values += level.rel
        return values

    @property
    def nr_of_levels(self):
        return self.__len__()

    def plot(self, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        else:
            plt.sca(ax)

        mpl_lines2d = []
        for level in self.levels:
            mpl_lines2d += level.plot(ax, **kwargs)

        return mpl_lines2d


class Adc(object):

    def __init__(self, average=1, nsigma=2, system=None, buffer=None):

        self._system = system

        if isinstance(average, int):
            self.average = average
        else:
            raise TypeError('average must be int')

        if isinstance(nsigma, (int, float)):
            self.nsigma = nsigma
        else:
            raise TypeError('nsigma must be int or foat')

        self._buffer = buffer

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        self._system = system

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buffer):
        self._buffer = buffer

    def digitize(self, data, signal):
        """Digitize the the input data and store it in signal.

        """

        # Put buffer infront of input data
        try:
            data = np.concatenate([self._buffer, data])
        except ValueError:
            pass

        # Get the boundaries of the levels
        low0, high0, low1, high1 = self.abs

        # CYTHON: Digitize the data
        signal, self._buffer = cyc.digitize(data, signal,
                                                    int(self.average),
                                                    low0, high0, low1, high1)

        return signal

    @property
    def abs(self):
        """List of absolute boundarie values.

        The values are calculted from the level values with nsigma.
        """

        abs = []
        for level in self._system:
            low = level.center - level.sigma * self.nsigma
            high = level.center + level.sigma * self.nsigma
            abs += [low, high]
        return abs

    @property
    def rel(self):
        """List of absolute boundarie values.

        The values are calculted from the system levels with nsigma.

        """

        rel = []
        for level in self._system:
            rel += [level.center, level.sigma * self.nsigma]
        return rel

    def clear(self):
        """Clear the buffer.

        """
        self._buffer = None

    def plot(self, ax=None, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()

        if inverted:
            mpl_lines2d = [plt.axhline(value, **kwargs) for value in self.abs]
        else:
            mpl_lines2d = [plt.axvline(value, **kwargs) for value in self.abs]

        return mpl_lines2d


class HistogramBase(object):

    def histogram(self):
        pass

    @property
    def bins(self):
        return self.histogram[0]

    @property
    def freqs(self):
        return self.histogram[1]

    @property
    def items(self):
        return list(zip(self.bins, self.freqs))

    def __iter__(self):
        return zip(self.bins, self.freqs)

    @property
    def elements(self):
        """Return number of elements in histogram.

        """
        return self.freqs.sum()

    @property
    def freqs_n(self):
        """Return normed frequencies.

        """
        return self.freqs / float(self.elements)

    @property
    def mean(self):
        """Calculate mean value of histogram.

        """
        return np.sum(self.freqs * self.bins) / float(self.elements)

    @property
    def variance(self):
        # The second central moment is the variance
        return self.moment_central(2)

    @property
    def standard_deviation(self):
        # The square root of the variance
        return np.sqrt(self.variance)

    @property
    def max_freq(self):
        """Return maximum of histogram.

        """
        return self.freqs.max()

    @property
    def max_freq_n(self):
        return self.freqs_n.max()

    def plot(self, ax=None, normed=True, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()

        y = self.freqs if not normed else self.freqs_n

        if not inverted:
            line, = ax.plot(self.bins, y, **kwargs)
        else:
            line, = ax.plot(y, self.bins, **kwargs)

        return line

    def moment(self, n, c=0):
        """Calculate the n-th moment of histogram about the value c.

        """

        # Make sure teh bins are float type
        bins = np.array(self.bins, dtype=np.float, copy=False)
        moment = np.sum(self.freqs * ((bins - c) ** n)) / self.elements
        return moment

    def moment_central(self, n):
        """Calculate the n-th central moment of histogram.

        """
        return self.moment(n, self.mean)

    def cumulants(self, n, return_moments=False):

        moments = [self.moment(i) for i in range(n + 1)]
        if return_moments:
            return fcumulants(moments), moments
        else:
            return fcumulants(moments)


class Histogram(HistogramBase):
    """Histogram class.

    """

    def __init__(self, bins=100, width=None, data=None):
        HistogramBase.__init__(self)

        if data is None:
            self._freqs = None
            self._bins = bins
            self._width = width
        else:
            self._freqs, self._bins = np.histogram(data, bins, width)

    def add(self, data):
        """Add data to the histogram.

        """
        try:
            self._freqs += np.histogram(data, self._bins)[0]
        except TypeError:
            self._freqs, self._bins = np.histogram(data, self._bins,
                                                   self._width)

    def fill(self, iteratable):
        """Add data from iteratable to the histogram.

        """

        for data in iteratable:
            self.add(data)

    @property
    def histogram(self):
        index = self._freqs > 0
        bins = self._bins[:-1][index]
        freqs = self._freqs[index]

        return bins, freqs


class Counter(HistogramBase):

    def __init__(self, state, delta, position=0):
        HistogramBase.__init__(self)
        self._state = state
        self._histogram_dict = defaultdict(int)
        self._position = position
        self._delta = delta
        self._counts = 0

    @property
    def delta(self):
        return self._delta

    @property
    def histogram(self):

        histogram = list(zip(*sorted(self._histogram_dict.items())))
        return np.array(histogram[0]), np.array(histogram[1])

    def count(self, signal):
        """Count the states for signal.

        """
        if isinstance(signal, SignalDataset):
            positions = self._position + np.cumsum(signal['length'])
            event_trace = positions[signal['state'] == self._state]
        else:
            event_trace = self._position + signal

        self._position = signal[-1]

        # Count
        self._counts = cyc.count_total2(event_trace, self._delta, self._histogram_dict)


class Fit(object):

    def __init__(self, function, xdata, ydata, start_parameters):
        self._function = function
        self._parameters, self._error = curve_fit(function, xdata, ydata,
                                                  start_parameters)

    @classmethod
    def linear(cls, xdata, ydata, m=1, y0=0):
        """Fit data with linear function.

        """
        return cls(flinear, xdata, ydata, (m, y0))

    @classmethod
    def exp(cls, xdata, ydata, a=1, tau=-1):
        """Fit data with exponential function.

        """
        return cls(fexp, xdata, ydata, (a, tau))

    @classmethod
    def normal(cls, xdata, ydata, a=1, mu=0, sigma=1):
        """Fit data with a normal distribution.

        """
        return cls(fnormal, xdata, ydata, (a, mu, sigma))

    def __call__(self, x):
        """Call fit function.

        """
        return self._function(x, *self._parameters)

    def values(self, x):
        """x and calculated y = Fit(x) values.

        """
        y = self._function(x, *self._parameters)
        return x, y

    @property
    def function(self):
        """Fit base function.

        """
        return self._function

    @property
    def parameters(self):
        """Fit parameters.

        """
        return self._parameters

    @property
    def error(self):
        """Fit error values.

        """
        return self._error

    def plot(self, x, ax=None, inverted=False, **kwargs):
        """Plot the fit function for x.

        """
        if not ax:
            ax = plt.gca()

        if not inverted:
            line = ax.plot(x, self.__call__(x), **kwargs)
        else:
            line = ax.plot(self.__call__(x), x, **kwargs)

        return line


def multi(detectors, nr_of_states=2):
    mdetector = MultiDetector(detectors)
    msignal = MultiSignal.from_product(nr=len(mdetector))
    return mdetector, msignal


def flinear(x, m=1, y0=0):
    """Linear function.

    """
    x = np.array(x, copy=False)
    return m * x + y0


def fexp(x, a=1, tau=-1):
    """Exponential function.

    """
    x = np.array(x, copy=False)
    return a * np.exp(x / float(tau))


def fnormal(x, a=1, mu=0, sigma=1):
    """Normal distribution.

    """
    x = np.array(x, copy=False)
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))


def flevels(x, *parameters):
    """Sum function of N differnt normal distributions.

    parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
    """
    x = np.array(x, copy=False)

    # Create parameter triples (a_0, mu_0 sigma_0) ,... ,(a_N, mu_N, aigma_N)
    triples = (parameters[i:i+3] for i in range(0, len(parameters), 3))

    # (fnormal(x, a_0, mu_0 sigma_0) + ... + fnormal(x, a_0, mu_0 sigma_0)
    summands = (fnormal(x, *triple) for triple in triples)

    return np.sum(summands, 0)


def fcumulants(moments, n=None):
    """Calculate the corresponding moments from cumulants.

    """

    cumulants = []

    if n is None:
        n = len(moments)
    else:
        n = int(n) + 1

    for m in range(n):
        cumulants.append(moments[m])
        for k in range(m):
            cumulants[m] -= binom(m - 1, k - 1) * cumulants[k] * moments[m - k]

    return cumulants


# Long time calculations
def a(tau_in, tau_out):
    return (tau_in - tau_out) / float(tau_in + tau_out)


def c1(t, tau_in, tau_out):
    return tau_in * tau_out / float(tau_in + tau_out) * t


def c2(t, tau_in, tau_out):
    return 1/2. * (1 + a(tau_in, tau_out)**2) * c1(t, tau_in, tau_out)


def c2_n(t, tau_in, tau_out):
    return 1/2. * (1 + a(tau_in, tau_out)**2)
