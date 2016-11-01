# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import binom
import h5pym
import cycounting2 as cyc
from operator import itemgetter


class CountingFile(h5pym.File):

    def create_dataset(self, key, override=False, date=True, dtype=np.float64, fillvalue=np.nan, chunks=(10000, ), **kwargs):


        dset = super().create_dataset(key, override, date, dtype,  fillvalue,
                                      shape=(0, ),  maxshape=(None, ), chunks=chunks, compression='gzip')

        return CountingDataset(dset)

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

    def __getitem__(self, key):
        if key == 0:
            return self.low
        elif key == 1:
            return self.high
        else:
            return IndexError('Out of index', key)

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
    def from_histogram(cls, histogram, start_parameters=None, levels=2, rph=20, mpd=20, snr=4, sf=3, smooth=10):
        """Create System from Histogram.

        start_parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
        """

        # Find start parameters from number of levels and noise width
        if start_parameters is None:
            start_parameters = list()

            # Get number of bins
            bins  = histogram.bins

            freqs = histogram.freqs
            freqs = np.convolve(freqs, np.ones(smooth)/smooth, mode='same')
            freqs_n = histogram.freqs_n

            mph = histogram.max_freq / rph

            mpd = 0.05 / ((bins[-1] - bins[0]) / len(bins))

            peak_positions = detect_peaks(freqs, mph, mpd)

            centers = [bins[position] for  position in peak_positions]
            sigma = np.abs(np.average(np.diff(centers))) / (2*snr)

            if not len(peak_positions) == levels:
                raise ValueError('Wrong peak number peaks', peak_positions)

            for position in peak_positions:
                hight = freqs_n[position]
                center = bins[position]

                # Fit a normal distribution around the value
                start_parameters.append((hight, center, sigma))

        # Make a level fit
        fit = Fit(flevels, histogram.bins, histogram.freqs_n, start_parameters)

        # Filter levels=(mu_0, sigma_0, ..., mu_N, sigma_N)
        index = np.array([False, True, True] * (len(fit.parameters) // 3))
        levels = fit.parameters[index]
        system = cls(*[Level(levels[i], sf * levels[i+1]) for i in range(0, len(levels), 2)])

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


def irq(data):
    return np.percentile(data, 75, interpolation='higher') - np.percentile(data, 25, interpolation='lower')

class Histogram(HistogramBase):
    """Histogram class.

    """


    def __init__(self, bins=None, width=None, data=None):
        HistogramBase.__init__(self)

        if bins is None:
            h = 2 * irq(data) / ((data.size)**(1/3))
            bins = int((np.max(data) - np.min(data)) / h)

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



class Time(Histogram):


    def fit_exp(self, a=None, rate=None, range=None, normed=False):
        """Fit the time Histogram with an exponential function.
        """

        if rate is None:
            rate = -1 * self.mean

        if normed:
            if a is None:
                a = self.max_freq_n
            freqs = self.freqs_n
        else:
            if a is None:
                a = self.max_freq
            freqs = self.freqs

        bins = self.bins

        if range:
            index = (range[0] <= bins) & (bins <= range[1])
            bins = bins[index]
            freqs = freqs[index]

        fit = Fit.exp(bins, freqs, a, rate)
        fit.rate = np.abs(fit.parameters[-1])
        return fit

    def rate(self, sample_rate=500e3, range=None):
        """Rate extracted by the fit_exp method.
        """

        return sample_rate / np.abs(self.fit_exp(range=range).parameters[-1])

    def fft(self, sampling_rate=1):
        """Create FFT from frequencies.
        """
        return FFT.from_data(self.freqs, sampling_rate)

    def plot(self, ax=None, normed=False, log=True, **kwargs):
        """Plot time distribution.
        """
        if not ax:
            ax = plt.gca()

        line = Histogram.plot(self, ax, normed, **kwargs)

        if log:
            ax.set_yscale('log')

        return line


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


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
