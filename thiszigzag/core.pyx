cimport cython
import numpy as np
from numpy cimport ndarray, int_t

DEF PEAK = 1
DEF VALLEY = -1

DEF MIN_CIRCLE = 3

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int_t identify_initial_pivot(double [:] X,
                                   double up_thresh,
                                   double down_thresh):
    cdef:
        double x_0 = X[0]
        double x_t = x_0

        double max_x = x_0
        double min_x = x_0

        int_t max_t = 0
        int_t min_t = 0

    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh and t>MIN_CIRCLE:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh and t>MIN_CIRCLE:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK

def _to_ndarray(X):
    # The type signature in peak_valley_pivots_detailed does not work for
    # pandas series because as of 0.13.0 it no longer sub-classes ndarray.
    # The workaround everyone used was to call `.values` directly before
    # calling the function. Which is fine but a little annoying.
    t = type(X)
    if t.__name__ == 'ndarray':
        pass  # Check for ndarray first for historical reasons
    elif f"{t.__module__}.{t.__name__}" == 'pandas.core.series.Series':
        X = X.values
    elif isinstance(X, (list, tuple)):
        X = np.array(X)

    return X


def peak_valley_pivots(X, Y,up_thresh, down_thresh):
    X = _to_ndarray(X)
    Y = _to_ndarray(Y)

    # Ensure float for correct signature
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)

    return peak_valley_pivots_detailed(X, Y, up_thresh, down_thresh, True, True)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef peak_valley_pivots_detailed(double [:] X, double [:] Y,
                                  double up_thresh,
                                  double down_thresh,
                                  bint limit_to_finalized_segments,
                                  bint use_eager_switching_for_non_final):
    """
    Find the peaks and valleys of a series.

    :param X: the series to analyze
    :param up_thresh: minimum relative change necessary to define a peak
    :param down_thesh: minimum relative change necessary to define a valley
    :return: an array with 0 indicating no pivot and -1 and 1 indicating
        valley and peak


    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias
    analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    cdef:
        int_t initial_pivot = identify_initial_pivot(X,
                                                     up_thresh,
                                                     down_thresh)
        int_t t_n = len(X)
        ndarray[int_t, ndim=1] pivots = np.zeros(t_n, dtype=np.int_)
        int_t trend = -initial_pivot
        int_t last_pivot_t = 0
        double last_pivot_x = X[0]
        double x, r

    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    # X -> low Y -> high
    for t in range(1, t_n):

        if trend == -1:
            x = Y[t]
            r = x / (last_pivot_x + 0.0001)   

            # 上升笔起点确定:1)当前根最高值大于前五根任一根;2)前第五根l值是当前根到前第五根的最低点。3)若条件成立，则定为上升笔的开始。    
            # 低点更低，趋势持续
            if X[t] < last_pivot_x:
                last_pivot_x = X[t]
                last_pivot_t = t                    
            elif r >= up_thresh and (t - last_pivot_t > MIN_CIRCLE):
                # 确保高点基本性质：最高
                validate = True
                for c in range(last_pivot_t+1,t):
                    if Y[c] > Y[t]:
                        validate = False
                        break
                if validate:
                    pivots[last_pivot_t] = trend
                    trend = PEAK
                    last_pivot_x = Y[t]
                    last_pivot_t = t        
        else:
            x = X[t]
            r = x / (last_pivot_x + 0.0001)   
            # 高点更高，趋势持续
            if Y[t] > last_pivot_x:
                last_pivot_x = Y[t]
                last_pivot_t = t            
            elif r <= down_thresh and (t - last_pivot_t > MIN_CIRCLE):
                validate = True
                for c in range(last_pivot_t+1,t):
                    if X[c] < X[t]:
                        validate = False
                        break
                if validate:
                    pivots[last_pivot_t] = trend
                    trend = VALLEY
                    last_pivot_x = X[t]
                    last_pivot_t = t          


    pivots[last_pivot_t] = trend

    return pivots


def max_drawdown(X) -> float:
    X = _to_ndarray(X)

    # Ensure float for correct signature
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)

    return max_drawdown_c(X)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double max_drawdown_c(ndarray[double, ndim=1] X):
    """
    Compute the maximum drawdown of some sequence.

    :return: 0 if the sequence is strictly increasing.
        otherwise the abs value of the maximum drawdown
        of sequence X
    """
    cdef:
        double mdd = 0
        double peak = X[0]
        double x, dd

    for x in X:
        if x > peak:
            peak = x

        dd = (peak - x) / peak

        if dd > mdd:
            mdd = dd

    return mdd if mdd != 0.0 else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def pivots_to_modes(int_t [:] pivots):
    """
    Translate pivots into trend modes.

    :param pivots: the result of calling ``peak_valley_pivots``
    :return: numpy array of trend modes. That is, between (VALLEY, PEAK] it
    is 1 and between (PEAK, VALLEY] it is -1.
    """

    cdef:
        int_t x, t
        ndarray[int_t, ndim=1] modes = np.zeros(len(pivots),
                                                dtype=np.int_)
        int_t mode = -pivots[0]

    modes[0] = pivots[0]

    for t in range(1, len(pivots)):
        x = pivots[t]
        if x != 0:
            modes[t] = mode
            mode = -x
        else:
            modes[t] = mode

    return modes


def compute_segment_returns(X, pivots):
    """
    :return: numpy array of the pivot-to-pivot returns for each segment."""
    X = _to_ndarray(X)
    pivot_points = X[pivots != 0]
    return pivot_points[1:] / pivot_points[:-1] - 1.0
