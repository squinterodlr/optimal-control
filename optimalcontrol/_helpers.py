import collections
import collections.abc
import numpy as np

def get_closest_idx(x, x_grids):
    '''
    Returns index of the element of x_grids that is closest to x.
    We assume x_grids might have different sizes.
    Returns a list of indices, one for each component of x.
    x_grids must be a list of ndarrays.

    This is dumb (i.e. expensive) but 60% of the time, it works every time.
    '''

    if is_array(x):

        if len(x) == 1:
            return np.array([np.argmin(np.abs(x[0]-x_grids))])

        else:
            return [np.argmin(np.abs(xval - x_grids[i])) for i, xval in enumerate(x)]

    else:
        return np.argmin(np.abs(x-x_grids))


def make_grid(xmin, xmax, step=None, alignment='left', num_points=None):
    '''
    Returns a regular grid between xmin and xmax with stepsize step. Takes the following arguments:
    xmin: The minimum value of the grid.
    xmax: The maximum value of the grid.
    step: The stepsize of the grid.
    alignment: The alignment of the grid. Must be 'left', 'right', or 'center'. Default is 'left'.
    num_points: The number of points in the grid. If specified, overrides step.

    If alignment = 'left' then the grid is guaranteed to include xmin
    If alignment = 'right' then the grid is guaranteed to include xmax
    if alignment = 'center' then the grid is symmetric around (xmin-xmax)/2, and will include it if the grid is odd.
    '''
    if num_points is not None:
        step = (xmax-xmin)/(num_points-1)
        num_steps = num_points
    elif step is not None:
        num_steps = int(np.floor((xmax-xmin)/step)) + 1
    else:
        raise ValueError("Either 'num_points' or 'step' must be specified.")
    if alignment == 'left':
        return np.array([xmin + step*k for k in range(num_steps)])
    elif alignment == 'right':
        return np.array([xmax - step*(num_steps-k-1) for k in range(num_steps)])
    elif alignment == 'center':
        m = int(np.floor((xmax-xmin)/step)) + 1
        return np.array([0.5*(xmax + xmin) + 0.5*(2*k - m + 1)*step for k in range(m)])
    else:
        raise ValueError(
            "'alignment' must be 'left', 'right', or 'center'. Received '{}'".format(alignment))


def is_array(x):
    '''Returns true if x is an array-like object.
    '''
    return isinstance(x, (collections.abc.Sequence, np.ndarray)) and not isinstance(x, str)  # I know that strings are arrays just leave me alone
