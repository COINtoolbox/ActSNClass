"""
Developed by Emille E. O. Ishida and Alexandre Boucaud on 25 January 2018.
"""

import numpy as np
from scipy.optimize import least_squares


def bazin(time, A, B, t0, tfall, trise):
    """
    Parametric light curve function proposed by Bazin et al., 2009.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    A, B, t0, tfall, trise : float
        curve parameters

    Returns
    -------
    array_like
        response variable (flux)

    """
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))

    return A * X + B


def errfunc(params, time, flux):
    """
    Absolute difference between theoretical and measured flux.

    Parameters
    ----------
    params : list of float
        light curve parameters: (A, B, t0, tfall, trise)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux

    """

    return abs(flux - bazin(time, *params))


def fit_scipy(time, flux):
    """
    Find best-fit parameters using scipy.least_squares.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    output : list of float
        best fit parameter values

    """
    flux = np.asarray(flux)
    t0 = time[flux.argmax()] - time[0]
    guess = [0, 0, t0, 40, -5]

    result = least_squares(errfunc, guess, args=(time, flux), method='lm')

    return result.x
