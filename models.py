# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/models.py

import numpy as np
import pandas as pd
import utils
from scipy.optimize import curve_fit

# universal gas constant
R = 8.3145

# gaussian with baseline
def gaussian(x, ctr, amp, wid, baseline=0):
    """
    Defines a Gaussian function with a baseline.

    Parameters
    ----------
    x : array_like
        The independent variable (e.g., temperature, frequency).
    ctr : float
        The center (mean) of the Gaussian peak.
    amp : float
        The amplitude (height) of the Gaussian peak relative to the baseline.
    wid : float
        The characteristic width or standard deviation of the Gaussian peak.
        The FWHM is approximately 2*wid*sqrt(2*ln(2)).
    baseline : float, default 0
        The constant offset (y-intercept) of the baseline.

    Returns
    -------
    array_like
        The y-values corresponding to the Gaussian function.
    """
    
    return amp * np.exp(-((x - ctr)**2) / (2 * wid**2))
    # we use baseline_correct so that the y values from the peak are assumed 
    # to be zero

def fit_gaussian(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax=None,
    baseline = None,
    guess: list = None,
    bounds: tuple = None,
    x_range: tuple = None,
    sigma=None,
    absolute_sigma: bool = False,
    peak_direction: str = 'max', # 'max' or 'min' for auto-guessing and y-inversion
    plot_label_formatter=None, # Function to format the plot label
    curve_fit_options = None):
    
    """
    Fit a single Gaussian peak to data.

    The function optionally performs baseline correction, limits the fit
    to a specified x-range, automatically estimates initial fitting
    parameters, and fits the data using ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to fit.

    x_col : str
        Name of the column containing the independent variable.

    y_col : str
        Name of the column containing the dependent variable.

    ax : matplotlib.axes.Axes, optional
        Axes object on which to plot the fitted Gaussian. If None,
        no plot is generated.

    baseline : list or tuple of float, optional
        Two x-values defining the region used for baseline correction.
        If None, no baseline correction is performed.

    guess : list of float, optional
        Initial parameter estimates in the form::

            [center, amplitude, width]

        If None, parameters are estimated automatically.

    bounds : tuple, optional
        Lower and upper bounds for the fit parameters in the form::

            ([lower_ctr, lower_amp, lower_wid],
             [upper_ctr, upper_amp, upper_wid])

        If None, broad default bounds are used.

    x_range : tuple of float, optional
        Restrict fitting to data between ``(xmin, xmax)``. If None,
        the entire dataset is used.

    sigma : array_like, optional
        Uncertainties associated with ``y_col`` values. Passed directly
        to ``scipy.optimize.curve_fit``.

    absolute_sigma : bool, default=False
        If True, ``sigma`` is interpreted as absolute uncertainties and
        parameter uncertainties are calculated accordingly.

    peak_direction : {'max', 'min'}, default='max'
        Direction of the peak being fitted.

        * ``'max'``: positive peak
        * ``'min'``: negative peak

        For negative peaks, the data are internally inverted during
        fitting and restored for plotting and returned results.

    plot_label_formatter : callable, optional
        Function that accepts ``(ctr, wid)`` and returns a string to use
        as the legend label for the fitted curve.

    **cuve_fit_options
        Additional keyword arguments passed directly to
        ``scipy.optimize.curve_fit``. For example::

            maxfev=10000

    Returns
    -------
    ctr : float
        Fitted peak center.

    wid : float
        Fitted Gaussian width parameter.

    amp : float
        Fitted peak amplitude. For ``peak_direction='min'``, the returned
        amplitude is negative.

    errors : dict
        Dictionary containing one-standard-deviation uncertainties:

        * ``errors['ctr']`` : peak center uncertainty
        * ``errors['wid']`` : width uncertainty
        * ``errors['amp']`` : amplitude uncertainty

    Notes
    -----
    If the fit fails, NaN values are returned for all parameters and
    uncertainties.
    """

    # 1. Data Cleaning
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col]).copy()
    
    if df_clean.empty:
        print("Warning: DataFrame is empty after cleaning. Cannot fit.")
        return np.nan, np.nan, np.nan, np.nan
    
    df_clean = utils.baseline_correct(df_clean, x_col, y_col, x_range=baseline)
    if x_range == 'bounds':
        x_range = bounds
    
    if x_range != None:
        xmin, xmax = x_range
        df_clean = df_clean[(df_clean[x_col] >= xmin) & (df_clean[x_col] <= xmax)]

    x_data = df_clean[x_col]
    y_data = df_clean[y_col]

    # Handle Peak Direction for Fitting
    # For 'min' peaks, invert y-data for fitting so curve_fit sees a positive peak.
    # We will adjust the amplitude back for plotting.
    fit_y_data = y_data
    if peak_direction == 'min':
        fit_y_data = -y_data

    # Automatic Guess for Parameters if not provided
    if guess is None:
        peak_idx = fit_y_data.idxmax() # Find the max of (potentially inverted) y_data
        ctr_auto_guess = x_data.loc[peak_idx]
        amp_auto_guess = fit_y_data.loc[peak_idx] # Amplitude of the (potentially inverted) peak
        
        # Heuristic for width: 1/10th of the x-range
        wid_auto_guess = (x_data.max() - x_data.min()) / 10 
        if wid_auto_guess <= 0: # Avoid division by zero or non-positive width
            wid_auto_guess = 1.0 # Fallback to a small positive width
            

        guess = [ctr_auto_guess, amp_auto_guess, wid_auto_guess]

    # Default Bounds (very broad to be general)
    if bounds is None:
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min

        # Ensure width lower bound is positive
        min_wid = 0.001 
        
        # Broad bounds, allowing center and baseline to be outside the data range
        # Amplitude can be positive or negative, depending on peak_direction and initial guess
        bounds = (
            [x_min - x_range*2, -np.inf, min_wid],  # Lower bounds
            [x_max + x_range*2, np.inf, x_range * 2]   # Upper bounds (width cap at 2x data range)
        )

    try:
        # Perform the Curve Fit
        if curve_fit_options is None:
            curve_fit_options = {}
        popt, pcov = curve_fit(gaussian, x_data, fit_y_data, p0=guess,
                               bounds=bounds, sigma=sigma,
                               absolute_sigma=absolute_sigma, 
                               **curve_fit_options
                               )
        perr = np.sqrt(np.diag(pcov))

        # Unpack parameters and their uncertainties
        ctr, amp, wid = popt
        ctr_err, amp_err, wid_err = perr

        # Generate Fit Curve for Plotting
        fit_x = np.linspace(x_data.min(), x_data.max(), num=1000)
        
        # For plotting, if peak_direction was 'min', we need to invert the fitted amplitude
        # back to represent the true negative peak on the original y-axis scale.
        plot_amp = amp
        if peak_direction == 'min':
            plot_amp = -amp

        fit_y = gaussian(fit_x, ctr, plot_amp, wid)

        # Plot if ax is provided
        if ax:
            if plot_label_formatter:
                label = plot_label_formatter(ctr, wid)
            else:
                label = f'Center = {ctr:0.1f} \n Width = {wid:0.1f}'
            ax.plot(fit_x, fit_y, ':', color='k', label=label)
            ax.legend()
            
        errors = {'ctr':ctr_err,
                  'wid':wid_err,
                  'amp':amp_err}
        
        return ctr, amp, wid, errors

    except Exception as e:
        print(f"Gaussian fitting failed for {y_col} (x={x_col}): {e}")
        return np.nan, np.nan, np.nan, {'ctr':np.nan,
                                        'wid':np.nan,
                                        'amp':np.nan}
# arrhenius
def arrhenius(T, A, Ea):

    return A * np.exp(Ea / (R * T))

# vft
def vft(T, A, B, Tinf):
      
      return A * np.exp(B / (T - Tinf)) 

# ln versions
def ln_arrhenius(T, A, Ea):
      
      return np.log(A) + Ea / (R * T)

def ln_vft(T, A, B, Tinf):
      
      return np.log(A) + B / (T - Tinf)

def power_law(x, A, n):

      return A * x**n

def ln_power_law(x, A, n):
      
      return np.log(A) + n * x

# fractional linear solid?
