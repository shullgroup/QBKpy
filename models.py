# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/models.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .utils import baseline_correct

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
    **kwargs
):
    """
    Fits data to a single Gaussian peak using the external Gaussian function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    x_col : str
        Column name for the x-axis data (independent variable).
    y_col : str
        Column name for the y-axis data (dependent variable).
    ax : mpl.axes.Axes, optional
        Axes object to plot Gaussian fit on. If None, no plot is generated.
    baseline : list or tuple two floats
        X axis values used to establish baseline
    x_range : list or tuple of two floats
        X axis range for which fit will be attempted defaults to baseline
    guess : list, optional
        Initial guess (p0) for fitting parameters [center, amplitude, width, baseline].
        If None, an automatic guess is attempted based on peak_direction.
    bounds : tuple, optional
        Bounds for the fitting parameters: ([lower_ctr, lower_amp, lower_wid],
        [upper_ctr, upper_amp, upper_wid]).
        If None, very broad default bounds are used.
    sigma : array_like or None, optional
        Determines the uncertainty in ydata.
    absolute_sigma : bool, optional
        If True, sigma is used in an absolute sense and the estimated parameter
        covariance pcov reflects these absolute uncertainties.
    peak_direction : {'max', 'min'}, default 'max'
        Direction of the peak for automatic guessing and potential y-axis inversion.
        'max' for a peak pointing up, 'min' for a peak pointing down.
    plot_label_formatter : callable, optional
        A function that takes (ctr, wid) and returns a string for the plot label.
        If None, a default generic label will be used.
    **kwargs:
        Any additional keyword arguments are passed directly to `scipy.optimize.curve_fit`.
        (e.g., `maxfev=5000`)

    Returns
    -------
    ctr : float
        Center of Gaussian fit. Returns np.nan if fit fails.
    ctr_err : float
        Uncertainty in center of Gaussian fit. Returns np.nan if fit fails.
    wid : float
        Width of Gaussian fit. Returns np.nan if fit fails.
    wid_err : float
        Uncertainty in width of Gaussian fit. Returns np.nan if fit fails.
    """

    # 1. Data Cleaning
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col]).copy()
    
    if df_clean.empty:
        print("Warning: DataFrame is empty after cleaning. Cannot fit.")
        return np.nan, np.nan, np.nan, np.nan
    
    df_clean = baseline_correct(df_clean, x_col, y_col, x_range=baseline)
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
        y_min, y_max = y_data.min(), y_data.max() # Use original y_data for bounds reasoning
        x_range = x_max - x_min
        y_range = y_max - y_min

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
        popt, pcov = curve_fit(gaussian, x_data, fit_y_data, p0=guess,
                               bounds=bounds, sigma=sigma,
                               absolute_sigma=absolute_sigma, **kwargs)
        perr = np.sqrt(np.diag(pcov))

        # Unpack parameters and their uncertainties
        ctr, amp_fit, wid = popt
        ctr_err, amp_err, wid_err = perr

        # Generate Fit Curve for Plotting
        fit_x = np.linspace(x_data.min(), x_data.max(), num=1000)
        
        # For plotting, if peak_direction was 'min', we need to invert the fitted amplitude
        # back to represent the true negative peak on the original y-axis scale.
        plot_amp = amp_fit
        if peak_direction == 'min':
            plot_amp = -amp_fit

        fit_y = gaussian(fit_x, ctr, plot_amp, wid)

        # Plot if ax is provided
        if ax:
            if plot_label_formatter:
                label = plot_label_formatter(ctr, wid)
            else:
                label = f'Center = {ctr:0.1f} \n Width = {wid:0.1f}'
            ax.plot(fit_x, fit_y, ':', color='k', label=label)
            ax.legend()
        
        return ctr, ctr_err, wid, wid_err

    except Exception as e:
        print(f"Gaussian fitting failed for {y_col} (x={x_col}): {e}")
        return np.nan, np.nan, np.nan, np.nan


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
