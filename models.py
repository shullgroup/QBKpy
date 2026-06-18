# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/test/models.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# universal gas constant
R = 8.3145

# gaussian with baseline
def Gaussian(x, ctr, amp, wid, baseline=0):
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
    return baseline + amp * np.exp(-((x - ctr)**2) / (2 * wid**2))

def fitGaussian(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax=None,
    guess: list = None,
    bounds: tuple = None,
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
    guess : list, optional
        Initial guess (p0) for fitting parameters [center, amplitude, width, baseline].
        If None, an automatic guess is attempted based on peak_direction.
    bounds : tuple, optional
        Bounds for the fitting parameters: ([lower_ctr, lower_amp, lower_wid, lower_baseline],
        [upper_ctr, upper_amp, upper_wid, upper_baseline]).
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
    
    x_data = df_clean[x_col]
    y_data = df_clean[y_col]

    # 2. Handle Peak Direction for Fitting
    # For 'min' peaks, invert y-data for fitting so curve_fit sees a positive peak.
    # We will adjust the amplitude back for plotting.
    fit_y_data = y_data
    if peak_direction == 'min':
        fit_y_data = -y_data

    # 3. Automatic Guess for Parameters if not provided
    if guess is None:
        peak_idx = fit_y_data.idxmax() # Find the max of (potentially inverted) y_data
        ctr_auto_guess = x_data.loc[peak_idx]
        amp_auto_guess = fit_y_data.loc[peak_idx] # Amplitude of the (potentially inverted) peak
        
        # Heuristic for width: 1/10th of the x-range
        wid_auto_guess = (x_data.max() - x_data.min()) / 10 
        if wid_auto_guess <= 0: # Avoid division by zero or non-positive width
            wid_auto_guess = 1.0 # Fallback to a small positive width
            
        # Baseline: Average of first and last points of (potentially inverted) y_data
        baseline_auto_guess = fit_y_data.iloc[[0, -1]].mean()

        guess = [ctr_auto_guess, amp_auto_guess, wid_auto_guess, baseline_auto_guess]

    # 4. Default Bounds (very broad to be general)
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
            [x_min - x_range*2, -np.inf, min_wid, -np.inf],  # Lower bounds
            [x_max + x_range*2, np.inf, x_range * 2, np.inf]   # Upper bounds (width cap at 2x data range)
        )

    try:
        # 5. Perform the Curve Fit
        popt, pcov = curve_fit(Gaussian, x_data, fit_y_data, p0=guess,
                               bounds=bounds, sigma=sigma,
                               absolute_sigma=absolute_sigma, **kwargs)
        perr = np.sqrt(np.diag(pcov))

        # Unpack parameters and their uncertainties
        ctr, amp_fit, wid, baseline_fit = popt
        ctr_err, amp_err, wid_err, baseline_err = perr

        # 6. Generate Fit Curve for Plotting
        fit_x = np.linspace(x_data.min(), x_data.max(), num=1000)
        
        # For plotting, if peak_direction was 'min', we need to invert the fitted amplitude
        # back to represent the true negative peak on the original y-axis scale.
        plot_amp = amp_fit
        if peak_direction == 'min':
            plot_amp = -amp_fit

        fit_y = Gaussian(fit_x, ctr, plot_amp, wid, baseline_fit)

        # 7. Plot if ax is provided
        if ax:
            if plot_label_formatter:
                label = plot_label_formatter(ctr, wid)
            else:
                label = f'Center = {ctr:0.1f} \n Width = {wid:0.1f}'
            ax.plot(fit_x, fit_y, ':', color='k', label=label)
            ax.legend()
        
        # 8. Return results
        return ctr, ctr_err, wid, wid_err

    except Exception as e:
        print(f"Gaussian fitting failed for {y_col} (x={x_col}): {e}")
        return np.nan, np.nan, np.nan, np.nan

# def fitGaussian_old(df, xprop, yprop, **kwargs):
#     '''
#     Fits data to single Gaussian peak and adds to plot

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing DSC data.
#     ax : mpl.axes.Axes
#         Axes object to plot Gaussian fit on.
#     return_err : bool, default False
#         Option to return uncertainties for Tg and dT

#     Returns
#     -------
#     ctr : float
#         Center of Gaussian fit
#     ctr_err : float
#         Uncertainty in center of Gaussian fit
#     wid : float
#         Width of Gaussian fit
#     wid_err : float
#         Uncertainty in width of Gaussian fit

#     '''
    
#     ax = kwargs.get('ax', None)
#     return_err = kwargs.get('return_err', False)
#     guess = kwargs.get('guess', [100, 100, 100, 0])
#     bounds = kwargs.get('bounds', ([0,1,1,-1000],[1e5,1e5,1e4,1000]))
#     sigma = kwargs.get('sigma', None)
#     absolute_sigma = kwargs.get('absolute_sigma', False)

#     # clean up dataframe    
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df = df.dropna()
    
#     #guesses for Temp, amplitude, and width for peak
#     ctr_guess = df[yprop].max()
#     guess = kwargs.get('guess', [df.query(f'{yprop} == @ctr_guess')[xprop].iloc[0], 1000, 1000, 0])
    
#     #fit function to data and plot peak
#     popt, pcov = curve_fit(Gaussian, df[xprop], df[yprop], p0=guess,
#                            bounds=bounds, sigma=sigma, 
#                            absolute_sigma=absolute_sigma, maxfev=5000)
#     fit_x = np.linspace(df[xprop].min(),df[xprop].max(),num=1000)
#     fit_y = Gaussian(fit_x, *popt)
#     ctr = popt[0]
#     wid = popt[2]
    
#     # parameter uncertainties
#     perr = np.sqrt(np.diag(pcov))
#     ctr_err = perr[0]
#     wid_err = perr[2]

#     # add to a plot if axis given
#     if ax:
#         ax.plot(fit_x, fit_y, ':', color='k',
#                 label=f'Center = {ctr:0.0f} \n Width = {wid:0.0f}')
#         ax.legend()
    
#     return ctr, ctr_err, wid, wid_err

# arrhenius
def Arrhenius(T, A, Ea):

    return A * np.exp(Ea / (R * T))

# vft
def VFT(T, A, B, Tinf):
      
      return A * np.exp(B / (T - Tinf)) 

# ln versions
def ln_Arrhenius(T, A, Ea):
      
      return np.log(A) + Ea / (R * T)

def ln_VFT(T, A, B, Tinf):
      
      return np.log(A) + B / (T - Tinf)

def PowerLaw(x, A, n):

      return A * x**n

def ln_PowerLaw(x, A, n):
      
      return np.log(A) + n * x

# fractional linear solid?
