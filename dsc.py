# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/dsc.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from utils import read_data_file, default_cycler
from graphics import double_headed_arrow, vline
from models import fit_gaussian as _fit_gauss_gen

def read_DSC(path, **kwargs):
    '''
    Read txt file from DSC experiment and convert to a DataFrame.
    Supports both conventional ('conv') and modulated ('mdsc') modes.
    
    Parameters
    ----------
    path : Path
        Path object to the .txt file containing the DSC data
    mode : str, default 'conv'
        Flag for mode of DSC, i.e. conventional (conv) vs. modulated (mdsc)
    apply_savgol : bool, default True
        Option to apply the Savitsky-Golay filter, especially useful
        for the derivatives.
    savgol_window : int, default 151
        Window length for filtering derivative data. Must be odd int.
    savgol_polyorder : int, default 4
        Polynomial degree for filtering derivative data. Must be 
        less than window length.
    sep : str, default '\t'
        Delimiter for the data file.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing relevant experimental data. 
        For 'conv': [time, temp, q, dqdT]
        For 'mdsc': [time, temp, q_rev, q_non, dq_revdT]
    '''
    mode = kwargs.get('mode', 'conv')
    apply_savgol = kwargs.get('apply_savgol', True)
    savgol_window = kwargs.get('savgol_window', 151)
    savgol_polyorder = kwargs.get('savgol_polyorder', 4)
    

    sep = kwargs.get('sep', '\t')
    # Determine columns based on mode
    if mode == 'mdsc':
        target_cols, names = [0, 1, 2, 3, 7], ['time', 'temp', 'q_rev', 'q_non', 'dq_revdT']
    else:
        target_cols, names = [0, 1, 2], ['time', 'temp', 'q']

    df = read_data_file(path, sep=sep,
                      target_cols=target_cols,
                      names=names)

    # Standardize Time (minutes to seconds)
    if kwargs.get('time_to_sec', True):
        df['time'] = df['time'] * 60

    # Handle Derivatives for conventional mode
    if mode == 'conv':
        try:
            df['dqdT'] = np.gradient(df['q'], df['temp'])
        except:
            df['dqdT'] = np.nan
        

    # apply Savitsky-Golay filter to smooth data
    # because normally very coarse/jumpy directly from instrument
    if apply_savgol:
        # use reversing heat flow if mDSC
        q_to_smooth = 'q_rev' if mode == 'mdsc' else 'q'
        dq_to_smooth = 'dq_revdT' if mode == 'mdsc' else 'dqdT'
        
        df[q_to_smooth] = savgol_filter(df[q_to_smooth], savgol_window, savgol_polyorder)
        df[dq_to_smooth] = savgol_filter(df[dq_to_smooth], savgol_window, savgol_polyorder)

    return df

def plot_DSC(df, **kwargs):
    '''
    Generate typical plots for DSC experiments. Emphasis primarily placed
    on finding Tg as opposed to other transitions for now.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data read in from the readDSC function
    mode : str, default 'conv'
        DSC mode used for experiment. Options are 'conv' for conventional DSC
        or 'mdsc' for temperature modulated DSC.
    ax : mpl.axes.Axes, default None
        Axes for the heat flow if one already exists.
    twin : mpl.axes.Axes, default None
        Twin axis for the derivative data if one already exists.
    title : str, default None
        Title for the plot.
    savepath : Path, default None
        Path for saving the plot.
    min_temp : float, default None
        Minimum temperature for the plot range.
    max_temp : float, default None
        Maximum temperature for the plot range.
    showTg : bool, default True
        Option to show Tg fit from fitGaussianDSC.
    no_legend : bool, default False
        Option to remove legend.
    legendloc : int or str, default 0
        Location of legend.
    legendsize : float, default 10.
        Size of legend.
    show_deriv_ticks : bool, default False
        Option to show the tick marks on the twin axis.
    orientation : str, default 'exo_up'
        Orientation for heat flow annotation ('exo_up' or 'endo_up').

    Returns
    -------
    Tg : float
        Glass transition temperature (Tg) in deg. Celsius.
    ax : mpl.axes.Axes
        Axes instance used for plotting the heat flow.
    twin : mpl.axes.Axes
        Axes instance used for plotting the derivative heat flow.
    '''
    mode = kwargs.get('mode', 'conv')
    ax = kwargs.get('ax', None)
    twin = kwargs.get('twin', None)
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)
    min_temp = kwargs.get('min_temp', df['temp'].min())
    max_temp = kwargs.get('max_temp', df['temp'].max())
    no_legend = kwargs.get('no_legend', False)
    legendloc = kwargs.get('legendloc', 0)
    legendsize = kwargs.get('legendsize', 10.)
    show_deriv_ticks = kwargs.get('show_deriv_ticks', False)
    orientation = kwargs.get('orientation', 'exo_up')

    # clean up the dataframe
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.query('temp >= @min_temp and temp <= @max_temp')
    
    # Identify columns based on mode
    q_col = 'q_rev' if mode == 'mdsc' else 'q'
    dq_col = 'dq_revdT' if mode == 'mdsc' else 'dqdT'

    # if not given an axis to plot on, make one
    if not ax:
        fig, ax = plt.subplots(figsize=(4,3), constrained_layout=True)
        ax.set_xlabel('Temperature ($^\\circ$C)')
        ax.set_ylabel('Norm. Heat Flow (W/g)')
        ax.set_title(title)
    
    # Apply your custom cycler to the specific axes
    ax.set_prop_cycle(default_cycler)

    # if not given a twin axis for the derivative, make one
    if not twin:
        twin = ax.twinx()
        twin.set_ylabel('Deriv. Heat Flow (W/g*$^{\\circ}$C)')
        twin.set_prop_cycle(default_cycler)

    # plot heat flow and derivative w.r.t. temperature
    ax.plot(df['temp'], df[q_col], '-', label='Heat Flow')
    twin.plot(df['temp'], -df[dq_col], '--', label='dq/dT')
    
    # Fitting
    Tg = None
    if kwargs.get('showTg', True):
        # We pass the column name to fit function so it knows what to fit
        Tg, Tg_err, dT, dT_err = fit_gaussian(df, twin, target_dq=dq_col, **kwargs)
    
    # add everything to the same legend unless you don't want a legend
    if not no_legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles=h1+h2, labels=l1+l2, loc=legendloc, prop={'size':legendsize})
        
    # add annotation for the heat flow orientation
    if orientation == 'exo_up':
        ax.annotate('Exo Up', (5,5), xycoords='axes points')
    elif orientation == 'endo_up':
        ax.annotate('Endo Up', (5,5), xycoords='axes points')

    # usually don't show derivative y ticks because too crowded
    if not show_deriv_ticks:   
        twin.set_yticks([])

    # save plot if given path
    if savepath:
        plt.savefig(savepath)

    return Tg, ax, twin


def fit_gaussian(df: pd.DataFrame, ax, **kwargs):
    """
    Fits DSC derivative data to a single Gaussian peak (for Glass Transition).

    This function utilizes the general `fitGaussian_general` to perform the fitting,
    providing DSC-specific column names, default bounds, and ensuring a 'min' peak direction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DSC data.
    ax : mpl.axes.Axes
        Axes object to plot Gaussian fit on.
    target_dq : str, default 'dqdT'
        The column name of the derivative data to fit.
    **kwargs:
        Additional keyword arguments passed to `fitGaussian_general`.
        These can include `guess`, `bounds` (to override defaults), `sigma`,
        `absolute_sigma`, `maxfev`, etc.

    Returns
    -------
    Tg : float
        Glass transition temperature from center of Gaussian fit in deg. C.
        Returns np.nan if fit fails.
    Tg_err : float
        Uncertainty in Tg fit in deg. C. Returns np.nan if fit fails.
    dT : float
        Breadth of glass transition from width of Gaussian fit in deg. C.
        Returns np.nan if fit fails.
    dT_err : float
        Uncertainty in dT fit in deg. C. Returns np.nan if fit fails.
    """
    # DSC-specific column names and parameters
    dsc_x_col = 'temp' 
    dsc_y_col = kwargs.pop('target_dq', 'dqdT') # 'dqdT' is the default target derivative column
    
    # DSC-specific default bounds (if not provided in kwargs)
    dsc_bounds = kwargs.pop('bounds', ([-100, 0, 0, -1], [200, 1, 30, 1]))
    
    # Custom plot label formatter for DSC data
    def dsc_label_formatter(ctr, wid):
        return f'T$_g = {ctr:0.1f} ^\\circ$C \n $\u03b4T = {wid:0.1f} ^\\circ$C'

    # Call the general fitGaussian function
    ctr, ctr_err, wid, wid_err = _fit_gauss_gen(
        df,
        x_col=dsc_x_col,
        y_col=dsc_y_col,
        ax=ax,
        bounds=dsc_bounds,
        peak_direction='min', # DSC derivative peaks are typically negative
        plot_label_formatter=dsc_label_formatter,
        **kwargs # Pass remaining kwargs like guess, sigma, maxfev etc.
    )

    # Alias return values for DSC-specific nomenclature
    Tg, Tg_err, dT, dT_err = ctr, ctr_err, wid, wid_err
    return Tg, Tg_err, dT, dT_err

# def fitGaussian_old(df, ax, **kwargs):
#     # CHANGE THIS TO UTILIZE GENERALIZED fitGaussian FUNCTION IN MODELS
#     '''
#     Fits data to single Gaussian peak and adds to DSC plot

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing DSC data.
#     ax : mpl.axes.Axes
#         Axes object to plot Gaussian fit on. Typically twin from DSC plot.
#     target_dq : str, default 'dqdT'
#         The column name of the derivative data to fit.
#     bounds : tuple, default ([-100,0,0,-1],[200,1,30,1])
#         Bounds for the fitting parameters.
#     guess : list, default Calculated based on peak min
#         Initial guess (p0) for fitting.

#     Returns
#     -------
#     Tg : float
#         Glass transition temperature from center of Gaussian fit in deg. C
#     Tg_err : float
#         Uncertainty in Tg fit in deg. C
#     dT : float
#         Breadth of glass transition from width of Gaussian fit in deg. C
#     dT_err : float
#         Uncertainty in dT fit in deg. C
#     '''
#     target_dq = kwargs.get('target_dq', 'dqdT')
#     bounds = kwargs.get('bounds', ([-100,0,0,-1],[200,1,30,1]))

#     # gaussian with baseline y0
#     def Gaussian(x, ctr, amp, wid, y0):
#         return y0 + amp * np.exp(-((x - ctr) / (2 * wid)) ** 2)

#     # clean up dataframe a bit
#     df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
#     # find peak in derivative (min because negative usually)
#     peak_idx = df[target_dq].idxmin()
#     guess = kwargs.get('guess', [df.loc[peak_idx, 'temp'], 3e-2, 10, 0])
    
#     try:
#         popt, pcov = curve_fit(Gaussian, df['temp'], -df[target_dq], p0=guess, bounds=bounds)
#         perr = np.sqrt(np.diag(pcov))
        
#         fit_temp = np.linspace(df['temp'].min(), df['temp'].max(), 1000)
#         fit_curve = Gaussian(fit_temp, *popt)
        
#         # plot the fit
#         ax.plot(fit_temp, fit_curve, ':', color='k',
#                 label=f'T$_g = {popt[0]:0.1f} ^\\circ$C \n $\u03b4T = {popt[2]:0.1f} ^\\circ$C')
        
#         # return the peak center and error and the peak width and error
#         return popt[0], perr[0], popt[2], perr[2]
#     except Exception:
#         return np.nan, np.nan, np.nan, np.nan
