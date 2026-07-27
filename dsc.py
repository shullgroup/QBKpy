# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/dsc.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .utils import (read_data_file, baseline_correct)
from .models import fit_gaussian

labels = {'temp': r'T ($^\circ$C)',
          'time': r't (min.)',
          'q': r'q (Watts$\cdot$g$^{-1}$)',
          'q_r': r'$q/r$ (J$\cdot$g$^{-1}\cdot$K$^{-1}$)',
          'dqdT': '$dq/dT$ (Watts$\cdot$K$^{-1}$g$^{-1}$)'}

def read_dsc(path, mode='conv', apply_savgol=True, savgol_window=151,
             savgol_polyorder=4, n_crit=50, **kwargs):
    """
    Read txt file from DSC experiment and convert to a DataFrame.
    Supports both conventional ('conv') and modulated ('mdsc') modes.

    Parameters
    ----------
    path : Path
        Path object to the .txt file containing the DSC data.
    mode : str, default 'conv'
        'conv' for conventional DSC, 'mdsc' for modulated DSC.
    apply_savgol : bool, default True
        Apply Savitzky–Golay smoothing to temp and power.
    savgol_window : int, default 151
        Window length for Savitzky–Golay filter (must be odd).
    savgol_polyorder : int, default 4
        Polynomial order for Savitzky–Golay filter.
    n_crit : int, default 50
        Number of consecutive slope points used for segmentation.
    columns : list of integers
        Columns to read - defaults determined by DSC type
    **kwargs : dict
        Optional keyword arguments such as:
        - sep : delimiter for input file
        - time_to_sec : convert minutes → seconds


    Returns
    -------
    df : pd.DataFrame
        Cleaned and optionally smoothed DSC data.
    """

    sep = kwargs.get('sep', '\t')

    # Determine columns based on mode
    if mode == 'mdsc':
        target_cols, names = [0, 1, 2, 3, 7], ['time', 'temp', 'q_rev', 'q_non', 'dq_revdT']
    else:
        target_cols, names = [0, 1, 2], ['time', 'temp_in', 'q_in']
        
    
    target_cols = kwargs.get('columns', target_cols)

    df = read_data_file(path, sep=sep,
                        target_cols=target_cols,
                        names=names)

    # Convert time if requested
    if kwargs.get('time_to_sec', False):
        df['time'] = df['time'] * 60

    # Conventional DSC derivative
    if mode == 'conv':
        df = df.dropna(subset=['temp_in', 'q_in'])

    # Apply Savitzky–Golay smoothing and insert columns in correct positions
    if mode == 'conv' and apply_savgol:
        # smoothed q and temp
        q_smoothed = savgol_filter(df['q_in'], savgol_window, savgol_polyorder)
        temp_smoothed = savgol_filter(df['temp_in'], savgol_window, savgol_polyorder)
        
        # Insert smoothed q right after 'q_in'
        q_pos = df.columns.get_loc('q_in') + 1
        df.insert(q_pos, 'q', q_smoothed)
        
        # Insert smoothed temp right after 'temp_in'
        temp_pos = df.columns.get_loc('temp_in') + 1
        df.insert(temp_pos, 'temp', temp_smoothed)
        
        # Compute dQ/dT safely
        dt = np.gradient(temp_smoothed)
        dq = np.gradient(q_smoothed)
        
        dqdT = np.full_like(q_smoothed, np.nan, dtype=float)
        
        valid = np.abs(dt) > 1e-12  # adjust tolerance if needed
        dqdT[valid] = dq[valid] / dt[valid]
        
        # Insert dqdT right after the smoothed q column
        dqdT_pos = df.columns.get_loc('q') + 1
        df.insert(dqdT_pos, 'dqdT', dqdT)

    return df


def read_segmented_dsc(
    path,
    mode='conv',
    apply_savgol=True,
    savgol_window=151,
    savgol_polyorder=4,
    frac_thresh=0.05,
    n_crit=50,
    **kwargs
):
    """
    Read a multi‑sheet XLS file where each sheet contains DSC data.
    For each sheet:
        1. Run read_dsc‑like preprocessing
        2. Run find_monotonic_segments
    Returns a dict keyed by sheet name:
        {
            sheet_name: {
                'df': cleaned dataframe,
                'segments': segment dictionary
            }
        }
    """


    dfs = read_data_file(
        path,
        sheet_name=None,
        usecols=[0, 1, 2],
        names=['time', 'temp_in', 'q_in']
    )
    
    df = pd.concat(dfs.values(), ignore_index=True)


    return df



def plot_dsc(df_in, ax, xdata, ydata, **kwargs):
    '''
    Generate typical plots for DSC experiments. Emphasis primarily placed
    on finding Tg as opposed to other transitions for now.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing experimental data read in from the readDSC function
    ax : mpl.axes.Axes, default None
            Axes for the heat flow if one already exists.
    mode : str, default 'conv'
        DSC mode used for experiment. Options are 'conv' for conventional DSC
        or 'mdsc' for temperature modulated DSC.

    baseline : list of two numbers
        x vlues to use for baseline correction
    label : string, default ''
        legend label, default '' gives no label
    deriv_plot : bool, deault False
        option to put the dQ/dT on a twinned right axis
    showTg : bool, default True
        Option to show Tg fit from fit_gaussian.
    orientation : str, default 'exo_up'
        Orientation for heat flow annotation ('exo_up' or 'endo_up').
    fmt : str, default '-'
        Format string
    linewidth : float, default None
        Linewidth for plot

    Returns
    -------
    Tg : float
        Glass transition temperature (Tg) in deg. Celsius.

    '''
    mode = kwargs.get('mode', 'conv')  # 'conv' is assumed now'
    fmt = kwargs.get('fmt', '-')
    linewidth = kwargs.get('linewidth', 1)
    label = kwargs.get('label', '')

    orientation = kwargs.get('orientation', 'exo_up')
    df = df_in.copy()

    # apply baseline correction if needed
    if 'baseline' in kwargs.keys():
        df = baseline_correct(df, xdata, ydata, kwargs.get('baseline'))
    
    # clean up dataframe
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['temp', 'q'])
    
    # optional temperature filtering
    T_range = kwargs.get('T_range')
    if T_range is not None:
        df = df.loc[df['temp'].between(*T_range)]
    
    
    # ceate axis lables
    ax.set_xlabel(labels[xdata])
    ax.set_ylabel(labels[ydata])
    
    # ax.set_prop_cycle(default_cycler)
    ax.plot(df[xdata], df[ydata], fmt, linewidth = linewidth,
            label = label)

           
    # add annotation for the heat flow orientation
    if ydata == 'q':
        if orientation == 'exo_up':
            ax.annotate('Exo Up', (5,5), xycoords='axes points')
        elif orientation == 'endo_up':
            ax.annotate('Endo Up', (5,5), xycoords='axes points')
    
    return df


def fit_gaussian_dsc(df: pd.DataFrame, ax, **kwargs):
    """
    Fits DSC derivative data to a single Gaussian peak (for Glass Transition).

    This function utilizes the general `fitGaussian_general` to perform the fitting,
    providing DSC-specific column names, default bounds, and ensuring a 'min' peak direction.
    The fitting function is baseline + amp * np.exp(-((x - ctr)**2) / (2 * wid**2))

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DSC data.
    ax : mpl.axes.Axes
        Axes object to plot Gaussian fit on.
    x_col : string, default 'temp'
        x column for data
    y_col : string, default 'dqdT'
        y column for data
    peak_direction : string (either 'max' or 'min')
        direction of peak
    baseline : list or tuple of two flots
        Temperatures to use for baseline subtraction (default None)
    T_range : list or tuple of two floats 
        Temperature range for fitting the Gaussian (default baseline)

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
    dsc_x_col = kwargs.pop('x_col', 'temp')
    dsc_y_col = kwargs.pop('y_col', 'dqdT') # 'dqdT' is the default target derivative column
    peak_direction = kwargs.pop('peak_direction', 'min')
    baseline = kwargs.pop('baseline', None)
    
    # DSC-specific default bounds (if not provided in kwargs)
    dsc_bounds = kwargs.pop('bounds', ([-100, 0, 0], [200, 1, 30]))
    T_range = kwargs.pop('T_range', baseline)
    
    # Custom plot label formatter for DSC data
    def dsc_label_formatter(ctr, wid):
        return f'T$_g = {ctr:0.1f} ^\\circ$C \n $\u03b4T = {wid:0.1f} ^\\circ$C'

    # Call the general fitGaussian function
    ctr, ctr_err, wid, wid_err = fit_gaussian(
        df,
        x_col=dsc_x_col,
        y_col=dsc_y_col,
        ax=ax,
        baseline = baseline,
        x_range = T_range,
        bounds=dsc_bounds,
        peak_direction=peak_direction, 
        plot_label_formatter=dsc_label_formatter,
        **kwargs # Pass remaining kwargs like guess, sigma, maxfev etc.
    )

    # Alias return values for DSC-specific nomenclature
    Tg, Tg_err, dT, dT_err = ctr, ctr_err, wid, wid_err
    return Tg, Tg_err, dT, dT_err


def find_monotonic_segments(df, frac_thresh=0.05, n_crit=50, ax=None):
    """
    Identify increasing/decreasing segments lasting at least n_crit rows.
    Adds a 'segment' column to df indicating segment membership.
    Constant portions receive segment = -1.

    Returns a dictionary keyed by segment number:
        - type ('increasing' or 'decreasing')
        - s (start index)
        - e (end index)
        - ramp_rate
        - idxvals (array of indices in the segment)
    """

    slope = np.gradient(df['temp'].values, df['time'].values)

    # dynamic threshold
    max_slope = np.max(np.abs(slope))
    threshold = frac_thresh * max_slope

    # classify slope
    def classify(s):
        if abs(s) <= threshold:
            return 'constant'
        return 'increasing' if s > 0 else 'decreasing'

    labels = np.array([classify(s) for s in slope])

    # enforce minimum run length
    final_labels = labels.copy()
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            run_label = labels[start]
            run_len = i - start
            if run_label in ('increasing', 'decreasing') and run_len < n_crit:
                final_labels[start:i] = 'constant'
            start = i

    # --- Extract monotonic segments ---
    segments = {}
    seg_num = 0
    start = 0

    # initialize segment column
    df['segment'] = -1

    for i in range(1, len(final_labels) + 1):
        if i == len(final_labels) or final_labels[i] != final_labels[start]:
            label = final_labels[start]

            if label in ('increasing', 'decreasing'):
                s = int(start)
                e = i - 1

                segments[seg_num] = {
                    'type': label,
                    's': s,
                    'e': e,
                    'ramp_rate': (
                        (df.temp.iloc[e] - df.temp.iloc[s]) /
                        (df.time.iloc[e] - df.time.iloc[s])
                    ),
                    'idxvals': np.r_[s:e]
                }

                # assign segment number to df
                df.loc[s:e, 'segment'] = seg_num
                segments[seg_num]['df'] = df[df['segment'] == seg_num].copy()
                seg_num += 1

            start = i

    # --- Optional plotting ---
    if ax is not None and len(segments) > 0:
        plot_dsc(df, ax, 'time', 'temp', fmt='--', linewidth=0.5)
        for seg in segments.keys():
            s = segments[seg]['s']
            e = segments[seg]['e']
            plot_dsc(df.iloc[s:e+1], ax, 'time', 'temp', fmt=f'C{seg}')
        ax.legend()

    return segments


def remove_extreme_temps(df, n):
    """
    Remove rows where 'temp' is within n degrees of the min or max temp.
    """
    tmin = df['temp'].min()
    tmax = df['temp'].max()

    mask = (df['temp'] < tmin + n) | (df['temp'] > tmax - n)
    return df.loc[~mask].copy()

