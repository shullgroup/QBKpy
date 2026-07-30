# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/dsc.py
import numpy as np
import pandas as pd
import utils
from models import fit_gaussian

labels = {'temp': r'T ($^\circ$C)',
          'temp_in': r'T ($^\circ$C)',
          'time': r't (min.)',
          'time_in': r't (min.)',
          'q': r'q (Watts$\cdot$g$^{-1}$)'+'\n'+r'exo $\longrightarrow$',
          'q_in': r'q (Watts$\cdot$g$^{-1}$)'+'\n'+r'exo $\longrightarrow$',
          'q_r': r'$q/r$ (J$\cdot$g$^{-1}\cdot$K$^{-1}$)',
          'dqdT': ('$dq/dT$ (Watts$\cdot$K$^{-1}$g$^{-1}$)'+
                   '\n'+r'exo $\longrightarrow$')}


def read_dsc(path, mode='conv', **kwargs):
    """
    Read Differential Scanning Calorimetry (DSC) data from a text,
    CSV, or spreadsheet file and return the results as a pandas
    DataFrame.

    The function supports both conventional DSC ("conv") and
    modulated DSC ("mdsc") data formats. Input files are read using
    `utils.read_data_file()`, selected columns are renamed to
    standardized variable names, and optional preprocessing steps are
    applied.
 
    For conventional DSC data, the raw signals are smoothed using
    `utils.spline_smooth()`, after which first derivatives with
    respect to time and temperature are calculated.
 
    Parameters
    ----------
    path : str or pathlib.Path
        Path to the DSC data file.
 
    mode : {'conv', 'mdsc'}, optional
        Type of DSC data to process.
 
        - 'conv' : Conventional DSC data containing time,
          temperature, and heat-flow measurements.
        - 'mdsc' : Modulated DSC data containing reversing and
          non-reversing heat-flow components.
 
        Default is 'conv'.
 
    Other Parameters
    ----------------
    smooth_factor : float, optional
        Smoothing factor passed to `utils.spline_smooth()` when
        processing conventional DSC data. Larger values produce
        smoother signals. Default is 1.
 
    sep : str, optional
        Column delimiter used when reading text files.
        Default is '\\t'.
 
    sheet_name : int, str, list, or None, optional
        Worksheet(s) to read when the input file is an Excel workbook.
        Passed directly to `utils.read_data_file()`.
        Default is [0].
 
    columns : sequence of int, optional
        Column indices to extract from the source file. If omitted,
        mode-specific defaults are used.
 
    time_to_sec : bool, optional
        If True, converts the input time values from minutes to
        seconds. Default is False.
 
    Returns
    -------
    pandas.DataFrame
        Processed DSC data.
 
        For conventional DSC mode, the returned DataFrame contains:
 
        - time_in : raw time values
        - temp_in : raw temperature values
        - q_in : raw heat-flow values
        - time : smoothed time values
        - temp : smoothed temperature values
        - q : smoothed heat-flow values
        - dTdt : heating/cooling rate (dT/dt)
        - dqdt : heat-flow rate (dq/dt)
        - dqdT : heat-flow derivative with respect to temperature
          (dq/dT)
 
        For modulated DSC mode, the returned DataFrame contains the
        selected and renamed input columns without derivative
        calculations.
 
    Notes
    -----
    Rows containing non-numeric values in required conventional DSC
    columns are automatically removed before smoothing and derivative
    calculations.
 
    Derivatives are computed using `numpy.gradient()`, which provides
    numerically stable central-difference estimates for uniformly
    sampled data.
    """

    smooth_factor = kwargs.get('smooth_factor', 1)
    sep = kwargs.get('sep', '\t')
    sheet_name = kwargs.get('sheet_name', [0])

    # Determine columns based on mode
    if mode == 'mdsc':
        target_cols, names = [0, 1, 2, 3, 7], ['time_in', 'temp', 'q_rev', 'q_non', 'dq_revdT']
    else:
        target_cols, names = [0, 1, 2], ['time_in', 'temp_in', 'q_in']
        
    
    target_cols = kwargs.get('columns', target_cols)

    df = utils.read_data_file(path, sep=sep,
                        target_cols=target_cols,
                        names=names,
                        sheet_name = sheet_name)

    # Convert time if requested
    if kwargs.get('time_to_sec', False):
        df['time_in'] = df['time_in'] * 60

    # Conventional DSC derivative
    if mode == 'conv':
        cols = ['time_in', 'temp_in', 'q_in']
        
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=cols)


    if mode == 'conv':
        for var in ['time', 'temp', 'q']:
            var_smooth = utils.savgol_interpolate(df[f'{var}_in'])
            df.insert(df.columns.get_loc(f'{var}_in')+1, var, var_smooth)
       
    return df


def read_segmented_dsc(path,
    apply_savgol=True,
    savgol_window=151,
    savgol_polyorder=4,
    end_temp_width = 10
    ):

    df_dict = pd.read_excel(path, sheet_name = None, header=None, 
                            names = ['time', 'temp_in', 'q_in'],
                            skiprows=3)
    segments = {}
    k=0
    for i in np.arange(len(df_dict.keys())):
        segments[k] = {}
        segments[k]['name'] = list(df_dict.keys())[i]
        df = df_dict[segments[k]['name']]
        df = df.dropna(subset=['time', 'temp_in', 'q_in'])
        if len(df) == 0:
            continue
        df['temp'] = df['temp_in']
        df['q'] = df['q_in']
        segments[k]['rate'] = ((df.iloc[-1]['temp']-df.iloc[0]['temp'])/
                               (df.iloc[-1]['time']-df.iloc[0]['time']))
        if abs(segments[k]['rate'])>0.9:
            df = remove_extreme_temps(df, end_temp_width)
        if apply_savgol:
            df['q'] = utils.savgol_smooth(df['q'], 
                                    window_length=savgol_window,
                                    polyorder=savgol_polyorder)
            df['temp'] = utils.savgol_smooth(df['temp'], 
                                    window_length=savgol_window,
                                    polyorder=savgol_polyorder)
        
        dq = np.gradient(df['q'])
        dT = np.gradient(df['temp'])
        
        df['dqdT'] = np.divide(dq, dT,
            out=np.full_like(dq, np.nan, dtype=float),
            where=dT != 0)
        
        segments[k]['df'] = df
        k += 1
    return segments


def plot_dsc(df_in, ax, xdata, ydata, 
             baseline = None,
             T_range = None,
             showTg = False,
             fmt = '-',
             **plot_options):
    '''
    Generate typical plots for DSC experiments. Emphasis primarily placed
    on finding Tg as opposed to other transitions for now.
    Data assumed to be 'Exo Up' - reverse the sign before plotting if this is
    not the case

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing experimental data read in from the readDSC function
    ax : mpl.axes.Axes, default None
            Axes for the heat flow if one already exists.
    baseline : list of two floats, default None
        x vlues to use for baseline correction, ignored if equal to []
    T_range : list of two floats, default None
        x range to plot, not used by default
    showTg : bool, default True
        Option to show Tg fit from fit_gaussian.
    fmt : str, default '-'
        Format string
    plot_options : dictionary of plot options passed directly to the 
        matplotlib plot function

    
    Returns
    -------
    Tg : float
        Glass transition temperature (Tg) in deg. Celsius.

    '''

    df = df_in.copy()

    # apply baseline correction if needed
    if baseline != None and len(baseline) == 2:
        df = utils.baseline_correct(df, xdata, ydata, baseline)
    
    # clean up dataframe
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['temp', 'q'])
    
    # optional temperature filtering
    if T_range != None and len(T_range) == 2:
        df = df.loc[df['temp'].between(*T_range)]
    
    # ceate axis lables
    ax.set_xlabel(labels[xdata])
    ax.set_ylabel(labels[ydata])
    
    # ax.set_prop_cycle(default_cycler)
    ax.plot(df[xdata], df[ydata], fmt, **plot_options)
           

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


def find_monotonic_segments(df_in,
                            frac_thresh=0.05,
                            n_crit=50,
                            ax=None,
                            **kwargs):
    """
    Identify monotonic temperature-program segments from piecewise
    linear DSC data.

    Unlike derivative-based approaches, this routine estimates ramp
    rates from large blocks of data using endpoint slopes. This makes
    the segmentation highly resistant to quantization artifacts,
    limited numerical precision, and other small fluctuations that
    often produce spurious dT/dt values.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame containing at least:

            - time
            - temp

    frac_thresh : float, optional
        Fraction of the largest absolute ramp rate used to distinguish
        increasing/decreasing segments from approximately constant
        regions.

        Default is 0.05.

    n_crit : int, optional
        Minimum number of rows required for a segment to be retained.

        Default is 50.

    ax : matplotlib.axes.Axes, optional
        Axes on which identified segments are plotted.

    Other Parameters
    ----------------
    block_size : int, optional
        Number of points used to estimate each local ramp rate.

        Larger values produce more robust segmentation.

        Typical values:

            100   -> sensitive
            500   -> recommended
            1000  -> very robust

        Default is 500.

    fmt : str, optional
        Plot format string.

        Default is '-'.

    Returns
    -------
    dict
        Dictionary of segments keyed by segment number.

        Each segment contains:

            - type
            - s
            - e
            - ramp_rate
            - idxvals
            - df

    Notes
    -----
    Ramp rates are estimated as

        (T_end - T_start)/(t_end - t_start)

    on coarse blocks rather than from numerical derivatives. This is
    generally much more reliable for DSC temperature programs that
    consist of a small number of long linear ramps.
    """

    df = df_in.copy()

    time = df['time'].to_numpy()
    temp = df['temp'].to_numpy()

    fmt = kwargs.get('fmt', '-')
    block_size = kwargs.get('block_size', 500)

    n = len(df)

    # segment assignment for every row
    labels = np.full(n, 'constant', dtype=object)

    # ---- determine block slopes ----

    block_slopes = []
    block_ranges = []

    for s in range(0, n, block_size):

        e = min(s + block_size, n)

        if e - s < 2:
            continue

        slope = (
            temp[e - 1] - temp[s]
        ) / (
            time[e - 1] - time[s]
        )

        block_slopes.append(slope)
        block_ranges.append((s, e))

    if len(block_slopes) == 0:
        return {}

    max_slope = np.max(np.abs(block_slopes))
    threshold = frac_thresh * max_slope

    def classify(slope):

        if abs(slope) <= threshold:
            return 'constant'

        return 'increasing' if slope > 0 else 'decreasing'

    # assign label to each block
    for slope, (s, e) in zip(block_slopes, block_ranges):

        labels[s:e] = classify(slope)

    # ---- merge contiguous regions ----

    df['segment'] = -1

    segments = {}
    seg_num = 0

    start = 0

    for i in range(1, n + 1):
    
        end_of_run = (
            i == n or
            labels[i] != labels[start]
        )
    
        if end_of_run:
    
            label = labels[start]
    
            if label != 'constant':
    
                seg_len = i - start
    
                if seg_len >= n_crit:
    
                    s = start
                    e = i - 1
    
                    ramp_rate = (
                        temp[e] - temp[s]
                    ) / (
                        time[e] - time[s]
                    )
    
                    df.loc[s:e, 'segment'] = seg_num
    
                    segments[seg_num] = {
                        'type': label,
                        's': s,
                        'e': e,
                        'ramp_rate': ramp_rate,
                        'idxvals': np.arange(s, e + 1),
                        'df': df.iloc[s:e + 1].copy()
                    }
    
                    seg_num += 1
    
            start = i

    # ---- optional plotting ----

    if ax is not None and len(segments):

        plot_dsc(
            df,
            ax,
            'time',
            'temp',
            fmt='--',
            linewidth=0.5
        )

        for seg in segments:

            s = segments[seg]['s']
            e = segments[seg]['e']

            plot_dsc(
                df.iloc[s:e + 1],
                ax,
                'time',
                'temp',
                fmt=fmt + f'C{seg}'
            )

        ax.legend()

    return segments


def remove_extreme_temps(df, n):
    """
    Set 'temp' and 'q' to NaN where 'temp' is within n degrees
    of the minimum or maximum temperature.
    """
    df = df.copy()

    tmin = df['temp'].min()
    tmax = df['temp'].max()

    mask = (df['temp'] < tmin + n) | (df['temp'] > tmax - n)

    df.loc[mask, ['temp', 'q']] = np.nan

    return df

