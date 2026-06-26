# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/rheo.py

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from scipy.signal import savgol_filter,detrend
from scipy.fft import fft, fftfreq
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma as gammaf
from scipy.special import digamma
from pymittagleffler import mittag_leffler

from .utils import readDataFile, remove_step_lines, DEFAULT_CYCLER
from .graphics import double_headed_arrow, vline
from .models import Arrhenius, fitGaussian as _fit_gauss_gen

def readRheo(path, **kwargs):
    chain_time = kwargs.get('chain_time', True)
    chirp = kwargs.get('chirp', False)

    # read if it is a windowed chirp experiment
    if chirp:

        sep = kwargs.get('sep', '\t')
        target_cols = kwargs.get('target_cols', [0,1,2,5,6])
        names = kwargs.get('names', ['time', 'temp', 'torque',
                                     'stress', 'strain'])
        df = readDataFile(path, sep=sep,
                          target_cols=target_cols,
                          names=names)

        
        # remove lines if multiple steps
        df = remove_step_lines(df)
        # force numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # convert stress to Pa from MPa
        df['stress'] = df['stress']*1e6
        # convert strain from % to unitless
        df['strain'] = df['strain']/100

    else:
        try:
            sep = kwargs.get('sep', '\t')
            target_cols = kwargs.get('target_cols', [0,1,2,3,4,5,6,9])
            names = kwargs.get('names', ['storage','loss','tand',
                                        'ang_freq','torque','time',
                                        'temp','viscosity'])
        except Exception as e:
            sep = kwargs.get('sep', '\t')
            target_cols = kwargs.get('target_cols', [0,1,2,3,4,5,6])
            names = kwargs.get('names', ['storage','loss','tand',
                                        'ang_freq','torque','time',
                                        'temp'])

        # remove lines if multiple steps
        df = remove_step_lines(df)
        # force numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # convert moduli to Pa from MPa
        df['storage'] = df['storage']*1e6
        df['loss'] = df['loss']*1e6

        # convert angular frequency to Hz
        df['freq'] = df['ang_freq']/(2*np.pi)

        # if there isn't a viscosity column, calculate one
        try:
            if df['viscosity'].isna().any():
                df['viscosity'] = np.sqrt(df['storage']**2 + df['loss']**2)/df['ang_freq']
        except KeyError:
            df['viscosity'] = np.sqrt(df['storage']**2 + df['loss']**2)/df['ang_freq']

    # round the temperatures
    df['temp'] = df['temp'].round() 

    # chain time together if multiple steps
    if chain_time:
        # Initialize an array to hold the cumulative time offset
        time_offsets = np.zeros(len(df))
        cumulative_offset = 0

        for t in range(1, len(df)):
            prev = df['time'].iloc[t-1]
            curr = df['time'].iloc[t]

            if curr < prev:
                # Time reset detected, add the previous time to cumulative offset
                cumulative_offset += prev
            time_offsets[t] = cumulative_offset

        # Add the cumulative offsets to the original time
        df['time'] = df['time'] + time_offsets
        df['time_min'] = df['time'] / 60
    
    else:
        df['time_min'] = df['time'] / 60

    # drop NaNs
    df = df.dropna()

    return df

def plotRheo(df, **kwargs):

    yaxis = kwargs.get('yaxis', 'moduli')
    mode = kwargs.get('mode', None)
    ax = kwargs.get('ax', None)
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)

    # if not given an ax to plot on, make one
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
        # Apply custom cycler to the specific axes
        ax.set_prop_cycle(DEFAULT_CYCLER)

        # set the title
        ax.set_title(title)
        # set x-axis
        ax.set_xlabel('Frequency (Hz)')

        # set the y-axis labels
        if yaxis == 'moduli':
            ax.set_ylabel('G$^\\prime$, G$^{\\prime\\prime}$ (Pa)')
        elif yaxis == 'viscosity':
            ax.set_ylabel('$|\\eta^*|$ (Pa*s)')
            

    if yaxis == 'moduli':
        
        if len(df['temp'].unique()) > 1:

            temps = df['temp'].unique()

            cmap = plt.cm.magma
            # CHANGE THIS SO NOT EVENLY SPACED
            colors = cmap(np.linspace(0,1, len(temps)))
            
            #CONTINUOUS BAR INSTEAD OF BOUNDARY DEFINED
            norm = mpl.colors.Normalize(vmin=temps.min(), vmax=temps.max())
            color_dict = {t:cmap(norm(t)) for t in temps}
            
            ax.loglog()
            for t in temps:
                ax.plot(df.query('temp == @t')['freq'], 
                        df.query('temp == @t')['storage'], 
                        '-', color=color_dict[t],
                        label='G$^\\prime$')
                ax.plot(df.query('temp == @t')['freq'], 
                        df.query('temp == @t')['loss'],
                        '--', color=color_dict[t],
                        label='G$^{\\prime\\prime}$')
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cb = plt.colorbar(sm, ax=ax, cmap=cmap, norm=norm, label='Temperature ($^{\\circ}$C)')

            legend_lines = [Line2D([0], [0], color='k', 
                                   lw=3, label='G$^\\prime$'),
                            Line2D([0], [0], color='k', 
                                   lw=3, ls='--', label='G$^{\\prime\\prime}$')]
            ax.legend(handles=legend_lines)

        else:
            ax.loglog()
            ax.plot(df['freq'], df['storage'], '-',
                    label='G$^\\prime$')
            ax.plot(df['freq'], df['loss'], '-',
                    label='G$^{\\prime\\prime}$')
            ax.legend()

    elif yaxis == 'viscosity':
        if len(df['temp'].unique()) > 1:

            temps = df['temp'].unique()

            cmap = plt.cm.magma
            # CHANGE THIS SO NOT EVENLY SPACED
            colors = cmap(np.linspace(0,1, len(temps)))
            
            #CONTINUOUS BAR INSTEAD OF BOUNDARY DEFINED
            norm = mpl.colors.Normalize(vmin=temps.min(), vmax=temps.max())
            color_dict = {t:cmap(norm(t)) for t in temps}
            
            ax.loglog()
            
            for t in temps:
                ax.plot(df.query('temp == @t')['freq'], 
                        df.query('temp == @t')['viscosity'], 
                        '-', color = color_dict[t],
                        label='$|\\eta^*|$')
                            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cb = plt.colorbar(sm, ax=ax, cmap=cmap, norm=norm, label='Temperature ($^{\\circ}$C)')

        else:
            ax.loglog()

            ax.plot(df['freq'], df['viscosity'], '-',
                label='$|\\eta^*|$')
            ax.legend()
    
    if savepath:
        plt.savefig(savepath)

    return ax

def viscosity(df, **kwargs):

    freq_min = kwargs.get('freq_min', None)
    freq_max = kwargs.get('freq_max', None)
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)

    if freq_min:
        df = df.query('freq > @freq_min')
    
    if freq_max:
        df = df.query('freq < @freq_max')

    if len(df['temp'].unique()) > 1:
        temps = df['temp'].unique()

        eta_dict = {t:{} for t in temps}
        for t in temps:
            eta_dict[t]['eta'] = np.mean(df.query('temp == @t')['viscosity'])
            eta_dict[t]['eta_err'] = np.std(df.query('temp == @t')['viscosity'])
        
        p0 = [1e-10, 50e3]
        popt, pcov = curve_fit(Arrhenius, [t+273 for t in temps], 
                               [eta_dict[t]['eta'] for t in temps],
                               p0=p0)
        perr = np.sqrt(np.diag(pcov))
        
        fit_temps = np.linspace(temps.min(), temps.max(), 50)
        fit_eta = Arrhenius(fit_temps+273, popt[0], popt[1])

        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
        # Apply custom cycler to the specific axes
        ax.set_prop_cycle(DEFAULT_CYCLER)

        ax.set_xlabel('Temperature ($^{\\circ}$C)')
        ax.set_ylabel('$|\\eta^*|$ (Pa*s)')
        ax.set_title(title)

        ax.errorbar(temps,
                    [eta_dict[t]['eta'] for t in temps],
                    yerr=[eta_dict[t]['eta_err'] for t in temps],
                    marker='o', label = 'Expt.')
        ax.plot(fit_temps, fit_eta, '--',
                label = f'$\\eta_0$ = {popt[0]:0.1e} $\\pm$ {perr[0]:0.1e} Pa*s \n $E_a$ = {popt[1]/1000:0.1f} $\\pm$ {perr[1]/1000:0.1f} kJ/mol')
        ax.legend()

        if savepath:
            plt.savefig(savepath)
        
        return eta_dict

    else:
        eta = np.mean(df['viscosity'])
        eta_err = np.std(df['viscosity'])

        return eta, eta_err

def plotCure(df, **kwargs):

    yaxis = kwargs.get('yaxis', 'moduli')
    ax = kwargs.get('ax', None)
    title = kwargs.get('title', None)
    titlesize = kwargs.get('titlesize', 12)
    savepath = kwargs.get('savepath', None)

    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    # Apply custom cycler to the specific axes
    ax.set_prop_cycle(DEFAULT_CYCLER)

    if yaxis == 'moduli':
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('G$^\\prime$, G$^{\\prime\\prime}$ (Pa)')

        ax.semilogy()
        ax.plot(df['time_min'], df['storage'], '-',
                label='G$^\\prime$')
        ax.plot(df['time_min'], df['loss'], '-',
                label='G$^{\\prime\\prime}$')
        ax.legend()

    elif yaxis == 'viscosity':
        
        ax.semilogy()
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('$|\\eta^*|$ (Pa*s)')

        ax.plot(df['time_min'], df['viscosity'], '-',
            label='$|\\eta^*|$')
        ax.legend()
    
    ax.set_title(title, fontsize=titlesize)
    
    if savepath:
        plt.savefig(savepath)

    return ax

def processChirp(df, **kwargs):
    '''
    Processes the data from a rheological chirp experiment from
    the time domain to the frequency domain.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame output from the rheoRheo function with
        chirp=True

    highf_cutoff : float, default 50.1
        High frequency cutoff to remove frequencies greater
        than the intended input signal
    
    Returns
    -------
    f_df : pd.DataFrame
        Transformed and processed DataFrame
    
    Notes
    -----
    - The complex, storage, and loss moduli are calculated and output, but
        the viscosity is left in the complex form
    
    '''

    highf_cutoff = kwargs.get('highf_cutoff', 50.1)

    # sampling rate information
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    fs = 1/dt
    N = len(df)

    # linear detrend for baseline
    df['torque'] = detrend(df['torque'], type='linear')
    df['stress'] = detrend(df['stress'], type='linear')

    # perform fast Fourier transform and mask
    df['stress_f'] = fft(df['stress'])
    df['strain_f'] = fft(df['strain'])
    freqs_raw = fftfreq(N, dt)
    pos_mask = freqs_raw > 0
    pos_freqs = freqs_raw[pos_mask]
    freq_mask = pos_freqs < 50.1
    freqs = pos_freqs[freq_mask]
    stress_f = df['stress_f'][pos_mask][freq_mask]
    strain_f = df['strain_f'][pos_mask][freq_mask]

    # create new dataframe with transformed and filtered data
    f_df = pd.DataFrame({'freq':freqs, 'strain':strain_f, 'stress':stress_f})

    # additional conversions using stress and strain data
    f_df['modulus'] = f_df['stress']/f_df['strain']
    f_df['storage'] = np.real(f_df['modulus'])
    f_df['loss'] = np.imag(f_df['modulus'])
    f_df['viscosity'] = f_df['modulus']/(f_df['freq']*2*np.pi)

    return f_df

def fitGaussian(df: pd.DataFrame, xprop: str, yprop: str, ax=None, **kwargs):
    """
    Fits rheometer data to a single Gaussian peak.

    This function utilizes the general `fitGaussian_general` to perform the fitting,
    providing rheometer-specific default bounds and ensuring a 'max' peak direction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rheometer data.
    xprop : str
        Column name for the x-axis data (e.g., 'Temperature').
    yprop : str
        Column name for the y-axis data (e.g., 'G_prime_prime', 'Tan_delta').
    ax : mpl.axes.Axes, optional
        Axes object to plot Gaussian fit on. If None, no plot is generated.
    **kwargs:
        Additional keyword arguments passed to `fitGaussian_general`.
        These can include `guess`, `bounds` (to override defaults), `sigma`,
        `absolute_sigma`, `maxfev`, etc.

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
    # Rheometer-specific default bounds (if not provided in kwargs)
    # Based on your original rheometer function's bounds
    rheo_bounds = kwargs.pop('bounds', ([0, 0.01, 0.01, -1000], [1e5, 1e5, 1e4, 1000]))
    
    # Custom plot label formatter for rheometer data
    def rheo_label_formatter(ctr, wid):
        return f'Center = {ctr:0.1f} \n Width = {wid:0.1f}'

    # Call the general fitGaussian function
    return _fit_gauss_gen(
        df,
        x_col=xprop,
        y_col=yprop,
        ax=ax,
        bounds=rheo_bounds,
        peak_direction='max', # Rheometer peaks are typically positive
        plot_label_formatter=rheo_label_formatter,
        **kwargs # Pass remaining kwargs like guess, sigma, maxfev etc.
    )

def fitGaussian(df, xprop, yprop, **kwargs):
    # CHANGE TO USE GENERALIZED fitGaussian IN MODELS
    '''
    Fits data to single Gaussian peak and adds to plot

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DSC data.
    ax : mpl.axes.Axes
        Axes object to plot Gaussian fit on.
    return_err : bool, default False
        Option to return uncertainties for Tg and dT

    Returns
    -------
    ctr : float
        Center of Gaussian fit
    ctr_err : float
        Uncertainty in center of Gaussian fit
    wid : float
        Width of Gaussian fit
    wid_err : float
        Uncertainty in width of Gaussian fit

    '''
    
    ax = kwargs.get('ax', None)
    guess = kwargs.get('guess', [100, 100, 100, 0])
    bounds = kwargs.get('bounds', ([0,1,1,-1000],[1e5,1e5,1e4,1000]))
    sigma = kwargs.get('sigma', None)
    absolute_sigma = kwargs.get('absolute_sigma', False)

    def Gaussian(x, *params):
        y = np.zeros_like(x)
        ctr = params[0]
        amp = params[1]
        wid = params[2]
        y0 = params[3]
        y = y + y0 + amp*np.exp(-((x - ctr)/(2*wid))**2)
        return y

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    #guesses for Temp, amplitude, and width for peak
    ctr_guess = df[yprop].max()
    guess = kwargs.get('guess', [df.query(f'{yprop} == @ctr_guess')[xprop].iloc[0], 1000, 1000, 0])
    
    #fit function to data and plot peak
    popt, pcov = curve_fit(Gaussian, df[xprop], df[yprop], p0=guess,
                           bounds=bounds, sigma=sigma, 
                           absolute_sigma=absolute_sigma, maxfev=5000)
    fit_x = np.linspace(df[xprop].min(),df[xprop].max(),num=1000)
    fit_y = Gaussian(fit_x, *popt)
    ctr = popt[0]
    wid = popt[2]
    
    # parameter uncertainties
    perr = np.sqrt(np.diag(pcov))
    ctr_err = perr[0]
    wid_err = perr[2]

    if ax:
        ax.plot(fit_x, fit_y, ':', color='k',
                label=f'Center = {ctr:0.0f} \n Width = {wid:0.0f}')
        ax.legend()
    
    return ctr, ctr_err, wid, wid_err

def plotCureDeriv(df, **kwargs):

    yaxis = kwargs.get('yaxis', 'modulus')
    ax = kwargs.get('ax', None)
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)
    use_log = kwargs.get('use_log', True)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

    if yaxis == 'modulus':
        #dG = np.diff(df['storage'])
        #dG = savgol_filter(dG, 100, 2)
        #dt = np.diff(df['time_min'])
        #df['dGdt'] = [np.nan] + [d/t for d,t in zip(dG,dt)]

        if use_log:
            log_storage = np.log(df['storage'])

            smoothing = len(df['time_min']) * np.var(log_storage) * 5e-6
            spline = UnivariateSpline(df['time_min'].values,
                                    log_storage.values,
                                    s=smoothing)
            df['dGdt'] = df['storage']*spline.derivative()(df['time_min'].values)

        else:
            # fit smoothing spline to data
            smoothing = len(df['time_min']) * np.var(df['storage']) * 5e-6
            spline = UnivariateSpline(df['time_min'].values,
                                    df['storage'].values,
                                    s=smoothing)
            df['dGdt'] = spline.derivative()(df['time_min'].values)

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('dG$^\\prime$/dt (Pa/min)')

        #ax.semilogy()
        ax.plot(df['time_min'], 
                df['dGdt'], 
                '-',
                label='dG$^\\prime$/dt')
        #ax.legend()

    elif yaxis == 'viscosity':
        
        ax.semilogy()
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('$|\\eta^*|$ (Pa*s)')

        ax.plot(df['time_min'], df['viscosity'], '-',
            label='$|\\eta^*|$')
        ax.legend()
    
    ax.set_title(title)
    
    if savepath:
        plt.savefig(savepath)

    return ax