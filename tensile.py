# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/tensile.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import read_data_file

def readTensile(path, dimensions, **kwargs):
    '''
    Read uniaxial tensile testing data and convert to DataFrame.
    
    Parameters
    ----------
    path : Path
        Path object to the csv file containing the data.
    dimensions : dict
        Dictionary containing sample dimensions in format
        {'w': list (array of widths across sample gauge),
         't': list (array of thicknesses across gauge),
         'L0': float (original grip separation)}
    extensometer : bool, default False
        Flag for whether or not an extensometer was used.
    units : str, default 'metric'
        Convention of units. Currently 'metric' (mm and N)
        or 'imperial' (in and lbf)
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with raw (force and displacement) and 
        processed (stress and strain) data
    
    Notes
    -----
    - Currently (9-10-2025) written for csv files from Instron.
    - Even if given imperial units, will process data to metric
    
    '''

    sep = kwargs.get('sep',',')
    
    extensometer = kwargs.get('extensometer', False)
    units = kwargs.get('units', 'metric')

    width = dimensions['w']
    thickness = dimensions['t']

    try:
        L0 = dimensions['L0']
    except KeyError:
        extensometer = True
    
    if units == 'imperial':
        #read in data
        if extensometer:
            target_cols = kwargs.get('target_cols', [0,1,2,3])
            names = kwargs.get('names', ['time', 'disp_in',
                                         'force_lbf', 'strain%'])
            df = read_data_file(path, sep=sep,
                                target_cols=target_cols,
                                names=names)

        else:
            target_cols = kwargs.get('target_cols', [0,1,2])
            names = kwargs.get('names', ['time', 'disp_in',
                                         'force_lbf'])
            df = read_data_file(path, sep=sep,
                                target_cols=target_cols,
                                names=names)

        df = df.drop([0])
        #convert to SI units (mm and N)
        df['disp'] = df['disp_in']*25.4
        df['force_N'] = df['force_lbf']*4.44822

        width = [25.4*w for w in width]
        thickness = [25.4*t for t in thickness]
        L0 = L0*25.4

    else:
        #read in data
        if extensometer:
            target_cols = kwargs.get('target_cols', [0,1,2,3])
            names = kwargs.get('names', ['time', 'disp',
                                         'force_N', 'strain%'])
            df = read_data_file(path, sep=sep,
                                target_cols=target_cols,
                                names=names)
        else:
            target_cols = kwargs.get('target_cols', [0,1,2])
            names = kwargs.get('names', ['time', 'disp_in',
                                         'force_lbf'])
            df = read_data_file(path, sep=sep,
                                target_cols=target_cols,
                                names=names)

        if 'kN' in df['force_N'].iloc[0]:
             df = df.rename(columns={'force_N': 'force_kN'})
             df = df.drop([0]).astype(float)
             df['force_N'] = df['force_kN'] * 1000
        
        else:
             df = df.drop([0]).astype(float)

    #filter out data after failure
    Pmax = max(df['force_N'])
    disp_max = df.query('force_N == @Pmax')['disp'].iloc[0]
    df = df.query('disp <= @disp_max')

    #filter out data prior to tension and correct L0
    #this also tries to filter out artifacts prior to test
    df = df.query('force_N > 0')
    index_minP = df['force_N'].idxmin()
    for i in np.arange(0,index_minP):
        try:
            df = df.drop([i])
        except KeyError:
            continue

    disp_offset = df['disp'].iloc[0]
    if not extensometer:
        L0 = L0 + disp_offset
    df['disp'] = df['disp'] - disp_offset

    #convert to stress and strain
    if extensometer:
        df['strain'] = df['strain%']/100
    else:
        df['strain'] = df['disp']*(1/L0)
    
    df['stretch'] = df['strain'] + 1.
    df['stress'] = df['force_N']*(1/(np.mean(width)*np.mean(thickness)))
    df['strain_true'] = np.log(df['stretch'])
    df['stress_true'] = df['stress']*df['stretch']
    df['stress_MR'] = df['stress']/(df['stretch'] - df['stretch']**-2)
    #uncertainty in stress due only to thickness and width variation
    #can expand to other variables if needed
    t_var = (np.std(thickness)/(np.mean(width)*np.mean(thickness)**2)*df['force_N'])**2
    w_var = (np.std(width)/(np.mean(thickness)*np.mean(width)**2)*df['force_N'])**2
    df['stress_err'] = np.sqrt(t_var + w_var)
    df['stress_low'] = df['stress'] - df['stress_err']
    df['stress_hi'] = df['stress'] + df['stress_err']

    return df

def plotTensile(*df, **kwargs):
    '''
    Function description.
    
    Parameters
    ----------
    variable : type
       description
    
    Returns
    -------
    variable : type
       description
    
    Notes
    -----
    - additional notes
    
    '''
    labels = kwargs.get('labels', None)
    max_strain = kwargs.get('max_strain', None)
    max_stress = kwargs.get('max_stress', None)
    title = kwargs.get('title', None)
    legend_size = kwargs.get('legend_size', 10)
    legend_columns = kwargs.get('legend_columns', 1)
    savepath = kwargs.get('savepath', None)
    ax = kwargs.get('ax', None)
     
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
        ax.set_title(title)

    for i in np.arange(len(df)):
        if labels:
            ax.plot(df[i]['strain'], df[i]['stress'], 
                    label=labels[i])
        else:
            ax.plot(df[i]['strain'], df[i]['stress'])
        ax.fill_between(df[i]['strain'], df[i]['stress_hi'], y2=df[i]['stress_low'],
                alpha=0.3)
        ax.set_xlabel('Eng. Strain')
        if max_strain:
            ax.set_xlim([0, max_strain])
        else:
            ax.set_xlim(left=0)
        ax.set_ylabel('Eng. Stress (MPa)')
        if max_stress:
            ax.set_ylim([0, max_stress])
        else:
            ax.set_ylim(bottom=0)
        if labels:
            ax.legend(prop={'size':legend_size}, ncol=legend_columns)
        

    if savepath:
         plt.savefig(savepath)
    
    plt.show(block=False)
     
    return

def plotMR(df, **kwargs):
    # Work in progress

    def linear(x, m, b):
        return m*x + b

    df = df.query('stretch > 1.05 and stretch < 2')
    popt, pcov = curve_fit(linear, 1/df['stretch'], df['stress_MR'],
                           p0=[0.1,0.5],
                           bounds=([0,0],[10,10]))

    perr = np.sqrt(np.diag(pcov))
    C1 = popt[1]/2
    C2 = popt[0]/2
    fitrange = np.arange(1/(max(df['stretch'])), 0.95, 0.02)

    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

    ax.plot(1/df['stretch'], df['stress_MR'], 'o')
    ax.plot(fitrange, [popt[0]*x + popt[1] for x in fitrange], '--', color='#000000',
            label=f'C1 = {C1:0.2f} MPa \n C2 = {C2:0.2f} MPa')
    ax.set_xlabel('1/$\\lambda$')
    ax.set_ylabel('$\\frac{\\sigma}{\\lambda - \\lambda^{-2}}$ (MPa)')
    ax.legend()

    plt.show(block=False)

    return