# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/fracture.py

import numpy as np
import pandas as pd

from .utils import default_cycler, read_data_file

def read_kic(path, **kwargs):

    sep = kwargs.get('sep', ',') 
    target_cols = kwargs.get('target_cols', [0, 1, 2])
    names = kwargs.get('names', ['time', 'disp', 'force'])

    df = read_data_file(path, sep=sep,
                        target_cols=target_cols,
                        names=names)
    
    return df

def KIcfx(x, **kwargs):
    '''
    Eq A1.1 or A2.1 from ASTM D5045

    Parameters
    ----------
    x : float
        Intended to be the pre-crack to sample width ratio (a/W) for a
        KIc experiment as defined in ASTM D5045.

    geometry : str, default 'SENB'
        Geometry used for the fracture experiment, 'SENB' or 'CT'

    Returns
    -------
    float
        Function value for Equation A1.1 (SENB) or A2.1 (CT)

    '''

    geometry = kwargs.get('geometry', 'CT')

    if geometry == 'SENB':
        return 6*np.sqrt(x)*(1.99 - x*(1-x)*(2.15 - 3.93*x + 2.7*x**2))/((1 + 2*x)*(1 - x)**(3/2))
    
    elif geometry == 'CT':
        return (2 + x)*(0.886 + 4.64*x - 13.32*x**2 + 14.72*x**3 - 5.6*x**4)/(1 - x)**(1.5)

def deriv_KIcfx(x, **kwargs):
    '''
    Derivative of Eq A1.1 or Eq A2.1 from ASTM D5045 with respect to x.  
    This is used for error propagation calculation.

    Parameters
    ----------
    x : float
        Intended to be the pre-crack to sample width ratio (a/W) for a
        KIc experiment as defined in ASTM D5045.
    
    geometry : str, default 'SENB'
        Sample geometry used for KIc testing. 'SENB' or 'CT'

    Returns
    -------
    float
        Derivative value of Equation A1.1 or A2.1 evaluated at x.

    '''

    geometry = kwargs.get('geometry', 'CT')

    if geometry == 'SENB':
        return (-16.2*x**6 + 36.09*x**5 - 11.61*x**4 - 23.0175*x**3 + 31.515*x**2 - 4.8375*x + 1.4925) \
            / ((x-1)**2*np.sqrt((1-x)*x)*(x + 0.5)**2)
    
    elif geometry == 'CT':
        return (19.6*x**5 - 36.8*x**4 - 10.1*x**3 + 59.36*x**2 - 38.917*x + 12.824)/(1 - x)**(2.5)

def P_to_K(P, B, W, x, **kwargs):
    '''
    Conversion of loads to stress intensity for fracture specimens.
    
    Parameters
    ----------
    P : float
        Load in N
    
    B : float
        Sample thickness in mm
    
    W : float
        Sample width in mm
    
    x : float
        Ratio of crack length (a) to width (W)
    
    geometry : str, default 'CT'
        Fracture sample geometry. 'CT' or 'SENB'
    
    Returns
    -------
    K : float
        Stress intensity factor in MPa*sqrt(m)
    
    Notes
    -----
    - Uses the KIcfx function, which can take CT or SENB geometries.
    
    '''
    geometry = kwargs.get('geometry', 'CT')
    
    K = (P/1000)*KIcfx(x, geometry=geometry)/((B/10)*np.sqrt(W/10))

    return K