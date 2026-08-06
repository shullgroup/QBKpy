# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/swell.py

import numpy as np

# example solvent dictionaries
toluene = {'rho':0.867, 'molmass':92.14}
toluene['molvol'] = toluene['molmass']/toluene['rho']
water = {'rho':1, 'molmass':18}
water['molvol'] = water['molmass']/water['rho']

# normally build a DataFrame with 'mair' and 'mh2o' data for each specimen

def swelling(df, solvent_dict):
    m_err = 0.005 #g
    df['vol'] = df['mair'] - df['mh2o'] #implied /rho_h20 which =1
    vol_err = np.sqrt(2)*m_err
    df['rho'] = df['mair']/df['vol']
    df['rho_err'] = np.sqrt((m_err/df['vol'])**2 + (vol_err*df['rho']/df['vol'])**2)
    df['Q'] = 1 + (df['rho'].iloc[0]/solvent_dict['rho'])*((df['mair']/df['mair'].iloc[0]) - 1)
    df['Q_err'] = np.sqrt((df['rho_err'].iloc[0]*(df['mair']-df['mair'].iloc[0])/(solvent_dict['rho']*df['mair'].iloc[0]))**2 + \
                          (m_err*df['rho']/(solvent_dict['rho']*df['mair'].iloc[0]))**2 + \
                            (m_err*df['rho']*df['mair']/(solvent_dict['rho']*df['mair'].iloc[0]**2))**2)
    df['phi'] = 1/df['Q']
    df['phi_err'] = df['Q_err']/df['Q']**2

    return df

def drying(df):
    m_err = 0.005 #g
    df['vol'] = df['mair'] - df['mh2o'] #implied /rho_h20 which =1
    vol_err = np.sqrt(2)*m_err
    df['rho'] = df['mair']/df['vol']
    df['rho_err'] = np.sqrt((m_err/df['vol'])**2 + (vol_err*df['rho']/df['vol'])**2)

    return df

def gel_frac_calcs(df):

    m0 = df['mair'].iloc[0]
    m1 = df['mair'].iloc[1]
    gel_frac = m1/m0
    gel_frac_err = np.sqrt((0.005/m0)**2 + (0.005*m1/(m0**2))**2)

    return gel_frac, gel_frac_err