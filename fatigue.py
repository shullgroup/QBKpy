import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter
from sympy import symbols, diff, log, sqrt, lambdify, Rational
from copy import deepcopy

# a is the crack length
# W is the sample width in the direction of crack propgation
# L is the distance beteen the bottom loading points
# B is sample thickness

# Most of this is from the following reference, Eqs. 20-23
# “Stress Intensity factor, compliance and CMOD for a General Three-Point-Bend 
# Beam.” G.V. Guinea, J.Y. Pastor, J. Planas & M. Elices, International 
# Journal of Fracture 89, 103–116 (1998) 
# (http://dx.doi.org/10.1023/A:1007498132504)


# define a bunch of symolic variables for function reaction
E_sym, nu_sym, B_sym, W_sym, L_sym, P_sym, alpha_sym, beta_sym, C_in_sym = (
    symbols(['E', 'nu', 'B', 'W', 'L', 'P', 'alpha', 'beta', 'C_in'], 
    positive=True, real=True))

E_r_sym = E_sym/(1-nu_sym**2)

# compliance expressions from Guinea et al. Eqs. 20-23
# International Journal of Fracture 89: 103–116, 1998
c1 = (-0.378*alpha_sym**3*log(1-alpha_sym)+alpha_sym**2*
            (0.29+1.39*alpha_sym-1.6*alpha_sym**2)/
            (1+0.54*alpha_sym-0.84*alpha_sym**2))

c2 = (1.1*alpha_sym**3*log(1-alpha_sym)+alpha_sym**2*
            (-3.22-16.4*alpha_sym+28.1*alpha_sym**2-11.4*alpha_sym**3)/
            ((1-alpha_sym)*(1+4.7*alpha_sym-4*alpha_sym**2)))

c3 = (-0.176*alpha_sym**3*log(1-alpha_sym)+alpha_sym**2*
            (8.91-4.88*alpha_sym-0.435*alpha_sym**2+0.26*alpha_sym**3)/
            ((1-alpha_sym)**2*(1+2.9*alpha_sym)))

C_sym = (beta_sym**3/(48*E_sym*B_sym))+(1/(E_r_sym*B_sym))*(c1 + 
                                 beta_sym*c2+beta_sym**2*c3)


# now calculate G and K from this compliance expression
G_sym = (P_sym**2/(2*W_sym*B_sym))*diff(C_sym,alpha_sym)
K_sym = sqrt(G_sym*E_r_sym)


# define the estimate for K valid for beta = 4
K4 = (P_sym/(B_sym*W_sym**0.5))*6*alpha_sym**0.5*(1.99-alpha_sym*
        (1-alpha_sym)*(2.15-3.93*alpha_sym+2.7*alpha_sym**2))/(
            (1+2*alpha_sym)*(1-alpha_sym)**1.5)
            

# expression enabling alpha to be extracte from delC (Guinea, Eqs. 24, 25)
q1 = 0.98+3.77*beta_sym
q2 = -(9.1+2.9*beta_sym**2)/(1+0.168*beta_sym)
q3 = -3.2*beta_sym+8.9*beta_sym**2

alpha_calc = (C_in_sym*E_r_sym*B_sym)**(Rational(1,2))/((C_in_sym*E_r_sym*B_sym + 
                               q1*(C_in_sym*E_r_sym*B_sym)**Rational(1,2)+ 
                               q2*(C_in_sym*E_r_sym*B_sym)**Rational(1,3) + 
                               q3)**Rational(1,2))

def calc_K(alpha, beta, B, P, W, nu):
    """
    Calculate K.

    Parameters:
        alpha (float): a/W (crack length normalize by its maximum val.).
        beta (float): L/W (span normlized by sample width).
        P (float):  Normal applied in 3-point bend test.
        B (float): Sample dimension parallel to crack.
        W (float): Sample depth in direction of crack propagation.
        nu (float): Poisson's ratio.

    Returns:
        float: K, stress intensity factor.
    """
    func = lambdify([alpha_sym, beta_sym, P_sym, B_sym,  W_sym, nu_sym], K_sym, 'numpy')
    return func(alpha, beta, P, B, W, nu)



def calc_K4(alpha, P, B, W):
    """
    Estimate of K for beta=4

    Parameters:
        alpha (float): a/W (crack length normalize by its maximum val.).
        P (float):  Normal applied in 3-point bend test.
        B (float): Sample dimension parallel to crack.
        W (float): Sample depth in direction of crack propagation.


    Returns:
        float: K, stress intensity factor for beta = 4.
    """
    func = lambdify([alpha_sym, P_sym, B_sym, W_sym], K4, 'numpy')
    return func(alpha, B, P, W)



def calc_P(alpha, beta, B, K, W, nu):
    """
    Calculate the value of the symbolic expression K_sym using provided parameters.

    Parameters:
        alpha (float): a/W (crack length normalize by its maximum val.).
        beta (float): L/W (span normlized by sample width).
        B (float): Sample dimension parallel to crack.
        P (float): Applied load.
        W (float): Sample depth in direction of crack propagation.
        nu (float): Poisson's ratio.

    Returns:
        float: Stress intesity factor, K.
    """
    func = lambdify([alpha_sym, beta_sym, P_sym, B_sym,  W_sym, nu_sym], 
                    K_sym, 'numpy')
    return K/func(alpha, beta, 1, B, W, nu)

def calc_C(alpha, beta, B, E, nu):
    """
    Calculate the value of the symbolic expression C_sym using provided parameters.

    Parameters:
        alpha (float): a/W (crack length normalize by its maximum val.).
        beta (float): .
        B (float): Sample dimension parallel to crack.
        E (float): Young's modulus.

    Returns:
        float: Calculated sample compliance.
    """
    func = lambdify([alpha_sym, beta_sym, B_sym, E_sym, nu_sym], C_sym, 'numpy')
    return func(alpha, beta, B, E, nu)

from scipy.optimize import root_scalar

def calc_alpha_single(C, beta, B, E, nu):
    """
    Find the value of alpha (0 < alpha < 1) that yields the target compliance C.

    Parameters:
        C (float): Desired compliance value.
        beta (float): Parameter beta.
        B (float): Sample dimension parallel to crack.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.


    Returns:
        float: Alpha value that gives the target C.

    Raises:
        ValueError: If root finding does not converge.
    """
    def objective(alpha):
        return calc_C(alpha, beta, B, E, nu) - C
    
    try:
        result = root_scalar(objective, bracket=(0.01, 0.99), method='brentq')
    except:
        print(f'problem with beta={beta}, B={B}, E={E}, nu={nu}, C={C}')
        sys.exit()
    
    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge. "
            "Try adjusting alpha_bounds or check input parameters.")
        
calc_alpha = np.vectorize(calc_alpha_single)

def calc_alpha_approx(C, beta, B, E):
    """
    Calculate the value of alpha from the compliance for KIC sample,
    using Guinea, Eqs. 24, 25.

    Parameters:
        C (float): Sample compliance.
        beta (float): L/W (span normlized by sample width).
        B (float): Sample dimension parallel to crack.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.

    Returns:
        float: Evaluated result of alpha_calc with the given inputs.
    """
    func = lambdify([C_in_sym, beta_sym, B_sym, E_sym, nu_sym], alpha_calc, 'numpy')
    return func(C, beta, B, E)


def read_file(directory, df_in, row):
    # read fatigue data into a dataframe
    filename = df_in['Name'][row]+'.CSV'
    file = os.path.join(directory,filename)
    f = open(file)
    lines=f.readlines()
    header=[]
    i = 0
    # might want to comue up with a more robust way of doing this
    while 'Elapsed Time' not in lines[i]:
        header.append(lines[i])
        i=i+1
    
    header=np.array(header)
    header=pd.DataFrame(header)
    
    # read the header information
    print(f'reading {file}')
    df = pd.read_csv(file,  header=i+1, sep=',', encoding="ISO-8859-1",
                     skip_blank_lines=False, usecols=[0,1,2,3,4],
                     names=['Points', 't_tot', 't_scan', 'd', 'P'])
    
    # get rid of any rows with null values
    breaks = df.isnull().any(axis=1).to_numpy().nonzero()[0]
    df.drop(breaks, inplace=True)
    df = df.reset_index(drop=True)
    
    # change d unit from mm to m
    df['d'] = df['d']/1000
          
    # integrate the load displacement curve
    df['diss']=cumulative_trapezoid(df['P'], df['d'], initial=0)

    return header, df


def read_fatigue_CT(folder, sample_dict, **kwargs):
    '''
    Read fatigue data from a folder containing the tracking csv and stop csv files.
    Designed for outputs from Instron Electropuls 10000
    
    Parameters
    ----------
    folder : Path
        Path object to folder containing tracking and Stop csv files for a given test.

    sample_dict : dict
        Dictionary containing sample dimensions (B and W) and modulus (E)

    skiprows : int, default 1
        Number of rows to skip when reading csv files
    
    multistep : bool, default False
        Flag to plot an extra step on the da/dN vs DeltaK plots
    
    allstep : bool, default False
        Flag to plot all steps on the da/dN vs DeltaK plots
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing relevant sample data, calculated values like
        compliance, crack lengths, crack growth rates, and stress intensities
    
    Notes
    -----
    - Uses fatigueCompliance method for calculation of values from experimental data.
    
    '''
    

    skiprows = kwargs.get('skiprows', 1)
    multistep = kwargs.get('multistep', False)
    allstep = kwargs.get('allstep', False)
    
    # open the folder and pull the tracking and end files
    trackfile = list(folder.glob('*.tracking.csv'))[0]
    endfile = list(folder.glob('*.Stop.csv'))[0]
    files = [trackfile, endfile]

    df_concat = pd.DataFrame()
    for p in files:
        # read data file
        with open(p, 'r') as f:
                df = pd.read_csv(f, delimiter=",", skiprows=skiprows,
                                usecols=[0,2,4,6,7],
                                names=['time','cycles','step','position','force'])
        
        # normalize position data make displacement
        df = df.query('step in [2,4,6,8,10,12,14]')
        maxstep = df['step'].max()
        if len(df.query('step == @maxstep')['cycles']) < 10:
            maxstep = maxstep - 2

        # include extra step if failed at beginning of step
        if multistep:
            extrastep = maxstep - 2
            df = df.query('step == @maxstep or step == @extrastep')
        
        # show all steps if interested
        elif allstep:
            df = df
        
        # otherwise just the last step (where failure occurs)
        else:
            df = df.query('step == @maxstep')

        #normalize cycle numbers
        df['cycles'] = df['cycles'] - min(df['cycles']) + 1

        # temp arrays of position and force for conversion to displacement
        # and building dataframe for each cycle and then concatenating into
        # one big dataframe
        temp_pos = []
        temp_force = []
        df_reduced = pd.DataFrame(columns=['cycles','position','force'])
        #skipper = 0
        for c in df['cycles'].unique():
            temp_pos = df.query('cycles == @c')['position'].tolist()
            temp_force = df.query('cycles == @c')['force'].tolist()
            disp = [x - min(temp_pos) for x in temp_pos]
            new_row = pd.DataFrame([{'cycles':c, 'position':temp_pos,
                                'disp':disp, 'force':temp_force}])
            df_reduced = pd.concat([df_reduced, new_row])

        
        #df_comp = fatigueCompliance(df_reduced, sample_dict)
        df_concat = pd.concat([df_concat, df_reduced])

    # perform compliance calculation to crack lengths
    df = fatigue_compliance_CT(df_concat, sample_dict)

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

def CT_compliance(E, C, C_err, B):
    '''
    Conversion of modulus (E), compliance (C), and thickness (B) data to relative crack length (a/W)
        for a CT specimen with displacements measured along the load line.  Equation and parameters
        taken from ASTM E647-00 Fig A1.3
    
    Parameters
    ----------
    E : float
        Elastic modulus of sample in MPa.
    
    C : float
        Compliance (displacement/load) in mm/N
    
    B : float
        Sample thickness in mm
    
    Returns
    -------
    x : float
        Relative crack length (a/W)
    
    Notes
    -----
    - Can add uncertainty propagation at some point
    
    '''
    
    C0 = 1.0002
    C1 = -4.0632
    C2 = 11.242
    C3 = -106.04
    C4 = 464.33
    C5 = -650.68

    u = 1/(np.sqrt(E*C*B) + 1)

    x = C0 + C1*u + C2*u**2 + C3*u**3 + C4*u**4 + C5*u**5
    dxdu = C1 + 2*C2*u + 3*C3*u**2 + 4*C4*u**3 + 5*C5*u**4
    #dudE = -(C*B*u**2)/(2*np.sqrt(E*C*B))
    dudC = -(E*B*u**2)/(2*np.sqrt(E*C*B))
    #dudB = -(E*C*u**2)/(2*np.sqrt(E*C*B))

    #dxdE = dxdu*dudE
    dxdC = dxdu*dudC
    #dxdB = dxdu*dudB
    varC = C_err**2 * dxdC**2
    x_err = np.sqrt(varC) #add other terms if want to add their uncertainty

    return x, x_err

def fatigue_compliance_CT(df, sample_dict, **kwargs):
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
    
    
    step_length = kwargs.get('step_length', 10000)
    min_ratio = kwargs.get('min_ratio', 0.05)
    max_offset = kwargs.get('max_offset', 4)
    front_filter = kwargs.get('front_filter', 50)

    # remove first few cycles from each step which are unstable
    df = df.query('cycles > @front_filter')
    df = df.reset_index()
    for i in np.arange(1,8):
        df = df.drop(df[(df.cycles > i*step_length) & (df.cycles < i*step_length + front_filter)].index)
    
    # drop last cycle
    df = df.drop(df[(df.cycles == max(df.cycles))].index)

    temp_C = []
    temp_C_err = []
    temp_Pmin = []
    temp_Pmax = []
    for c in df['cycles']:
        # pull force and displacement for each cycle
        disp = df.query('cycles == @c')['disp'].iloc[0]
        force = df.query('cycles == @c')['force'].iloc[0]

        # get min and max forces for Delta K calculations later
        temp_Pmin.append(min(force))
        temp_Pmax.append(max(force))

        #filter force-displacement data for linear fitting for compliance
        min_i = round(min_ratio*len(disp))
        max_i = force.index(max(force)) - max_offset
        disp_filt = disp[min_i:max_i]
        force_filt = force[min_i:max_i]

        p, V = np.polyfit(force_filt, disp_filt, 1, cov=True)

        temp_C.append(p[0])
        temp_C_err.append(np.sqrt(np.diag(V))[0])

    df['C'] = temp_C
    df['C_err'] = temp_C_err
    df['Pmin'] = temp_Pmin
    df['Pmax'] = temp_Pmax
    B = np.mean(sample_dict['B'])
    W = np.mean(sample_dict['W'])

    # smooth the compliance values to reduce noise
    # uses a Savitsky-Golay filter of length 5 and degree 1
    df['C_smooth'] = savgol_filter(df['C'].to_numpy(),
                                   window_length=5, polyorder=1)

    df['alpha'], df['alpha_err'] = CT_compliance(sample_dict['E'], df['C_smooth'], 
                                                 df['C_err'], B)
    
    #df['alpha_window'] = savgol_filter(df['alpha'].to_numpy(), 
    #                                   window_length=9, polyorder=2)
    df['a'] = df['alpha']*W
    df['a_err'] = df['alpha_err']*W
    df['Kmin'] = P_to_K(df['Pmin'], B, W, df['alpha'])
    df['Kmax'] = P_to_K(df['Pmax'], B, W, df['alpha'])
    df['DeltaK'] = df['Kmax'] - df['Kmin']


    da = np.diff(df['a'])
    dN = np.diff(df['cycles'])
    dadN = [x/y for x,y in zip(da,dN)]

    numerator = [i**2 + j**2 for i,j in zip(df['a_err'],df['a_err'].shift(-1))]
    vardadN = [x/y**2 for x,y in zip(numerator,dN)]
    dadN_err = np.sqrt(vardadN).tolist()

    df['dadN'] = [np.nan] + dadN
    df['dadN_err'] = [np.nan] + dadN_err

    df = df.dropna().query('dadN > 0')

    #df['dadN_log'] = np.log(df['dadN'])
    #df['dadN_log_smooth'] = savgol_filter(df['dadN_log'].to_numpy(),
    #                                      window_length=9, polyorder=2)
    #df['dadN_smooth'] = np.exp(df['dadN_log_smooth'])

    #print(df['a'].iloc[-1] - df['a'].iloc[0])
    #print(simpson(df['dadN'], df['cycles']))

    # add uncertainties


    return df

def fatigue_plot_CT(df, **kwargs):

    detailed = kwargs.get('detailed', False)
    min_speed = kwargs.get('min_speed', 5e-5)
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)

    # collect maximum delta K as upper bound
    Kc = max(df['DeltaK'])

    # use minimum crack speed to collect threshold K
    try:
        thr_cycle = df.query('dadN < @min_speed')['cycles'].iloc[-1]
    except IndexError:
        thr_cycle = 0
    Kthr = df.query('cycles > @thr_cycle')['DeltaK'].iloc[0]

    fit_df = df.query('DeltaK > @Kthr')
    fit_df = fit_df.query('DeltaK < @Kc')

    def parisLaw(k, A, m):
        return A*k**m

    popt, pcov = curve_fit(parisLaw, fit_df['DeltaK'], fit_df['dadN'],
                            p0=[1e-5, 20],
                            bounds=([0,1],[10,40]))
    
    A, m = popt
    A_err, m_err = np.sqrt(np.diag(pcov))

    fitK = np.arange(Kthr, Kc, 0.01)
    fitdadN = parisLaw(fitK, A, m)

    if detailed:

        fig, ax = plt.subplots(1,3, figsize=(12,3), constrained_layout=True)
        
        # cycles force-displacement plot
        norm = mpl.colors.Normalize(vmin=min(df['cycles']), vmax=max(df['cycles']))
        cmap = mpl.colormaps['magma']
        ax[0].set_xlabel('Displacement (mm)')
        ax[0].set_ylabel('Force (N)')
        ax[0].set_xlim([-0.025, 1.1*max(df['disp'].iloc[-1])])

        for c in df['cycles']:
            ax[0].plot(df.query('cycles == @c')['disp'].iloc[0],
                       df.query('cycles == @c')['force'].iloc[0],
                       color=cmap(norm(c)))
        
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, ax=ax[0], cmap=cmap, norm=norm, label='Cycle')
    
        # crack length vs. cycles plot

        ax[1].plot(df['cycles'], df['a'], 'o')
        ax[1].set_xlabel('Cycles')
        ax[1].set_ylabel('Crack Length, a (mm)')

        #da/dN vs. DeltaK plot
        ax[2].loglog(df['DeltaK'], df['dadN'], 'o')
        ax[2].set_xlabel('$\\Delta$K (MPa$\\sqrt{m}$)')
        ax[2].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax[2].xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        ax[2].set_ylabel('da/dN (mm/cycle)')

        ax[2].plot(fitK, fitdadN, '--', color='#000000', 
                    label=f'm = {m:0.2f} $\\pm$ {m_err:0.2f}')
        ax[2].legend()
    
    else:

        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

        ax.loglog(df['DeltaK'], df['dadN'], 'o')
        ax.set_xlabel('$\\Delta$K (MPa$\\sqrt{m}$)')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        ax.set_ylabel('da/dN (mm/cycle)')

        ax.plot(fitK, fitdadN, '--', color='#000000', 
                label=f'm = {m:0.2f} $\\pm$ {m_err:0.2f}')
        ax.legend()

    if title:
        plt.suptitle(title)
    plt.show(block=False)

    if savepath:
        plt.savefig(savepath)

    return m, m_err, Kthr, Kc

def make_data_dict(directory, df_in, row):
    ''' Create dictionary Bose-formatted input fatigue data file specified by a
    specific row in the input dataframe
    Args:
        directory (string):
            Directory containing the input data files
        df_in (dataframe):
            dataframe containing the filenames to read, dimensions of
            each sample, along with any identifying sample information.
        filetype (string):
            'crack': standard fatigue file for sample with precrack
            'nocrack': uncracked sample used for determination of Gamma
            
    Returns:
        data_dict (dictionary):
            Dictionary of data suitable for subsequent analysis.
    '''
                
    header, df = read_file(directory, df_in, row)
    Description = df_in['Description'][row]

    # note that we use the python convention that the first group, or the
    # first cycle within a group, is labeled as 0 and not 1
    P_array = np.array(df['P'])
    t_array = np.array(df['t_tot'])

    # find indices of different groups of data  
    end_idx = np.asarray(df.index[(df['Points'] < df['Points'].shift())]-1)
    end_idx = np.append(end_idx, df.index[-1])

    start_idx = np.asarray(df.index[df['Points'].astype(int) == 1])
    
    ngroups = len(start_idx)

    #  number of different data groups included in dataframe
    # (each dataset assumed to have 4 cycles for now)
    
    # initialize the dictionary
    data_dict={'ngroups':ngroups}
    detail={}
    borders_idx={}
    Wdiss=np.zeros(ngroups) # hysteresis energy
    dmin = np.zeros(ngroups)  # minimum sdisplacement
    dmax = np.zeros(ngroups)  # maximum displacement
    Pmin = np.zeros(ngroups)  # minimum load
    Pmax = np.zeros(ngroups)  # maximum load   
    phi = np.zeros(ngroups)  # effective phase angle
    cycle = np.zeros(ngroups)  # cycle number for current group
    
    for g in np.arange(ngroups):
        detail[g]={'load':{}, 'unload':{}}  
        # first index for dictionary refers to the group
        # second index corresponds to cycle within that group
        idx_all = np.arange(start_idx[g], end_idx[g])
        t = df['t_tot'][idx_all]
        
        peaks, _ = find_peaks(savgol_filter(df['P'][idx_all], 51, 3))
        valleys, _ = find_peaks(-savgol_filter(df['P'][idx_all], 51, 3))
        borders_idx = sorted(np.concatenate([t.iloc[peaks].index, 
                                             t.iloc[valleys].index]))
        borders_idx = np.concatenate([borders_idx, [idx_all[-1]]])
        
        # don't count load or unloading portions less than 10% of load
        # amplitude
        while (abs(P_array[borders_idx[1]]-P_array[borders_idx[0]]) <
            0.1*(max(P_array)-min(P_array)) and len(borders_idx)>2):
                borders_idx=np.delete(borders_idx,0)
                
        while (abs(P_array[borders_idx[-2]]-P_array[borders_idx[-1]]) <
            0.1*(max(P_array)-min(P_array))  and len(borders_idx)>2):
                borders_idx=np.delete(borders_idx,-1)          
        
        # don't count this group if there aren't enough borders
        if (len(peaks)+len(valleys)+1)<7:
            print('short borders_idx for group '+str(g)+' of '+str(ngroups))
            del detail[g]
            break                
        
        detail[g]['borders_idx']=borders_idx
        
        # fill up 'all' dictionary 
        detail[g]['all']={}
        detail[g]['all']['idx']=idx_all
        for val in ['P', 'd', 't_tot','diss']:
            detail[g]['all'][val]=df[val][idx_all]
                            
        # calculate the period of the oscillation - here we just make sure our
        # assumption of 100 points per cycle, 2 cycles per second is not too far
        # out of whack
        delt=df['t_tot'][end_idx[g]-1]-df['t_tot'][end_idx[g]-2]
        if abs(delt-0.005)/0.005 > 0.1:
            # We assume cylcing at 2 Hz, 100 data points per cycle
            print('something amiss with delta t')
            
        previous_cycles = int(round(2*t_array[start_idx[g]])) 
        cycle[g] = previous_cycles+1 # more or less an avg. of the group of 4
        # calculate energy within hysteresis loop
        
        Wdiss[g]=np.average((np.array(detail[g]['all']['diss'][100:])-
        np.array(detail[g]['all']['diss'][0:-100])))
        
        # now determine max and min for d and P and effective phase angle
        dmin[g]=min(detail[g]['all']['d'])
        dmax[g]=max(detail[g]['all']['d'])
        Pmin[g]=min(detail[g]['all']['P'])
        Pmax[g]=max(detail[g]['all']['P'])
        
        phi[g] = (180/np.pi)*np.arcsin((4*Wdiss[g])/(np.pi*(dmax[g]-
                                        dmin[g])*(Pmax[g]-Pmin[g])))
      
        # fill up 'load' and 'unload' sub-dictionaries
        gload=0
        gunload=0
 
        for b in range(len(borders_idx) - 1):  

            if P_array[borders_idx[b+1]]>P_array[borders_idx[b]]:
                # make sure we have caught only increasing portions of the
                # displacement, otherwise we have problems with the spline
                # fit
                while P_array[borders_idx[b]]>=P_array[borders_idx[b]+1]:
                    borders_idx[b]=borders_idx[b]+1
                while P_array[borders_idx[b+1]-1]>=P_array[borders_idx[b+1]]:
                  borders_idx[b+1]=borders_idx[b+1]-1
                              
                typ='load'
                gnum=gload
                gload=gload+1
            else:
                typ='unload'
                gnum=gunload
                gunload=gunload+1
  
            idx_range = np.arange(borders_idx[b], borders_idx[b+1])
            detail[g][typ][gnum]={'idx':idx_range}
            detail[g][typ][gnum]['cycle_num']=previous_cycles+gnum+1
            
            for val in ['P', 'd', 't_tot', 'diss']:
                detail[g][typ][gnum][val]=df[val][detail[g][typ][gnum]['idx']]
                
    summary_df = pd.DataFrame({  
    'cycle':cycle,            
    'Wdiss': Wdiss,
    'dmin': dmin,
    'dmax': dmax,
    'Pmin': Pmin,
    'Pmax': Pmax,
    'phi': phi})  
    data_dict['ncycles']= previous_cycles+4
    data_dict['summary'] = summary_df
    data_dict['detail']=detail
    data_dict['filename']=df_in['Name'][row]
    data_dict['header']=header
    
    # add the dimensions to the dictionary
    parms = {}
    parms['W'] = df_in['W'][row]
    parms['L'] = df_in['L'][row]
    parms['B'] = df_in['B'][row]
    parms['E'] = df_in['E'][row]
    parms['nu'] = df_in['nu'][row]
    
    data_dict['parms']=parms
    data_dict['Description']=Description

    return data_dict, header, df


def solve_KIC(data_dict, win_size):
    soln_dict = deepcopy(data_dict)
    parms = soln_dict['parms']
    beta = parms['L']/parms['W']
    
    ngroups = soln_dict['ngroups']
    d0=np.zeros(ngroups)
    a = np.zeros(ngroups)
    
    C=np.zeros(ngroups)
    for g in soln_dict['detail'].keys():
        d = soln_dict['detail'][g]['load'][0]['d']
        P = soln_dict['detail'][g]['load'][0]['P']
        slope, intercept = calculate_slope(d,P)
        d0[g] = -1*intercept/slope
        d = d-d0[g]
        C[g] = 1/slope

        a[g] = parms['W']*calc_alpha(C[g], beta, parms['B'], parms['E'])
    
    
    soln_dict['summary']['C'] = C
    soln_dict['summary']['a'] = a
    cycle = np.asarray(soln_dict['summary']['cycle'][1:])

    a=a[1:]

    try:
        a_filter = savgol_filter(a, 7, 5)

    except:
        a_filter = a
        
    C=C[1:]

    # Calculate the number of derivative data points
    num_derivative_points = len(a) - win_size + 1

    # Initialize an array to store the derivatives
    derivatives = np.zeros(num_derivative_points)
    alpha = np.zeros(num_derivative_points)

    # Calculate the derivatives using the central difference method
    for i in range(num_derivative_points):
        dy = a_filter[-i - 1] - a_filter[-i-win_size]
        a_mean = np.mean(a_filter[-i-win_size : -i-1])
        dx = cycle[-i - 1] - cycle[-i-win_size]
        derivatives[-i-1] = dy / dx
        alpha[-i-1] = a_mean/parms['W']

    P_max = np.mean(soln_dict['summary']['Pmax'])
    P_min = np.mean(soln_dict['summary']['Pmin'])
    Kmax = calc_K(alpha, beta, parms['B'], P_max, parms['W'], parms['nu'])
    Kmin = calc_K(alpha, beta, parms['B'], P_min, parms['W'], parms['nu'])
    delK = Kmax-Kmin
    Gmax = ((1-parms['nu']**2)*(Kmax**2))/parms['E']
    
    soln_dict['a_filter'] = a_filter
    soln_dict['a'] = a
    soln_dict['C'] = C
    soln_dict['cycle'] = cycle
    soln_dict['delK'] = delK
    soln_dict['Kmax'] = Kmax
    soln_dict['Kmin'] = Kmin
    soln_dict['da/dN'] = derivatives
    soln_dict['Gmax'] = Gmax
    soln_dict['slope'] = slope
    
    return soln_dict

def calc_a(data_dict):
    """
    Calculate crack length for 
    Parameters
    ----------
    data_dict : Dictionary
        DESCRIPTION.

    Returns
    -------
    None.

    """

def calculate_slope(x, y, low_strain_threshold=0.03):
    # Filter the data for low strain values
    low_strain_indices = x < low_strain_threshold
    
    # Perform linear regression on the low strain data
    slope, intercept = np.polyfit(x[low_strain_indices], 
                                  y[low_strain_indices], 1)
    
    return slope, intercept

        
def make_intermediate_plots_KIC(data_dict, win_size):
        delK = data_dict['delK']
        a = data_dict['a']
        cycle = data_dict['cycle']
        dadn = data_dict['da/dN']
        #Make the intermediate plots
        fig, ax = plt.subplots(2,2, figsize = (8,6), constrained_layout = True,
                               num = 'intermediate plots')
        ax[0,0].plot(cycle,a*1000,'o',fillstyle='none')
    
        if win_size == 2:
            ax[1,0].semilogy(cycle[win_size//2:],dadn*1e9,'o',fillstyle='none')
            ax[0,1].plot(cycle[win_size//2:],delK/1e6,'o',fillstyle='none')
        else: 
            ax[1,0].semilogy(cycle[win_size//2:-win_size//2+1],
                             dadn*1e9,'o',fillstyle='none')
            ax[0,1].plot(cycle[win_size//2:-win_size//2+1],
                         delK/1e6,'o',fillstyle='none')
        ax[1,1].loglog(delK/1e6,dadn*1e9,'o',fillstyle='none')
        import matplotlib.ticker as ticker
        ax[1,1].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax[1,1].xaxis.set_minor_locator(ticker.MaxNLocator(4))
        
        ax[0,0].set_xlabel(r'Cycle')
        ax[0,0].set_ylabel(r'a (mm)')
        ax[0,0].set_title('(a)')
        ax[1,0].set_xlabel(r'Cycle')
        ax[1,0].set_ylabel(r'da/dn (nm/cycle)')
        ax[1,0].set_title('(c)')
        ax[0,1].set_xlabel(r'Cycle')
        ax[0,1].set_ylabel(r'$\Delta$K (MPa/$\sqrt{m}$)')
        ax[0,1].set_title('(b)')
        ax[1,1].set_ylabel(r'da/dn (nm/cycle)')
        ax[1,1].set_xlabel(r'$\Delta$K (MPa/$\sqrt{m}$)')
        ax[1,1].xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax[1,1].set_title('(d)')
        plt.savefig("../figures/intermediate/"+str(data_dict['filename'])+".pdf")
        
def make_intermediate_plots_pureshear(data_dict, win_size):
    a=6# placehlder



def powlaw(x, a, b) :
    return a * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata) :
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    
    # Iterate through the array and remove the Nan values
    for i in range(len(ydata_log)-1):
        if np.isnan(ydata_log[i]):
            # Replace the current NaN value with the next non-NaN value
            if ydata_log[i+1] is not None:
                ydata_log[i] = ydata_log[i+1]
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)

    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)






