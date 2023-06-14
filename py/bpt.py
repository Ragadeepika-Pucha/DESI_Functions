"""
This script consists of functions related to plotting and using BPT Diagrams.
It consists of the following functions:
    1) Kauffman_02_NII(xx)
    2) Kewley_01_NII(xx)
    3) Schawinsky_07_NII(xx)
    4) Kewley_01_SII(xx)
    5) Kewley_06_SII(xx)
    6) Kewley_01_OI(xx)
    7) Kewley_06_OI(xx)
    8) Plot_NII_BPT_Lines(axs, Kauffman, Kewley, Schawinsky, limits)
    9) Plot_SII_BPT_Lines(axs, limits)
    10) Plot_OI_BPT_Lines(axs, limits)
    11) Classify_NII_BPT(nii_ha, oiii_hb)
    12) Classify_SII_BPT(sii_ha, oiii_hb)
    13) Classify_OI_BPT(oi_ha, oiii_hb)
"""

####################################################################################################
####################################################################################################

import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

####################################################################################################
####################################################################################################

def Kauffman_03_NII(xx):
    """
    Function that outputs the Kauffman line values for given log([NII]/Ha) values.
     
    This can be used both for plotting the separation lines on the BPT Diagram and
    also for separating SF-dominated galaxies.
    
    Parameters
    ----------
    xx : float
        Input values of log([NII]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of 0.61/(log([NII]/Ha) - 0.05) + 1.3 for the input value of log([NII]/Ha)
    
    """
    # From Kauffman et. al. 2003
    yy = (0.61/(xx - 0.05)) + 1.3
    
    return (yy)

####################################################################################################
####################################################################################################

def Kewley_01_NII(xx):
    """
    Function that outputs the Kewley line values for given log([NII]/Ha) values.
    This can be used both for plotting the separation lines on the BPT Diagram and
    also for separating AGN-dominated galaxies.
    
    Parameters
    ----------
    xx : float
        Input values of log([NII]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of 0.61/(log([NII]/Ha) - 0.47) + 1.19 for the input value of [NII]/Ha 
    
    """
    # From Kewley  et. al. 2001
    yy = (0.61/(xx - 0.47)) + 1.19
    
    return (yy)

####################################################################################################
####################################################################################################

def Schawinsky_07_NII(xx):
    """
    Function that outputs the Schawinsky line values for given log([NII]/Ha) values.
    This separates the Seyferts from LINERS in [NII]-BPT Diagram.
    This can be used both for plotting the separate lines on the BPT Diagram and 
    for separating the sources.
    
    Parameters
    ----------
    xx : float
        Input values of log([NII]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of (1.05*log(NII/Ha))+0.45 for the input value of [NII]/Ha 
        
    """
    # From Schawinsky et al. 2007
    yy = (1.05*xx)+0.45
    
    return (yy)

####################################################################################################
####################################################################################################

def Kewley_01_SII(xx):
    """
    Function that outputs the Kewley line values for given log([SII]/Ha) values.
    This can be used both for plotting the separate lines on the BPT Diagram and 
    for separating AGN galaxies from SII-BPT diagram.
    
    Parameters
    ----------
    xx : float
        Input values of log([SII]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of 0.72/(log([SII/Ha]) - 0.32) + 1.30 for the input value of [SII]/Ha

    """
    # From Kewley et. al. 2001
    yy = (0.72/(xx - 0.32)) + 1.30
    
    return (yy)

####################################################################################################
####################################################################################################

def Kewley_06_SII(xx):
    """
    Function that outputs the Kewley+06 line values for given log([SII]/Ha) values.
    This can be used for both plotting the separation lines on the BPT Diagram and
    also for separating LINERs from Seyferts from SII-BPT diagram.
    
    Parameters
    ----------
    xx : float
        Input values of log([SII]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of (1.89*log([SII]/Ha)) + 0.76 for the input value of [SII]/Ha
    """
    
    # From Kewley et al. 2006
    yy = (1.89*xx)+0.76
    
    return (yy)

####################################################################################################
####################################################################################################

def Kewley_01_OI(xx):
    """
    Function that outputs the Kewley+01 line values for given log([OI]/Ha) values.
    This can be used for both plotting the separation lines on the BPT Diagram and 
    also for separating AGN from OI-BPT diagram.
    
    Parameters
    ----------
    xx : float
        Input values of log([OI]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of (0.73/(xx+0.59))+1.33 for the input value of [OI]/Ha    
    """
    
    # From Kewley et al. 2001
    yy = (0.73/(xx+0.59))+1.33
    
    return (yy)

####################################################################################################
####################################################################################################

def Kewley_06_OI(xx):
    """
    Function that outputs the Kewley+06 line values for given log([OI]/Ha) values.
    This can be used for both plotting the separation lines on the BPT Diagram and 
    also for separating LINERs from Seyferts from OI-BPT diagram.
    
    Parameters
    ----------
    xx : float
        Input values of log([OI]/Ha)
        
    Returns
    -------
    yy : float
        Outputs the value of (1.18*xx) + 1.30 for the input value of [OI]/Ha
    """
    
    # From Kewley et al. 2006
    yy = (1.18*xx) + 1.30
    
    return (yy)

####################################################################################################
####################################################################################################

def Plot_NII_BPT_Lines(axs = None, Kauffman = True, Kewley = True,\
                       Schawinsky = True, limits = True, legend = False):
    """
    Function to plot [NII]-BPT Lines
    
    Parameters
    ----------
    ax : AxesSubplot
        The axis onto which the [NII]-BPT lines will be plotted.
        If ax = None, then the plotting function uses plt, rather than axis.
        
    Kauffman : bool
        Whether or not to plot the Kauffman+03 [NII]-BPT line. Default is True.
    
    Kewley : bool
        Whether or not to plot the Kewley+01 [NII]-BPT line. Default is True.
        
    Schawinsky : bool
        Whether or not to plot the Schawinsky+07 [NII]-BPT line. Default is True.
        
    limits : bool
        Whether or not to limit x- and y-range of the plot. Default is True.
        
    Returns
    -------
    None.
    
    """
    
    if (axs == None):
        axs = plt.gca()
        
    if (Kauffman == True):
        xx = np.linspace(-2.5, 0.05, 1000)
        Ka03 = Kauffman_03_NII(xx)
        
        axs.plot(xx, Ka03, color = 'white', lw = 6.0)
        axs.plot(xx, Ka03, color = 'k', lw = 3.0, label = 'Kauffmann+03')
        
    if (Kewley == True):
        xx = np.linspace(-2.5, 0.47, 1000)
        Ke01 = Kewley_01_NII(xx)
        
        axs.plot(xx, Ke01, color = 'white', lw = 6.0)
        axs.plot(xx, Ke01, color = 'k', ls = '--', lw = 3.0, label = 'Kewley+01')
        
    if (Schawinsky == True):
        xx = np.linspace(-0.18, 1.0, 1000)
        Scha07 = Schawinsky_07_NII(xx)
        
        axs.plot(xx, Scha07, color = 'white', lw = 6.0)
        axs.plot(xx, Scha07, color = 'k', ls = ':', lw = 3.0, label = 'Schawinsky+07')
        
    if (limits == True):
        axs.set(xlim = [-2.5, 1.0], ylim = [-1.6, 1.5])
        
    if (legend == True):
        axs.legend(loc = 'lower left', fontsize = 16)
        
####################################################################################################
####################################################################################################

def Plot_SII_BPT_Lines(axs = None, limits = True):
    """
    Function to plot [SII]-BPT Lines
    
    Parameters
    ----------
    ax : AxesSubplot
        The axis onto which the [SII]-BPT lines will be plotted.
        If ax = None, then the plotting function uses plt, rather than axis.
        
    limits : bool
        Whether or not to limit x- and y-range of the plot. Default is True.
        
    Returns
    -------
    None.
    
    """
    
    if (axs == None):
        axs = plt.gca()
        
    xx = np.linspace(-2.5, 0.1, 1000)
    axs.plot(xx, Kewley_01_SII(xx), color = 'white', lw = 6.0)
    axs.plot(xx, Kewley_01_SII(xx), color = 'k', lw = 3.0)
    
    xx = np.linspace(-0.3, 0.5, 1000)
    axs.plot(xx, Kewley_06_SII(xx), color = 'white', lw = 6.0)
    axs.plot(xx, Kewley_06_SII(xx), color = 'k', lw = 3.0, ls = '--')
    
    if (limits == True):
        axs.set(xlim = [-1.5, 0.5], ylim = [-1.6, 1.5])
        
####################################################################################################
####################################################################################################

def Plot_OI_BPT_Lines(axs = None, limits = True):
    """
    Function to plot [OI]-BPT Lines
    
    Parameters
    ----------
    ax : AxesSubplot
        The axis onto which the [OI]-BPT lines will be plotted.
        If ax = None, then the plotting function uses plt, rather than axis.
        
    limits : bool
        Whether or not to limit x- and y-range of the plot. Default is True.
        
    Returns
    -------
    None.
    
    """
    
    if (axs == None):
        axs = plt.gca()
        
    xx = np.linspace(-2.5, -0.8, 1000)
    axs.plot(xx, Kewley_01_OI(xx), color = 'white', lw = 6.0)
    axs.plot(xx, Kewley_01_OI(xx), color = 'k', lw = 3.0)
    
    xx = np.linspace(-1.1, 0.5, 1000)
    axs.plot(xx, Kewley_06_OI(xx), color = 'white', lw = 6.0)
    axs.plot(xx, Kewley_06_OI(xx), color = 'k', lw = 3.0, ls = '--')
    
    if (limits == True):
        axs.set(xlim = [-2.5, 0.5], ylim = [-1.6, 1.5])
        
####################################################################################################
####################################################################################################

def Classify_NII_BPT(nii_ha, oiii_hb):
    """
    Classify sources into AGN, SF, and Composites using the [NII]-BPT diagram.
    
    Parameters 
    ----------
    nii_ha : float
        log ([NII]/Ha) values
        
    oiii_hb : float
        log ([OIII]/Hb) values
        
    Returns
    -------
    is_agn : numpy array
        Boolean array where AGN = True
        
    is_comp : numpy array
        Boolean array where composite = True
        
    is_sf : numpy array
        Boolean array where SF = True
    
    """
    
    Ka_03_vals = Kauffman_03_NII(nii_ha)
    Ke_01_vals = Kewley_01_NII(nii_ha)
    
    is_agn = ((oiii_hb >= Ke_01_vals)|(nii_ha >= 0.47))
    is_comp = ((oiii_hb >= Ka_03_vals)|(nii_ha >= 0.05))&(~is_agn)
    is_sf = (~is_agn)&(~is_comp)
    
    return (is_agn, is_comp, is_sf)
    
####################################################################################################
####################################################################################################
    
def Classify_SII_BPT(sii_ha, oiii_hb):
    """
    Classify sources into SF, Seyferts, and LINERS based on [SII]-BPT Diagram.
    
    Parameters 
    ----------
    sii_ha : float
        log ([SII]/Ha) values
        
    oiii_hb : float
        log ([OIII]/Hb) values
        
    Returns
    -------
    sf_sii : numpy array
        Boolean array where SF = True.
        
    sy_sii : numpy array
        Boolean array where Seyfert = True.
        
    liner_sii : numpy array
        Boolean array where LINER = True

    """
    Ke_01_vals = Kewley_01_SII(sii_ha)
    Ke_06_vals = Kewley_06_SII(sii_ha)
    
    sy_sii = ((oiii_hb >= Ke_01_vals)|(sii_ha >= 0.32))&(oiii_hb > Ke_06_vals)
    liner_sii = ((oiii_hb >= Ke_01_vals)|(sii_ha >= 0.32))&(oiii_hb < Ke_06_vals)
    sf_sii = (~sy_sii)&(~liner_sii)
    
    return (sf_sii, sy_sii, liner_sii)

####################################################################################################
####################################################################################################

def Classify_OI_BPT(oi_ha, oiii_hb):
    """
    Classify sources into SF, Seyferts, and LINERS based on [OI]-BPT Diagram.
    
    Parameters 
    ----------
    oi_ha : float
        log ([OI]/Ha) values
        
    oiii_hb : float
        log ([OIII]/Hb) values
        
    Returns
    -------
    sf_oi : numpy array
        Boolean array where SF = True.
        
    sy_oi : numpy array
        Boolean array where Seyfert = True.
        
    liner_oi : numpy array
        Boolean array where LINER = True
        
    """
    Ke_01_vals = Kewley_01_OI(oi_ha)
    Ke_06_vals = Kewley_06_OI(oi_ha)
    
    sy_oi = ((oiii_hb >= Ke_01_vals)|(oi_ha >= -0.59))&(oiii_hb > Ke_06_vals)
    liner_oi = ((oiii_hb >= Ke_01_vals)|(oi_ha >= -0.59))&(oiii_hb < Ke_06_vals)
    sf_oi = (~sy_oi)&(~liner_oi)
    
    return (sf_oi, sy_oi, liner_oi)

####################################################################################################
####################################################################################################

    
# def Classify_SII_BPT(sii_ha, oiii_hb):
#     """
#     Classify sources into Seyfert, LINERs, and SF.
#     """
    
#     bpt_agn_vals = BPT_SII_AGN_line(sii_ha)
#     bpt_lin_vals = BPT_SII_LINER_line(sii_ha)
    
#     valid = (sii_ha >= -1.5) & (sii_ha <= 1.0) & (oiii_hb >= -1.5) & (oiii_hb <= 1.5)
#     extremes = (~valid)
    
#     is_sy = (oiii_hb >= bpt_agn_vals) & (oiii_hb >= bpt_lin_vals) & valid
#     is_lin = (oiii_hb >= bpt_agn_vals)&(~is_sy)&(valid) 
#     is_sf = (~is_sy)&(~is_lin)&(valid)
    
#     return (is_sy, is_lin, is_sf)
        
####################################################################################################
####################################################################################################


# def Classify_NII_BPT(nii_ha, oiii_hb):
#     """
#     Classify sources into Star-Forming, Seyferts, LINERS, and Composites based on 
#     [NII]-BPT Diagram
#     """
#     Ka_03_vals = Kauffman_03_NII(nii_ha)
#     Ke_01_vals = Kewley_01_NII(nii_ha)
#     Scha_07_vals = Schawinsky_07_NII(nii_ha)
    
#     sy_nii = ((oiii_hb >= Ke_01_vals)|(nii_ha >= 0.47))&(oiii_hb > Scha_07_vals)
#     liner_nii = ((oiii_hb >= Ke_01_vals)|(nii_ha >= 0.47))&(oiii_hb < Scha_07_vals)
#     composite_nii = ((oiii_hb >= Ka_03_vals)|(nii_ha >= 0.05))&(~sy_nii)
#     sf_nii = (~sy_nii)&(~liner_nii)&(~composite_nii)
    
#     return (sf_nii, sy_nii, liner_nii, composite_nii)

####################################################################################################
####################################################################################################