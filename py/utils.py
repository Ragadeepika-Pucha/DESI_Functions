"""
This module consists of the following functions:

    1. get_absolute_magnitude (magnitude, redshift)
    2. get_fastspec_columns(table, em_lines, aon = True, snr = False, add = False)
    3. Flux_to_Luminosity(flux, redshift, flux_error = None)
    4. sigma_to_fwhm(sigma)
    
Author : Ragadeepika Pucha
Version : 2022 August 4
"""

####################################################################################################
####################################################################################################

import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
import fitsio

import desispec.io
from desispec import coaddition
from desitarget.targets import decode_targetid

####################################################################################################
####################################################################################################

def get_absolute_magnitude(magnitude, redshift):
    """
    Convert Apparent magnitude to Absolute magnitude.
    Uses WMAP9 cosmology.
    
    Parameters
    ----------
    magnitude : array
        Array of magnitudes that need to be converted    
    redshift : array
        Array of redshifts
    
    Returns
    -------
    Mag : array
        Array of absolute magnitudes for the input sources
    """
    dl = cosmo.luminosity_distance(redshift)
    Mag = magnitude - (5*np.log10(dl.value*1e+5))
    
    return (Mag)

####################################################################################################
####################################################################################################

def get_fastspec_columns(table, em_lines, aon = True, snr = False, add = False):
    """
    Function to get flux and AoN/SNR outputs from Fastspecfit catalog.
    
    Parameters
    ----------
    table : Astropy Table
        Table consisting of fastspecfit catalogs
        
    em_lines : str or list
        Emission lines whose output is required.
        If add = False, it is a single str: with fastspec column format.
        If add = True, the output flux is sum of lines and list of lines is inputted.
        
    aon : bool
        Whether or not AoN is required as output. Default is True
        
    snr : bool
        Whether or not SNR is required as output. Default is False
        
    add : bool
        Whether or not the fluxes of lines need to be added. 
        Useful for [SII], for example. The AoN or SNR is similarly corrected for sum of lines.
        Default is False
        
    Returns
    -------
    
    flux : array
        Array of flux values for the input list of sources
        
    flux_aon : array
        Only returned when aon = True. 
        Array of AoN values for the input list of sources
        
    flux_snr : array
        Only returned when snr = True.
        Array of AoN values for the input list of sources
    """
    
    if (aon):
        ## If AoN = True, then AoN is computed and returned
        if (add == False):
            ## If add = False, then the flux and AoN of a single emission line is returned.
            flux = table[f'{em_lines}_FLUX'].data
            flux_aon = table[f'{em_lines}_AMP'].data*np.sqrt(table[f'{em_lines}_AMP_IVAR'].data)
        else:
            ## If add = True, then flux of two emission lines is added.
            ## The AoN is corrected for the sum of the lines.
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
            amp = table[f'{em_lines[0]}_AMP'].data+table[f'{em_lines[1]}_AMP'].data
            amp_noise = np.sqrt((1/tab[f'{em_lines[0]}_AMP_IVAR'])+\
                                (1/tab[f'{em_lines[1]}_AMP_IVAR']))
            flux_aon = amp/amp_noise
        ## Returns flux and AoN when aon = True    
        return (flux, flux_aon)
    
    elif (snr):
        ## If SNR = True, then SNR is computed and returned
        if (add == False):
            ## If add = False, then the flux and SNR of a single emission line is returned.
            flux = table[f'{em_lines}_FLUX'].data
            flux_snr = table[f'{em_lines}_FLUX'].data*np.sqrt(table[f'{em_lines}_FLUX_IVAR'].data)
        else:
            ## If add = True, then flux of two emission lines is added.
            ## The SNR is corrected for the sum of the lines.
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
            flux_noise = np.sqrt((1/tab[f'{em_lines[0]}_FLUX_IVAR'])+\
                                 (1/tab[f'{em_lines[1]}_FLUX_IVAR']))
            flux_snr = flux/flux_noise
        ## Returns flux and SNR when snr = True
        return (flux, flux_snr)
    
    else:
        ## If both aon = False and snr = False, only flux is computed and returned
        if (add == False):
            flux = table[f'{em_lines}_FLUX'].data
        else:
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
        return (flux)

####################################################################################################
####################################################################################################

def Flux_to_Luminosity(flux, redshift, flux_error = None):
    """
    Convert flux to luminosity.
    Uses WMAP9 cosmology
    
    Parameters
    ----------
    flux : array
        Array of flux values that needs to be converted.
        
    redshift : array
        Array of redshift values of the sources.
        
    flux_error : array
        Array of flux error values. If None, then only luminosity values are returned.
        If array is given, luminosity error values are also returned.
        
    Returns
    -------
    lum : array
        Array of luminosity values for the sources
        
    lum_error : array
        Only returned when flux_error is not None.
        Array of luminosity error values for the sources
    """
    
    ## Compute luminosity distance
    dl = cosmo.luminosity_distance(redshift)
    dl = dl.to(u.centimeter)    # Convert to cm as flux is in ergs/s/cm^2
    
    # Luminosity = Flux*(4*pi*d^2)
    lum = flux*(4*np.pi)*(dl.value**2)
    
    ## If Flux_error is not None, then luminosity error is computed and returned.
    ## If Flux_error is None, then only luminosity values are returned.
    if (flux_error is not None):
        lum_error = flux_error*(4*np.pi)*(dl.value**2)
        return (lum, lum_error)
    else:
        return (lum)
    
####################################################################################################
####################################################################################################

def sigma_to_fwhm(sigma):
    """
    Calculate FWHM of an emission line from sigma values.
    FWHM = 2*sqrt(2*log(2))*sigma
    
    Parameters
    ----------
    sigma : array
        Array of sigma values
        
    Returns
    -------
    fwhm : array
        Array of FWHM values
    """
    
    fwhm = 2*np.sqrt(2*np.log(2))*sigma
    
    return (fwhm)

####################################################################################################
####################################################################################################

def calculate_bh_masses(ha_lum, ha_fwhm, epsilon = 1.):
    """
    Calculate Black Holes masses using the broad H-alpha emission line.
    Equation (5) from Reines+2013 - https://ui.adsabs.harvard.edu/abs/2013ApJ...775..116R/abstract
    
    Parameters
    ----------
    ha_lum : array
        Array of broad Ha luminosity values
        
    ha_fwhm : array
        Array of broad Ha FWHM values
        
    epsilon : float
        Value of scale factor (can take values from 0.75-1.4)
        e.g., Onken et al. 2004; Greene & Ho 2007a; Grier et al. 2013
        Default = 1.0
        
    Returns
    -------
    log_mbh : array
        Array of log(M_BH/M_sun) values for the input sources.
    """
    ## The three terms that are part of the equation
    term1 = np.log10(epsilon) + 6.57
    term2 = 0.47*np.log10(ha_lum/1e+42)
    term3 = 2.06*np.log10(ha_fwhm/1e+3)
    
    ## Log of BH mass
    log_mbh = term1 + term2 + term3 
    
    return (log_mbh)

####################################################################################################
####################################################################################################