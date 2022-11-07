"""
This module consists of the following functions:

    1. get_absolute_magnitude (magnitude, redshift)
    2. Flux_to_Luminosity(flux, redshift, flux_error = None)
    3. sigma_to_fwhm(sigma)
    4. calculate_bh_masses(ha_lum, ha_fwhm, epsilon = 1.)
    
Author : Ragadeepika Pucha
Version : 2022 August 15
"""

####################################################################################################
####################################################################################################

import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
import astropy.units as u
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

def sigma_to_fwhm(sigma, sigma_err):
    """
    Calculate FWHM of an emission line from sigma values.
    FWHM = 2*sqrt(2*log(2))*sigma
    
    Parameters
    ----------
    sigma : array
        Array of sigma values
        
    sigma_err : array
        Array of sigma error values
        
    Returns
    -------
    fwhm : array
        Array of FWHM values
        
    fwhm_err : array
        Array of FWHM error values
    """
    
    fwhm = 2*np.sqrt(2*np.log(2))*sigma
    fwhm_err = 2*np.sqrt(2*np.log(2))*sigma_err
    
    return (fwhm, fwhm_err)

####################################################################################################
####################################################################################################

def calculate_bh_masses(ha_lum, ha_lum_err, ha_fwhm, ha_fwhm_err, epsilon = 1.):
    """
    Calculate Black Holes masses and errors using the broad H-alpha emission line.
    Equation (5) from Reines+2013 - https://ui.adsabs.harvard.edu/abs/2013ApJ...775..116R/abstract
    
    Parameters
    ----------
    ha_lum : array
        Array of broad Ha luminosity values
        
    ha_lum_err : array
        Array of broad Ha luminosity error values
        
    ha_fwhm : array
        Array of broad Ha FWHM values
        
    ha_fwhm_err : array
        Array of broad Ha FWHM error values
        
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
    
    ## Error calculation
    term1 = 0.204*(ha_lum_err/ha_lum)
    term2 = 0.895*(ha_fwhm_err/ha_fwhm)
    
    log_mbh_err = term1 + term2
    
    return (log_mbh, log_mbh_err)

####################################################################################################
####################################################################################################