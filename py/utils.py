"""
This module consists of the following functions:

    1. get_absolute_magnitude (magnitude, redshift)
    2. Flux_to_Luminosity(flux, redshift, flux_error = None)
    3. sigma_to_fwhm(sigma)
    4. calculate_bh_masses(ha_lum, ha_fwhm, epsilon = 1.)
    5. tractor_flux_to_mag (table, band)
    6. tractor_flux_to_magnitude (flux, flux_ivar)
    
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

#from cosmoprimo.fiducial import DESI

####################################################################################################
####################################################################################################

def get_absolute_magnitude(magnitude, redshift):
    """
    Convert Apparent magnitude to Absolute magnitude.
    Uses Planck 2018 cosmology -- Fiducial DESI Cosmology.
    
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
    #cosmo = DESI()
    
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
    ## DESI Cosmology
    #cosmo = DESI()
    
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

def sigma_to_fwhm(sigma, sigma_err = None):
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
    
    if (sigma_err is not None):
        fwhm_err = 2*np.sqrt(2*np.log(2))*sigma_err
        return (fwhm, fwhm_err)
    else:
        return (fwhm)

####################################################################################################
####################################################################################################

def calculate_bh_masses(ha_lum, ha_fwhm, ha_lum_err = None, ha_fwhm_err = None, epsilon = 1.):
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
    
    if ((ha_lum_err is not None)&(ha_fwhm_err is not None)):
        ## Error calculation
        # term1 = 0.204*(ha_lum_err/ha_lum)
        # term2 = 0.895*(ha_fwhm_err/ha_fwhm)
        
        term1 = 0.47*((np.log10((ha_lum + ha_lum_err)/1e+42)) - ((np.log10((ha_lum - ha_lum_err)/1e+42))))/2
        term2 = 2.06*((np.log10((ha_fwhm + ha_fwhm_err)/1e+42)) - ((np.log10((ha_fwhm - ha_fwhm_err)/1e+42))))/2
        
        log_mbh_err = np.sqrt((term1**2) + (term2**2))

        return (log_mbh, log_mbh_err)
    else:
        return (log_mbh)

####################################################################################################
####################################################################################################

def tractor_flux_to_mag(table, band):
    """
    Convert Flux to Magnitude from the LS Tractor Catalog.
    
    Parameters
    ----------
    table : Astropy Table
        Table containing the columns from LS Tractor Catalog
    band : str
        Filter whose flux needs to be converted to magnitude values
        
    Returns
    --------
    mag : array
        Reddening corrected magnitudes for the particular band
    snr : array
        Signal-to-noise ratio for the particular band
    """
    
    flux = table['FLUX_'+band]
    flux_ivar = table['FLUX_IVAR_'+band]
    snr = flux*np.sqrt(flux_ivar)
    dered_flux = flux/table['MW_TRANSMISSION_'+band]
    
    mag = -2.5*np.log10(dered_flux) + 22.5
    
    return (mag, snr)

####################################################################################################
####################################################################################################

def tractor_flux_to_magnitude(flux, flux_ivar):
    """
    Convert Flux to Magnitude for LS Tractor Flux Values
    
    Parameters
    ----------
    flux : array
        Array of Flux Values
    flux_ivar : array
        Array of Inverse Variance Values
        
    Returns
    -------
    mag : array
        Array of Magnitude Values
    mag_err : array
        Array of Magnitude Error Values
    """
    
    snr = flux*np.sqrt(flux_ivar)
    mag = 22.5 - (2.5*np.log10(flux))
    mag_err = 2.5*np.log10(1+(1/snr))
    
    return (mag, mag_err) 
    
####################################################################################################
####################################################################################################

def r_kcorr(gr,z):
    '''
    This function returns the k correction for SDSS r band 
    
    According to the Chilingarian et al. 2010 from which this k-correction is based, 
    we can only apply this on z<0.5
    
    Function from Viraj Manwadkar
    
    '''
    
    # it is power of z * power of gr
    coeff_10 = -1.61294 * (z**1) * (gr**0)
    coeff_11 = 3.81378  * (z**1) * (gr**1)
    coeff_12 = -3.56114  * (z**1) * (gr**2)
    coeff_13 = 2.47133  * (z**1) * (gr**3)
    
    coeff_20 = 9.13285  * (z**2) * (gr**0)
    coeff_21 = 9.85141 * (z**2) * (gr**1)
    coeff_22 = -5.1432 * (z**2) * (gr**2)
    coeff_23 = -7.02213 * (z**2) * (gr**3)
    
    coeff_30 = -81.8341 * (z**3) * (gr**0)
    coeff_31 = -30.3631 * (z**3) * (gr**1)
    coeff_32 = 38.5052 * (z**3) * (gr**2)
    
    coeff_40 = 250.732 * (z**4) * (gr**0)
    coeff_41 = -25.0159 * (z**4) * (gr**1)
    
    coeff_50 = -215.377 * (z**5) * (gr**0)

    kr =  (coeff_10 + coeff_11 + coeff_12 + coeff_13) + \
    (coeff_20 + coeff_21 + coeff_22 + coeff_23) + \
    (coeff_30 + coeff_31 + coeff_32) + (coeff_40 + coeff_41) + (coeff_50)
    
    return (kr)

####################################################################################################
####################################################################################################
    
def get_stellar_mass(gr,rmag,zred):
    '''
    Computes the stellar mass of object using the SAGA 2 conversion
    
    It is given by Log10(Mstar) = 1.254 + 1.098*(g-r) - 0.4 M_r
    
    We would need to get the absolute r band mag above.
    We will also have to a K correction to the absolute magnitude
    
    Function from Viraj Manwadkar
    
    '''
    
    #convert the zred to the luminosity distance 
    #d = Planck18.luminosity_distance(zred)
    d = cosmo.luminosity_distance(zred)
    d_in_pc = d.value * 1e6
    
    #M = m + 5 - 5*log10(d/pc) - Kcor
    kr = r_kcorr(gr,zred)
    M_r = rmag + 5 - 5*np.log10(d_in_pc) - kr
    
    log_mstar = 1.254 + 1.098*gr - 0.4*M_r
    
    return (log_mstar)

####################################################################################################
####################################################################################################