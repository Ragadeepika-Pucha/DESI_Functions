"""
This module consists of the following functions:

    1. targetid_to_lsid (targetid)
    2. tractor_flux_to_mag (table, band)
    3. tractor_flux_to_magnitude (flux, flux_ivar)
    4. get_absolute_magnitude (magnitude, redshift)
    5. combine_observations (table, array_name, obj_num)
    
Author : Ragadeepika Pucha
Version : 2022 June 21
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
from dl import queryClient as qc

####################################################################################################
####################################################################################################

def targetid_to_lsid(targetid):
    """
    Function to convert TARGETID to LS ID
    
    Parameters 
    ----------
    targetid : int64
        Unique DESI target identifier
        
    Returns
    -------
    ls_id : int64
        Unique identifier for Legacy Survey objects
    """
    objid, brickid, release, _, _, _ = decode_targetid(targetid)
    ls_id = (release<<40)|(brickid<<16)|(objid)
    
    return(ls_id)

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

def combine_observations(table, array_name, obj_num):
    
    ## Find another place to put this function
    """
    Combine all the different measurements of a particular object and return as an array.
    This is for studying light curves.
    
    Parameters
    ----------
    table : astropy table
        Table with multiple instances of data
    
    array_name : str
        Name of the column for which we have multiple data
        
    obj_num : int
        Index of the object in the table
        
    Returns
    -------
    arr : array
        Array of combined measurements
    
    """
    num = np.arange(1, 16)
    
    arr = np.array([])
    
    for n in num:
        arr = np.append(arr, table[f'{array_name}_{n}'].data[obj_num])
    
    return (arr)

####################################################################################################
####################################################################################################