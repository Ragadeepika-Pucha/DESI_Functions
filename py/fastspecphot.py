"""
This script contains functions related to fastspec and fastphot catalogs.
The module consists of following functions
    
    1. combine_fastspec_fastphot(h_fspec, h_fphot)
    2. get_fastspec_columns(table, em_lines, aon = True, snr = False, add = False)

Author: Ragadeepika Pucha
Version: 2022 August 15
"""

###################################################################################################
###################################################################################################
import numpy as np

from astropy.io import fits
from astropy.table import Table, Column, join

###################################################################################################
###################################################################################################

def combine_fastspec_fastphot(h_fspec, h_fphot):
    """
    This function selects specific columns from fastspec and fastphot catalog and then join them
    to return a single table.
    
    Parameters
    ----------
    h_fspec : Astropy hdulist
        HDUList of the fastspec catalog. It has the format of any fastspec catalog
        
    h_fphot : Astropy hdulist
        HDUList of the fastphot catalog. It has the format of any fastphot catalog
        
    Returns
    -------
    t_fast : Astropy table
        Combined fastspec+fastphot catalog with required columns
        
    Note: Can be changed later to include more user-given columns
    """
    ## Fastspec tables and meta-data
    fspec = Table(h_fspec[1].data)
    fspec_meta = Table(h_fspec[2].data)
    
    ## Required columns
    fspec_cols = ['TARGETID', 'SURVEY', 'PROGRAM','HEALPIX',
                  'RCHI2','LINERCHI2_BROAD','DELTA_LINERCHI2',
                  'NARROW_Z', 'NARROW_SIGMA','OII_DOUBLET_RATIO','SII_DOUBLET_RATIO',
                  'HBETA_AMP','HBETA_AMP_IVAR', 'HBETA_EW', 'HBETA_EW_IVAR',
                  'HBETA_FLUX','HBETA_FLUX_IVAR','HBETA_SIGMA',
                  'HBETA_BROAD_AMP','HBETA_BROAD_AMP_IVAR', 
                  'HBETA_BROAD_EW', 'HBETA_BROAD_EW_IVAR',
                  'HBETA_BROAD_FLUX','HBETA_BROAD_FLUX_IVAR','HBETA_BROAD_SIGMA',
                  'OIII_4959_AMP','OIII_4959_AMP_IVAR', 'OIII_4959_EW', 'OIII_4959_EW_IVAR',
                  'OIII_4959_FLUX','OIII_4959_FLUX_IVAR','OIII_4959_SIGMA',
                  'OIII_5007_AMP','OIII_5007_AMP_IVAR', 'OIII_5007_EW', 'OIII_5007_EW_IVAR',
                  'OIII_5007_FLUX','OIII_5007_FLUX_IVAR','OIII_5007_SIGMA',
                  'OI_6300_AMP','OI_6300_AMP_IVAR', 'OI_6300_EW', 'OI_6300_EW_IVAR',
                  'OI_6300_FLUX','OI_6300_FLUX_IVAR','OI_6300_SIGMA',
                  'NII_6548_AMP','NII_6548_AMP_IVAR', 'NII_6548_EW', 'NII_6548_EW_IVAR',
                  'NII_6548_FLUX','NII_6548_FLUX_IVAR','NII_6548_SIGMA',
                  'HALPHA_AMP','HALPHA_AMP_IVAR', 'HALPHA_EW', 'HALPHA_EW_IVAR',
                  'HALPHA_FLUX','HALPHA_FLUX_IVAR','HALPHA_SIGMA',
                  'HALPHA_BROAD_AMP','HALPHA_BROAD_AMP_IVAR',
                  'HALPHA_BROAD_EW', 'HALPHA_BROAD_EW_IVAR',
                  'HALPHA_BROAD_FLUX','HALPHA_BROAD_FLUX_IVAR','HALPHA_BROAD_SIGMA',
                  'NII_6584_AMP','NII_6584_AMP_IVAR', 'NII_6584_EW', 'NII_6584_EW_IVAR',
                  'NII_6584_FLUX','NII_6584_FLUX_IVAR', 'NII_6584_SIGMA',
                  'SII_6716_AMP','SII_6716_AMP_IVAR', 'SII_6716_EW', 'SII_6716_EW_IVAR',
                  'SII_6716_FLUX','SII_6716_FLUX_IVAR', 'SII_6716_SIGMA', 
                  'SII_6731_AMP','SII_6731_AMP_IVAR', 'SII_6731_EW', 'SII_6731_EW_IVAR',
                  'SII_6731_FLUX','SII_6731_FLUX_IVAR', 'SII_6731_SIGMA']

    fspec_meta_cols = ['TARGETID', 'SURVEY', 'PROGRAM','HEALPIX', 'RA', 'DEC', 
                       'COADD_FIBERSTATUS', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'Z_RR', 'LS_ID']
    
    
    ## JOIN fastspec and meta data tables
    fspec_sel = fspec[fspec_cols]
    fspec_meta_sel = fspec_meta[fspec_meta_cols]
    fastspec = join(fspec_sel, fspec_meta_sel,\
                    keys = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX'])
    
    ## Fastphot table
    ## Meta tables are same as fastspec - so, we don't need it
    fphot = Table(h_fphot[1].data)
    
    ## Required columns
    fphot_cols = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                  'ABSMAG_SDSS_Z', 'ABSMAG_IVAR_SDSS_Z', 'LOGMSTAR', 'LOGL_5100']

    fastphot = fphot[fphot_cols]
    
    ## Join fastspec and fastphot
    t_fast = join(fastphot, fastspec, \
                  keys = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX'])
    
    return (t_fast)

###################################################################################################
###################################################################################################

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
    
    
    
    
    
    

    
    