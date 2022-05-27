import numpy as np

import fitsio
from astropy.table import Table

from dl import specClient as spec

import desispec.io
from desispec import coaddition

###########################################################################################################################################################
###########################################################################################################################################################

def get_desi_spectra(targetid, z, survey = 'sv', rest_frame = True):
    """
    Function to get the DESI spectra of any object given its TARGETID
    
    Returns the arrays of wavelength, flux, and inverse variance.
    Returns the rest-frame arrays if rest_frame = True
    
    Parameters
    ----------
    targetid : int64
        Unique TARGETID of the target
        
    z : float
        Redshift of the source
        
    survey : str
        Survey of the target in DESI - sv or main
    
    rest_frame : bool
        Whether or not to return the rest-frame values. Default is True

    Returns
    -------
    
    lam : array
        Wavelength array (Rest-frame values if rest_frame = True)
        
    flux : array
        Flux values array (Rest-frame values if rest_frame = True)
        
    ivar : array
        Inverse variance array (Rest-frame values if rest_frame = True)
    """
    
    ## Accessing DESI Spectra from DataLab
    ## Location of the spectra
    user = 'raga_steph'
    specred = 'everest'
    specred_dir = f'/dlusers/{user}/DESI/spectro/redux/{specred}'
    hpx_dir = f'{specred_dir}/healpix'
    
    # Accessing the zpix summary catalog
    
    zcat_file = f'/dlusers/raga_steph/AGN_in_Dwarfs/everest/catalogs/zpix-{survey}-summary.fits'
    zcat = Table.read(zcat_file)
    
    # Selecting only primary spectra objects
    zcat_prim = zcat[zcat['SPECPRIMARY']]
    
    # Selecting the row with the inpur targetid
    tgt = (zcat_prim['TARGETID'].data == targetid)
    t_tgt = zcat_prim[tgt]
    
    hpx = t_tgt['HPXPIXEL'].astype(str).data[0]   # Healpix Number
    survey = t_tgt['SURVEY'].astype(str).data[0]   # Survey -- sv1|sv2|sv3
    faprgrm = t_tgt['FAPRGRM'].astype(str).data[0] # dark|bright

    
    # Location of the coadded spectra for the given object
    tgt_dir = f'{hpx_dir}/{survey}/{faprgrm}/{hpx[:-2]}/{hpx}'
    coadd_name = f'{tgt_dir}/coadd-{survey}-{faprgrm}-{hpx}.fits'

    coadd_obj = desispec.io.read_spectra(coadd_name)    # Accessing all the spectra in the given healpix
    tgts = coadd_obj.target_ids().data       

    # Selecting the particular spectra of the targetid
    ii = (tgts == targetid)
    coadd_spec = coadd_obj[ii]

    # Coadding the b,r,z into a single spectra
    res = coaddition.coadd_cameras(coadd_spec)    
    lam = res.wave['brz']                # Wavelength array of the coadded spectra
    flux = res.flux['brz'][0]            # Flux array of the coadded spectra
    ivar = res.ivar['brz'][0]            # Inverse Variance array of the coadded spectra

    if (rest_frame == True):
        # If rest_frame = True, we convert the different arrays into their rest-frame values
        lam = lam/(1+z)
        flux = flux*(1+z)
        ivar = ivar/((1+z)**2)

    return (lam, flux, ivar)
    
###########################################################################################################################################################
###########################################################################################################################################################

def get_sdss_spectra(specobjid,  z, rest_frame = True):
    """
    Function to access the SDSS spectra of any object from the Data Lab file service.
    
    Returns the arrays of wavelength, flux, model and ivar.
    Returns the rest-frame arrays if rest_frame = True.
    If rest_frame = False, returns the observed-frame arrays.
    
    Parameters
    ----------
    specobjid : int64
        Unique SDSS spectrum id
        
    z : float
        Redshift of the source
    
    rest_frame : bool
        Whether or not to return the rest-frame values. Default is True.
    
    Returns
    -------
    wavelength : array
        Wavelength array (Rest-frame values if rest_frame = True)
        
    flux : array
        Flux values array (Rest-frame values if rest_frame = True)
    
    model : array
        Model values array (Rest-frame values if rest_frame = True)
        
    ivar : array
        Inverse variance array (Rest-frame values if rest_frame = True)

    """
    
    # getSpec from specClient can be used to obtain the spectra of any given object 
    # specobjid is required
    t = spec.getSpec(specobjid, fmt = 'numpy')
    
    # Extracting the different values - 
    loglam = t['loglam']       # Log wavelength
    flux = t['flux']           # Flux (f_lambda)  
    ivar = t['ivar']           # Inverse Variance
    model = t['model']         # Model
    wavelength = 10**(loglam)  # Wavelength values
    
    if (rest_frame == True):
        # If rest_frame = True, we convert the different arrays into their rest-frame values
        wavelength = wavelength/(1+z)
        flux = flux*(1+z)
        model = model*(1+z)
        ivar = ivar/((1+z)**2)
    
    return (wavelength, flux, model, ivar)

###########################################################################################################################################################
###########################################################################################################################################################
        
    
    