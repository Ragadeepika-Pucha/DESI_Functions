"""
The functions in this script are related TO DESI TARGET MASKS.
Updated to work with fuji/guadalupe.

Ragadeepika Pucha
Version : 2022, May 27
"""

####################################################################################################
import numpy as np

from astropy.table import Table

from desispec.io import specprod_root
from desitarget import targetmask
from desitarget.sv1 import sv1_targetmask
from desitarget.sv2 import sv2_targetmask
from desitarget.sv3 import sv3_targetmask
####################################################################################################

## The target masks that are associated with different DESI columns

target_dict = {
               'DESI_TARGET' : targetmask.desi_mask, \
               'BGS_TARGET' : targetmask.bgs_mask, \
               'MWS_TARGET' : targetmask.mws_mask, \
               'SCND_TARGET' : targetmask.scnd_mask, \
               'SV1_DESI_TARGET' : sv1_targetmask.desi_mask, \
               'SV1_BGS_TARGET' : sv1_targetmask.bgs_mask, \
               'SV1_MWS_TARGET' : sv1_targetmask.mws_mask, \
               'SV1_SCND_TARGET' : sv1_targetmask.scnd_mask, \
               'SV2_DESI_TARGET' : sv2_targetmask.desi_mask, \
               'SV2_BGS_TARGET' : sv2_targetmask.bgs_mask, \
               'SV2_MWS_TARGET' : sv2_targetmask.mws_mask, \
               'SV2_SCND_TARGET' : sv2_targetmask.scnd_mask, \
               'SV3_DESI_TARGET' : sv3_targetmask.desi_mask, \
               'SV3_BGS_TARGET' : sv3_targetmask.bgs_mask, \
               'SV3_MWS_TARGET' : sv3_targetmask.mws_mask, \
               'SV3_SCND_TARGET' : sv3_targetmask.scnd_mask
              }

####################################################################################################

def get_targetbit_info(column_value, column_name):
    """
    This function tells you what bits are set for the given column value.
    
    Parameters
    ----------
    column_value : int64
        The 64-bit integer value that is set for a source in the column
        
    column_name : str
        The name of the TARGET MASK column
        
    Returns
    -------
    bitnums : list
        List of the bits that are set for the column_value
        
    bitnames : list
        List of the bit names that describe the bits set
    """
    
    # Mask name corresponding to a given DESI column
    mask_name = target_dict[column_name]
    
    bitnums = []   ## Array for the bit numbers that are set
    bitnames = []   ## Arrat for the bit names described by the column
    
    for jj in range(63):
        if (column_value & (2**jj) != 0):
            bitnums.append(jj)
            bitnames.append(mask_name.bitname(jj))
                
    return (bitnums, bitnames)

####################################################################################################

def targetbit_catalog(column_name, bitname, specprod = 'guadalupe', dl = False):
    """
    The function to get the subset of redshift catalog given a BITMASK name.
    
    Parameters
    ----------
    column_name : int64
        The name of the TARGET MASK column that has the bit nam
        
    bitname : str
        BIT NAME which needs to be set to be selected.
        
    specprod : str
        Spectral Release Name
        
    dl : bool
        Whether the code is run on DataLab or not
        
    Returns
    -------
    tsel : Astropy Table
        Subset of the redshift catalog that consists of all sources for which the 'bitname' is set.
    """
    
    ## Specprod directory
    if (dl == True):
        specprod_dir = f'/dlusers/raga_steph/DESI/spectro/redux/{specprod}'
    else:
        specprod_dir = specprod_root(specprod)
        
    ## Summary redshift catalog
    zcat_file = f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits'
    t = Table.read(zcat_file)
    
    ## Mask name corresponding to a given DESI column
    mask_name = target_dict[column_name]
    
    ## Mask for the required bitname
    req_mask = mask_name[bitname]
    
    ## Selecting the rows based on the mask
    sel = (t[column_name] & req_mask != 0)
    
    ## Subset of the table that has the bitname set 
    tsel = t[sel]
    
    return (tsel)
    
####################################################################################################    
    
def targetbit_sv_catalog(column_name, bitname, dl = False):
    """
    The function to get the subset of redshift catalog given a BITMASK name for the entire SV.
    Works only for fuji - combines SV1+SV2+SV3.
    
    Parameters
    ----------
    column_name : int64
        The name of the TARGET MASK column that has the bit nam
        
    bitname : str
        BIT NAME which needs to be set to be selected.
        
    dl : bool
        Whether the code is run on DataLab or not
        
    Returns
    -------
    tsel : Astropy Table
        Subset of the redshift catalog (SV1+SV2+SV3),
        that consists of all sources for which the 'bitname' is set.
    
    """
    
    ## Specprod directory
    if (dl == True):
        specprod_dir = f'/dlusers/raga_steph/DESI/spectro/redux/fuji'
    else:
        specprod_dir = specprod_root('fuji')
        
    ## Summary redshift catalog
    zcat_file = f'{specprod_dir}/zcatalog/zall-pix-fuji.fits'
    t = Table.read(zcat_file)
    
    ## SV1 Mask 
    sv1_col = f'SV1_{column_name}'
    sv1_mask_name = target_dict[sv1_col]
    sv1_req_mask = sv1_mask_name[bitname]
    
    ## SV2 Mask 
    sv2_col = f'SV2_{column_name}'
    sv2_mask_name = target_dict[sv2_col]
    sv2_req_mask = sv2_mask_name[bitname]
    
    ## SV3 Mask 
    sv3_col = f'SV3_{column_name}'
    sv3_mask_name = target_dict[sv3_col]
    sv3_req_mask = sv3_mask_name[bitname]
    
    ## Selecting the rows based on the masks
    sel = (t[sv1_col] & sv1_req_mask != 0)|\
    (t[sv2_col] & sv2_req_mask != 0)|(t[sv3_col] & sv3_req_mask != 0)
    
    ## Subset of the table that has the bitname set 
    tsel = t[sel]
    
    return (tsel)
   
####################################################################################################        