"""
This module consists of the following functions that are related to Tractor and DL:

    1. targetid_to_lsid (targetid)
    2. combine_observations (table, array_name, obj_num)


Author : Ragadeepika Pucha
Version : 2022 August 4 
"""

####################################################################################################
####################################################################################################

import numpy as np
from astropy.table import Table

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