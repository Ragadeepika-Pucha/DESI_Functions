
"""
This script contains functions used for input and output of GELATO.

## Inputs needed for GELATO

1) Input spectra (one FITS file per spectrum)

- The log10 of the wavelengths in Angstroms, column name: "loglam"
- The spectral flux density in flam units, column name: "flux"
- The inverse variances of the data points, column name: "ivar"
- saved in `gelato_input/spectra/`

2) Input list

- a comma delimited file (ending in .csv) where each object occupies a different line. 
The first item in each line is the path to the spectrum. 
The second is the redshift of the spectrum.

- a FITS table (ending in .fits) where each object occupies a different entry in the table. 
The table must have the column "Path" for the path to the object, 
and "z" containing the redshift of the object.

- saved in `gelato_input/catalogs/`

## Notes to keep in mind
- the DESI masked pixels have ivar=0 and flux=0 so we don't need to carry the mask itself

The script consists of following functions:
    1) gelatify_spectra(specprod, survey, program, healpix, targetids)
    2) create_path_column(table)
    3) gelatify(table, output_name)
    4) divide_into_subsamples(table, n_samp, output_root)
    5) gelato_output(output_name = 'GELATO-results.fits')
    6) get_gelato_columns(table, em_line, prop = 'Flux')
    

Author : Ragadeepika
Version : 2022 October 25
"""

####################################################################################################
####################################################################################################

import numpy as np
import fitsio

from astropy.table import Table, Column
from desispec import coaddition
from desispec.io import specprod_root, read_spectra

from glob import glob

####################################################################################################
####################################################################################################

## Output files directories

## Path to spectra
spec_out = 'input/spectra'
## Path to catalogs
cat_out = 'input/catalogs'

####################################################################################################
####################################################################################################

def gelatify_spectra(specprod, survey, program, healpix, targetids):
    """
    Function to turn a given list of spectra into GELATO-input files.
    The input spectra files are saved in input/spectra folder.
    
    Parameters
    ----------
    specprod : str
        Spectral Release Name
        
    survey : str
        Survey name for the targetes. sv1|sv2|sv3|main|special
        
    program : str
        Observing program for the targets. dark|bright|other|backup
        
    healpix : int
        Healpix number for the targets
        
    targetids : numpy array
        Array of targetids in a given specprod, survey, program, and healpix
        
    Returns
    -------
    None.
        Saves the individual spectra in GELATO-required format.
    """
    
    # Root directory of DESI spectra for a given specprod
    specprod_dir = specprod_root(specprod)
    hpx_dir = f'{specprod_dir}/healpix'
    
    # Filename & directory 
    coadd_filename = f'coadd-{survey}-{program}-{healpix}.fits'
    tgt_dir = f'{hpx_dir}/{survey}/{program}/{healpix//100}/{healpix}'
    
    # Using desispec to read the spectra
    coadd_obj = read_spectra(f'{tgt_dir}/{coadd_filename}')
    coadd_tgts = coadd_obj.target_ids().data
    
    sel = np.isin(coadd_tgts,targetids)
    coadd_obj_sel = coadd_obj[sel]
    coadd_sel_tgts = coadd_obj_sel.target_ids().data
    
    # Combined Spectra - 
    spec_combined = coaddition.coadd_cameras(coadd_obj_sel)
    
    lam = spec_combined.wave['brz']
    loglam = np.log10(lam)
    
    # Loop over targetids
    for tgt in targetids:
        ## Find the index of each targetid
        ## Get flux and ivar values and write into a fits file
        i = (coadd_sel_tgts == tgt)
        flam = spec_combined.flux['brz'][i][0]
        flam_ivar = spec_combined.ivar['brz'][i][0]
        outfile = f'{spec_out}/gspec-{survey}-{program}-{healpix}-{tgt}.fits'
        fitsio.write(outfile,[loglam,flam,flam_ivar], names=['loglam','flux','ivar'], clobber=True)
        
####################################################################################################
####################################################################################################

def create_path_column(table):
    """
    Function to create the "Path" column for the GELATO input file.
    
    Parameters
    ----------
    table : astropy table
        Table of sources that needs to be "gelatified"
        
    Returns
    -------
    path_column : Column
        "Path" column containing the path to the different input spectra
    
    """
    
    ## Each spectrum filename - input/spectra/gspec-{survey}-{program}-{healpix}-{targetid}.fits
    
    survey = table['SURVEY'].astype(str).data.astype(object)
    program = table['PROGRAM'].astype(str).data.astype(object)
    healpix = table['HEALPIX'].astype(str).data.astype(object)
    targetid = table['TARGETID'].astype(str).data.astype(object)
    spec_out = np.array(['input/spectra/gspec-']*len(table)).astype(object)
    hyphen = np.array(['-']*len(table)).astype(object)
    fits = np.array(['.fits']*len(table)).astype(object)
    
    path = spec_out+survey+hyphen+program+hyphen+healpix+hyphen+targetid+fits
    path_column = Column(path, name = 'Path', dtype = 'str')
    
    return (path_column)

####################################################################################################
####################################################################################################    

def gelatify(table, output_name):
    """
    Function to create the required GELATO input files for a given table of targets.
    
    Parameters
    ----------
    table : astropy table
        Table of sources that needs to be "gelatified"
        
    output_name : str
        Name of the output catalog
        
    Returns
    -------
    None.
        The output catalog is saved with the given name in input/catalogs/ folder.
    """
    
    specprods = np.unique(table['SPECPROD'].astype(str))
    ## group by specprod
    for specprod in specprods:
        t_spec = table[table['SPECPROD'].astype(str) == specprod]
        
        ## group by survey
        surveys = np.unique(t_spec['SURVEY'].astype(str))
        for survey in surveys:
            t_surv = t_spec[t_spec['SURVEY'].astype(str) == survey]

            ## group by program
            programs = np.unique(t_spec['PROGRAM'].astype(str))
            for program in programs:
                t_prog = t_surv[t_surv['PROGRAM'].astype(str) == program]

                ## group by healpix
                healpixs = np.unique(t_prog['HEALPIX'])
                for hpx in healpixs:
                    t_hpx = t_prog[t_prog['HEALPIX'] == hpx]
                    tgts = t_hpx['TARGETID'].data
                    gelatify_spectra(specprod, survey, program, hpx, tgts)

    print (' ')
    print ('Done!')
    
    ## Create the GELATO input list with 'Path' and 'z'
    ## Initialize output table
    t_out = Table()
    path_col = create_path_column(table)
    t_out.add_column(path_col)
    t_out['z'] = table['Z']
    t_out.write(f'{cat_out}/{output_name}', overwrite = True)
    
####################################################################################################
####################################################################################################

def divide_into_subsamples(table, n_samp, output_root):
    """
    Function to divide the entire catalog into sub-samples to help with GELATO run.
    
    Parameters
    ----------
    table : astropy table
        GELATO input table that needs to be divided into sub-samples
        
    n_samp : int
        Number of sub-samples
        
    output_root : str
        Output root name for each of the sub-samples.
    
    Returns
    -------
    None.
        The tables for all the sub-samples are saved in input/catalogs/ folder.
    
    """

    for ii in range(n_samp):
        tab = table[ii::n_samp]
        tab.write(f'{cat_out}/{outpit_root}-{ii+1}.fits', overwrite = True)
        
####################################################################################################
####################################################################################################

def gelato_output(output_name = 'GELATO-results.fits'):
    """
    Function to combine all GELATO output files to get a single results file.
    The files should reside in Results/ folder.
    
    Parameters
    ----------
    output_name : str
        Path and filename for the resulting file. Default in GELATO-results.fits
        
    Returns
    -------
    None.
        The output file is saved at the required path.
    
    """
    
    files = glob('Results/gspec*.fits')
    
    tables = []

    for ii in range(len(files)):
        comps = files[ii][8:].split('-')
        filename = '-'.join(comps[:-1])+'.fits'
        targetid = comps[-2]
        parameters = fits.getdata(files[ii], 'PARAMS')

        ## Initialize Lists
        data = [targetid, filename]
        names = ['TARGETID', 'Name']
        dtype = ['>i8', np.unicode_] + [np.float64 for i in range(2*len(parameters.columns.names))]

        # Iterate over columns and add
        for n in parameters.columns.names:

            ps = parameters[n]
            ps = ps[np.invert(np.isinf(ps))]

            # Add medians
            data.append(np.nanmedian(ps))
            names.append(n)

            # Add errors
            data.append(np.nanstd(ps))
            names.append(n+'_err')

        # Append to list
        tables.append(Table(data=np.array(data),names=names,dtype = dtype))
        
    table = vstack(tables, join_type = 'outer')
    if not type(table.mask) == type(None):
        for c in table.colnames[1:]: table[c][table.mask[c]] = np.nan
        
       
    table.write(output_name, overwrite = True)
    
####################################################################################################
####################################################################################################
    
def get_gelato_columns(table, em_line, prop = 'Flux'):
    """
    Function to output GELATO columns for a given emission line.
    
    Parameters
    ----------
    table : astropy table
        GELATO results table from which to grab the required columns
        
    em_line : str
        Required emission-line.
        For now works only for Ha, Hb, [NII], [OIII], [SII]_6716/6731, 'Ha_broad', 'Hb_broad'.
        
    prop : str
        Property name whose column needs to be grabbed for the given emission line.
        Default is 'Flux'
        
    Returns
    -------
    column : Column
        Required emission-line column.
    
    """
    ## Dictionary to match emission-line with the required column
    gel_dict = {'Ha':'BPT_Ha_6564.613', 
                'Hb':'BPT_Hb_4862.683',
                '[NII]':'BPT_[NII]_6585.277',
                '[OIII]':'BPT_[OIII]_5008.239',
                '[SII]_6716':'BPT_[SII]_6718.294',
                '[SII]_6731':'BPT_[SII]_6732.673',
                'Ha_broad':'Broad_Ha_Broad_6564.613',
                'Hb_broad':'Broad_Hb_Broad_4862.683'}
    
    term = gel_dict[em_line]
    column_name = f'{term}_{prop}'
    column = table[column_name].data
    
    return (column)

####################################################################################################
####################################################################################################