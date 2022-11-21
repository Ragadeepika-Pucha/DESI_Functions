
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
    7) velocities_to_wavelengths(v, lam_0)
    8) plot_image_gelato_ha(table, index, title = None, gdir = None)
    

Author : Ragadeepika
Version : 2022 November 7
"""

####################################################################################################
####################################################################################################

import numpy as np
import fitsio

from astropy.io import fits
from astropy.table import Table, Column, vstack
from astropy.modeling.models import Gaussian1D

from desispec import coaddition
from desispec.io import specprod_root, read_spectra

from glob import glob
import plotting_functions as pf

import matplotlib.pyplot as plt

####################################################################################################
####################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':22,
    'axes.linewidth':2.0,
    'xtick.major.size':6.0,
    'xtick.minor.size':4.0,
    'xtick.major.width':2.0,
    'xtick.minor.width':1.5,
    'xtick.direction':'in', 
    'xtick.minor.visible':True,
    'xtick.top':True,
    'ytick.major.size':6.0,
    'ytick.minor.size':4.0,
    'ytick.major.width':2.0,
    'ytick.minor.width':1.5,
    'ytick.direction':'in', 
    'ytick.minor.visible':True,
    'ytick.right':True
}

plt.rcParams.update(**settings)


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
    #if not type(table.mask) == type(None):
    #    for c in table.colnames[1:]: table[c][table.mask[c]] = np.nan
        
       
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

def velocities_to_wavelengths(v, lam_0):
    """
    Function to convert from velocities (km/s) to wavelength values (in AA).
    del_lam = (v/c)*lam_0
    
    Parameters
    ----------
    v : array or float
        Velocities that need to be converted
        
    lam_0 : float
        Central wavelength for the frame of reference
        
    Returns
    -------
    del_lam : array or float
        Converted wavelength values
    """
    
    c = 3e+5 ## in km/s
    del_lam = (v/c)*lam_0
    
    return (del_lam)

####################################################################################################
####################################################################################################

def plot_image_gelato_ha(table, index, title = None, gdir = None):
    """
    Function to plot image-cutout and spectra+gelato model focused on the Ha+[NII] region.
    This function overplots the broad and narrow components of Ha.
    
    Parameters
    ----------
    table : astropy table
        The table of targets from which the object is selected.
    
    index : int
        Index of the object from the table that needs to be plotted.
        
    title : str
        Title that will go on the top of the plot. Default is None.
        
    gdir : str
        Path of the gelato output results files. Default is None.
        
    Returns
    -------
    fig : matplotlib figure
        Figure object containing the overall plot
    """ 
    ## Target information
    filename = table['Name'].astype(str).data[index]
    arr = filename.split('.')
    #targetid = table['TARGETID'].data[index]
    ra = table['RA'].data[index]
    dec = table['DEC'].data[index]

    gtab_file = f'{gdir}/{arr[0]}-results.fits'
    gtab = Table.read(gtab_file, hdu = 1)

    for c in gtab.columns:
        gtab[c].fill = 0.0

    gtab = gtab.filled()

    #c = 3e+5  ## in km/s
    #z = table['SSP_Redshift'].data[index]/c
    z = table['Z'].data[index]
    ## Spectra
    lam = 10**(gtab['loglam'])/(1+z)
    flam = gtab['flux']*(1+z)
    ivar = gtab['ivar']/((1+z)**2)

    model = gtab['MODEL']*(1+z)
    if ('PL' in gtab.colnames):
        cont = (gtab['SSP']+gtab['PL'])*(1+z)
    else:
        cont = gtab['SSP']*(1+z)
    residual = model - flam
    
    ## Gaussian components for Broad and Narow Halpha
    ha_mean = 6564.613

    ## Narrow H-alpha

    ha_amp = get_gelato_columns(table, 'Ha', prop = 'RAmp')[index]
    ha_sig = get_gelato_columns(table, 'Ha', prop = 'Dispersion')[index]

    ha_std = velocities_to_wavelengths(ha_sig, ha_mean)

    gauss_ha = Gaussian1D(amplitude = ha_amp, mean = ha_mean, stddev = ha_std)

    ## Broad H-alpha

    ha_broad_amp = get_gelato_columns(table, 'Ha_broad', prop = 'RAmp')[index]
    ha_broad_sig = get_gelato_columns(table, 'Ha_broad', prop = 'Dispersion')[index]
    ha_broad_std = velocities_to_wavelengths(ha_broad_sig, ha_mean)

    gauss_ha_broad = Gaussian1D(amplitude = ha_broad_amp, mean = ha_mean, stddev = ha_broad_std)
    
    ## Get image-cutout
    im = pf.get_image_cutout(ra_in = ra, dec_in = dec, cutout_size = 20)
    
    ## Region around [NII]+Ha
    lam_xx = np.arange(6400, 6700, 1)
    lam_ii = (lam >= 6400)&(lam <= 6700)
    cont_xx = np.median(cont[lam_ii])

    ## Plot
    
    fig = plt.figure(figsize = (14, 12))
    gs = fig.add_gridspec(6, 5)
    
    plt.suptitle(title, fontsize = 20)

    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0:2,2:])
    ax3 = fig.add_subplot(gs[2:5,:])
    ax4 = fig.add_subplot(gs[5,:], sharex = ax3)

    ax1.imshow(im)
    ax1.set(xticks = [], yticks = [])

    ax2.plot(lam, flam, color = 'grey', alpha = 0.8, label = 'Flux')
    ax2.plot(lam_xx, gauss_ha(lam_xx)+cont_xx, color = 'r', label = 'Narrow H$\\alpha$')
    ax2.plot(lam_xx, gauss_ha_broad(lam_xx)+cont_xx, color = 'b', label = 'Broad H$\\alpha$')
    ax2.legend(loc = 'best', fontsize = 12)
    ax2.set(ylabel = '$F_{\lambda}$', xlim = [6440, 6680], xlabel = '$\lambda$')

    x_bounds = ax2.get_xbound()
    x_ii = (lam >= x_bounds[0])&(lam <= x_bounds[1])
    flam_ii = flam[x_ii]
    ymin_ii = min(flam_ii)-1
    ymax_ii = max(flam_ii)+1
    ax2.set_ylim([ymin_ii, ymax_ii])

    ax3.plot(lam, flam, color = 'grey', alpha = 0.8, label = 'Flux')
    ax3.plot(lam, model, color = 'k', lw = 2.0, label = 'Model')
    ax3.legend(loc = 'best', fontsize = 16)
    ax3.set(ylabel = '$F_{\lambda}$', xlim = [4600, 6900])

    x_bounds = ax3.get_xbound()
    x_ii = (lam >= x_bounds[0])&(lam <= x_bounds[1])
    flam_ii = flam[x_ii]
    ymin_ii = min(flam_ii)-1
    ymax_ii = max(flam_ii)+1
    ax3.set_ylim([ymin_ii, ymax_ii])

    ax4.scatter(lam, residual, color = 'k', s = 10, marker = '.', alpha = 0.5)
    ax4.axhline(0.0, color = 'firebrick', ls = '--', lw = 3.0)
    ax4.set_ylabel('Residual', fontsize = 16)
    ax4.set(ylim = [-7,7], xlabel = '$\lambda$')

    plt.tight_layout(rect=[0, 0, 1., 0.99])
    plt.close()
    
    return (fig)
    
####################################################################################################
####################################################################################################
    
