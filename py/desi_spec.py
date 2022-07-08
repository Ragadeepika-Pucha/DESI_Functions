"""
This module is for accessing and plotting DESI spectra.
It works at both Data Lab and NERSC. Set dl = True for accessing at Data Lab.
The module consists the following functions: 
    1. get_desi_spectra (targetid, z, specprod, rest_frame, dl)
    2. plot_desi_spectra (targetid, z, specprod, rest_frame, **kwargs)
    3. get_fastspec_columns(table, em_lines, aon = True, snr = False, add = False)

Author : Ragadeepika Pucha
Version : 2022 July 8
"""
####################################################################################################
####################################################################################################

import numpy as np

from astropy.table import Table
import fitsio
from astropy.convolution import convolve, Gaussian1DKernel

import desispec.io
from desispec import coaddition

import plotting_functions as pf

####################################################################################################
####################################################################################################

def get_desi_spectra(targetid, z, specprod = 'fuji', rest_frame = True, dl = False):
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
        
    specprod : str
        Spectral Release Name. Works for fuji and above. Default is fuji.
    
    rest_frame : bool
        Whether or not to return the rest-frame values. Default is True
        
    dl : bool
        Whether or not the code is running on DataLab. Default is False

    Returns
    -------
    
    lam : array
        Wavelength array (Rest-frame values if rest_frame = True)
        
    flux : array
        Flux values array (Rest-frame values if rest_frame = True)
        
    ivar : array
        Inverse variance array (Rest-frame values if rest_frame = True)
    """
    
    ## Specprod directory
    if (dl == True):
        specprod_dir = f'/dlusers/raga_steph/DESI/spectro/redux/{specprod}'
    else:
        specprod_dir = desispec.io.specprod_root(specprod)
        
    ## Summary redshift catalog
    zcat_file = f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits'
    zcat = Table.read(zcat_file)
    
    ## Selecting only primary spectra objects
    zcat_prim = zcat[zcat['ZCAT_PRIMARY']]
    
    ## Selecting the row for the input targetid
    tgt = (zcat_prim['TARGETID'].data == targetid)
    t_tgt = zcat_prim[tgt]
    
    ## Required values for accessing the spectra
    ## Survey, Program, Healpix Values
    survey = t_tgt['SURVEY'].astype(str).data[0]
    program = t_tgt['PROGRAM'].astype(str).data[0]
    hpx = t_tgt['HEALPIX'].data[0]
    
    ## Healpix Directory
    hpx_dir = f'{specprod_dir}/healpix'
    tgt_dir = f'{hpx_dir}/{survey}/{program}/{hpx//100}/{hpx}'
    
    coadd_filename = f'{tgt_dir}/coadd-{survey}-{program}-{hpx}.fits'
    
    ## Coadded spectra object
    coadd_obj = desispec.io.read_spectra(coadd_filename) 
    ## List of all targets in the coadded spectra object
    coadd_tgts = coadd_obj.target_ids().data
    
    ## Selecting the particular spectra of the targetid
    ii = (coadd_tgts == targetid)
    coadd_spec = coadd_obj[ii]
    
    ## Coadding the b,r,z into a single spectra
    res = coaddition.coadd_cameras(coadd_spec)
    ## Wavelength array of the coadded spectra
    lam = res.wave['brz']                
    ## Flux array of the coadded spectra
    flux = res.flux['brz'][0]            
    ## Inverse Variance array of the coadded spectra
    ivar = res.ivar['brz'][0]            

    if (rest_frame == True):
        # If rest_frame = True, we convert the different arrays into their rest-frame values
        lam = lam/(1+z)
        flux = flux*(1+z)
        ivar = ivar/((1+z)**2)

    return (lam, flux, ivar)
    
####################################################################################################
####################################################################################################

def plot_desi_spectra(targetid, z, specprod = 'fuji', rest_frame = True, smoothed = True, \
                     figsize = (14, 6), xlim = None, ylim = None, \
                     emission_lines = False, absorption_lines = False, \
                     em_lines = None, abs_lines = None, axs = None, ylabel = True, \
                     spectra_kwargs = {'color':'grey', 'alpha':0.5}, \
                     smooth_kwargs = {'color':'k', 'linewidth':2.0}, dl = False):
    """
    Function to plot DESI spectra
    
    Parameters
    ----------
    
    targetid : int64
        Unique TARGETID of the object
        
    z : float
        Redshift of the object
        
    specprod : str
        Spectral Release Name. Works for fuji and above. Defaut is fuji.
        
    rest_frame : bool
        Whether or not to plot the spectra in the rest frame. Default = True
        
    smoothed : bool
        Whether or not to plot the smoothed spectra or not. Default = True
        
    figsize : tuple
        Figure size if axs = None
        
    xlim : list or tuple
        Setting the xrange of the plot
        
    ylim : list or tuple
        Setting the yrange of the plot
        
    emission_lines : bool
        Whether or not to overplot emission lines. Default is False
        
    absorption_lines : bool
        Whether or not to overplot absorption lines. Default is False
        
    em_lines : list
        List of emission lines to plot
        If not mentioned, all the lines in the default list will be plotted.
        
    abs_lines : list
        List of absorption lines to plot
        If not mentioned, all the lines in the default list will be plotted.
        
    axs : axis
        The axis where the spectra needs to be plotted. 
        If it is None, a figure plot will be returned with just the spectra.
        
    ylabel : bool
        Whether or not to label the y-axis. Default = True
        
    spectra_kwargs : dict
        Plotting keyword arguments for the spectra
        
    model_kwargs : dict
        Plotting keyword arguments for the smoothed spectra
        
    dl : bool
        Whether or not the code is running on DataLab. Default is False
    
    Returns
    -------
    
    fig : Figure
        Returns the spectra figure if axs = None

    """
    
    # Get the wavelength, flux and inverse variance given the targetid
    lam, flux, ivar = get_desi_spectra(targetid, z, specprod = specprod,\
                                       rest_frame = rest_frame, dl = dl)
    
    if (axs == None):
        fig = plt.figure(figsize = figsize)
        axs = plt.gca()
    
    # Masking where inverse variance = 0
    ivar = np.ma.masked_where(ivar <=0, ivar)
    err = 1/np.sqrt(ivar)

    # Plotting the spectra
    axs.plot(lam, err, 'm', ls = '--', alpha = 0.75, lw = 2.0)
    axs.plot(lam, flux, **spectra_kwargs)
    
    # Smoothing the spectra using Gaussian1DKernel - using 3 pixels
    if (smoothed == True):
        ## Plot smoothed spectra if True
        flux_smoothed = convolve(flux, Gaussian1DKernel(3))
        axs.plot(lam, flux_smoothed, **smooth_kwargs)
        
    if (ylabel == True):
        ## Set ylabel if true
        axs.set_ylabel('$F_{\lambda}$ [$10^{-17}~ergs~s^{-1}~cm^{-2}~\AA^{-1}$]')
        
    ## Finding ymin and ymax within the spectra bounds
    x_bounds = axs.get_xbound()
    x_ii = (lam >= x_bounds[0]) & (lam <= x_bounds[1])
    flux_bounds = flux[x_ii]
    ymin = min(flux_bounds)-1
    ymax = max(flux_bounds)+1
    
    # Setting xlim
    axs.set(xlim = xlim, xlabel = '$\lambda~[\AA]$')
    
    # Setting ylim
    if (ylim == None): 
        x_bounds = axs.get_xbound()
        x_ii = (lam >= x_bounds[0]) & (lam <= x_bounds[1])
        flux_bounds = flux[x_ii]
        
        if (len(flux_bounds) == 0):
            axs.set_ylim([ymin, ymax])
        else:
            ymin_self = min(flux_bounds)-1
            ymax_self = max(flux_bounds)+1
            axs.set_ylim([ymin_self, ymax_self])
    else:
        axs.set_ylim(ylim)
        
    # Plotting Absorption/Emission lines - only works if either of the lines is set to True
    if (emission_lines == True)|(absorption_lines == True):    
        if (emission_lines == False):
            # Sending empty array of em_lines if emission_lines = False
            em_lines = []
        if (absorption_lines == False):
            # Sending empty array of abs_lines 
            abs_lines = []
            
        # Plotting function to add emission/absorption lines
        pf.add_lines(ax = axs, z = z, rest_frame = rest_frame, \
                     em_label = True, abs_label = True, \
                     em_lines = em_lines, abs_lines = abs_lines) 

    if (axs == None):
        plt.close()
        return (fig)

####################################################################################################
####################################################################################################
# Update later to include the keyword of which columns

def get_fastspec_columns(table, em_lines, aon = True, snr = False, add = False):
    """
    Function to access flux-related columns from the fastspecfit catalog.
    
    Parameters
    ----------
    
    table : Astropy Table
        Table containing fastspecfit-related columns
        
    em_lines : str or list
        Name of the emission lines whose flux needs to be selected. 
        If add = False, a single string is required.
        If add = True, a list of two strings.
        
    aon : bool
        Whether or not to return the Ampliture-Over-Noise Ratio for the emission-line.
        Default is True.
        
    snr : bool
        Whether or not to return the Signal-to-Noise Ratio for the emission-line.
        Default is False.
        
    add : bool
        Whether or not the total flux needs to be added.
        This is for doublet-type lines (Example: [SII]6716,6731).
        Default is False.
    
    Returns
    -------
    flux : array
        Array of flux values for the table. 
        If add = True, total flux of the combined emission lines.
        
    flux_aon : array
        This is returned only if flux_aon = True.
        Array of amplitude-over-noise ratios for the table.
        
    flux_snr : array
        This is returned only if flux_snr = True.
        Array of signal-to-noise ratios for the table.
    """
    
    if (aon):
        ## This block runs if aon = True - Request for amplitude-over-noise 
        ## Returns both flux and AoN
        if (add == False):
            ## Input is only for a single emission-line
            ## AoN = Amplitude/Amplitude_Error = Amplitude*sqrt(Amplitude_Ivar)
            flux = table[f'{em_lines}_FLUX'].data
            flux_aon = table[f'{em_lines}_AMP'].data*np.sqrt(table[f'{em_lines}_AMP_IVAR'].data)
        else:
            ## Input is a doublet.
            ## The flux of the two lines is added.
            ## AoN = (A1+A2)/Total_Error
            ## Total Amplitude Error = sqrt((A1_Err^2) + (A2_Err^2))
            ## A_Err^2 = 1/A_Ivar
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
            amp = table[f'{em_lines[0]}_AMP'].data+table[f'{em_lines[1]}_AMP'].data
            amp_noise = np.sqrt((1/tab[f'{em_lines[0]}_AMP_IVAR'])+\
                                (1/tab[f'{em_lines[1]}_AMP_IVAR']))
            flux_aon = amp/amp_noise
        return (flux, flux_aon)
    
    elif (snr):
        ## This block runs if aon = False and snr = True - Request for signal-to-noise
        ## Returns both flux and SNR
        if (add == False):
            ## Input is only for a single emission-line
            ## SNR = Flux/Flux_Error = Flux*sqrt(Flux_Ivar)
            flux = table[f'{em_lines}_FLUX'].data
            flux_snr = table[f'{em_lines}_FLUX'].data*np.sqrt(table[f'{em_lines}_FLUX_IVAR'].data)
        else:
            ## Input is a doublet.
            ## The flux of the two lines is added.
            ## SNR = (F1+F2)/Total_Error
            ## Total Flux Error = sqrt((F1_Err^2)+(F2_Err^2))
            ## F_Err^2 = 1/F_Ivar
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
            flux_noise = np.sqrt((1/tab[f'{em_lines[0]}_FLUX_IVAR'])+\
                                 (1/tab[f'{em_lines[1]}_FLUX_IVAR']))
            flux_snr = flux/flux_noise
        return (flux, flux_snr)
    
    else:
        ## This block runs when both aon = False and snr = False
        ## Only returns the flux
        if (add == False):
            ## Input is only for a single emission-line
            flux = table[f'{em_lines}_FLUX'].data
        else:
            ## Input is a doublet.
            ## The flux of the two lines is added.
            flux = table[f'{em_lines[0]}_FLUX'].data + table[f'{em_lines[1]}_FLUX'].data
        return (flux)
    
####################################################################################################
####################################################################################################
                