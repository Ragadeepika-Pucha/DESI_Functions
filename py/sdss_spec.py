"""
This module is for accessing and plotting SDSS spectra
This works only at Data Lab. It consists of the following functions.
    1. get_sdss_spectra (specobjid, z, rest_frame = True)
    2. plot_sdss_spectra (specobjid, z, rest_frame = True, **kwargs)

Author : Ragadeepika Pucha
Version : 
"""
####################################################################################################
####################################################################################################

import numpy as np
from astropy.table import Table
from dl import specClient as spec

from astropy.convolution import convolve, Gaussian1DKernel
import plotting_functions as pf

####################################################################################################
####################################################################################################

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

####################################################################################################
####################################################################################################

def plot_sdss_spectra(specobjid, z, rest_frame = True, plot_model = True, smoothed = False, \
                      figsize = (14, 6), xlim = None, ylim = None, \
                      emission_lines = False, absorption_lines = False, \
                      em_lines = None, abs_lines = None, axs = None, ylabel = True, \
                      spectra_kwargs = {'color': 'grey', 'alpha': 0.5},\
                      model_kwargs = {'color': 'k', 'linewidth': 2.0}, \
                      smooth_kwargs = {'color':'k', 'linewidth':2.0}):
    
    """
    Function to plot the SDSS spectra.
    
    Parameters
    ----------
    specobjid : int64
        Unique SDSS spectrum id
    
    z : float
        Redshift of the source
    
    rest_frame : bool
        Whether or not to plot the spectra in the rest-frame. Default is True.
        
    plot_model : bool
        Whether or not to plot the model spectra. Default = True
        
    smoothed : bool
        Whether or not to plot the smoothed spectra. Default = False
        
    figsize : tuple
        Figure size if axs = None
    
    xlim : list or tuple
        Setting the xrange of the plot
        
    ylim : list or tupe
        Setting the yrange of the plot
    
    emission_lines : bool
        Whether or not to overplot emission lines. Default is False.
    
    absorption_lines : bool
        Whether or not to overplot absorpion lines. Default is False.
        
    em_lines : list
        List of emission lines to plot
        If not mentioned, all the lines in the default list will be plotted.
        
    abs_lines : list
        List of absorption lines to plot
        If not mentioned, all the lines in the default list will be plotted.
    
    spectra_kwargs : dict
        Plotting keyword arguments for the spectra
        
    model_kwargs : dict
        Plotting keyword arguments for the model
        
    spectra_kwargs : dict
        Plotting keyword arguments for the smoothed spectra
        
    axs : axis
        The axis where the spectra needs to be plotted.
        It it is None, a figure plot will be returned with just the spectra
        
    Returns
    -------
    fig : Figure
        Returns the figure with the sdss spectra, if axs = None.
    
    """
    
    # Getting the wavelength, flux, model and ivar arrays for the given source
    lam, flux, model, ivar = get_sdss_spectra(specobjid, z, rest_frame)
    
    if (axs == None):
        fig = plt.figure(figsize = figsize)
        axs = plt.gca()
    
    # Masking where inverse variance = 0
    ivar = np.ma.masked_where(ivar <=0, ivar)
    err = 1/np.sqrt(ivar)

    # Plotting the error spectra   
    axs.plot(lam, err, 'm', ls = '--', alpha = 0.75, lw = 2.0)
    # Plotting the spectra
    axs.plot(lam, flux, **spectra_kwargs)
    
    if (plot_model == True):
        ## Plotting the model is plot_model = True
        axs.plot(lam, model, **model_kwargs) 
        
    if (smoothed == True):
        ## Plotting the smoothed spectra if smoothed = True
        flux_smoothed = convolve(flux, Gaussian1DKernel(3))
        axs.plot(lam, flux_smoothed, **smooth_kwargs)
        
    if (ylabel == True):
        ## Set ylabel if True
        axs.set_ylabel('$F_{\lambda}$ [$10^{-17}~ergs~s^{-1}~cm^{-2}~\AA^{-1}$]')
    
    ## Getting the ymin and ymax bounds
    x_bounds = axs.get_xbound()
    x_ii = (lam >= x_bounds[0]) & (lam <= x_bounds[1])
    flux_bounds = flux[x_ii]
    ymin = min(flux_bounds)-1
    ymax = max(flux_bounds)+1
    
    # Setting the xlim
    axs.set(xlim = xlim, xlabel = '$\lambda~[\AA]$')
    
    ## Setting the ylim
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