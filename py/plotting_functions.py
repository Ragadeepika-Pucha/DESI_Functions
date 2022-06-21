"""
This module consists of different plotting-related functions, 
including getting image cutouts and plotting them.
It consists of following functions:
    1. get_image_cutout (ra_in, dec_in, **kwargs)
    2. get_multiple_image_cutouts (ra_in, dec_in, **kwargs)
    3. get_multiwavelength_cutouts (ra_in, dec_in, **kwargs)
    4. plot_cutouts (imgs, Nplotmax)
    5. add_lines (z, **kwargs)
    
Author : Ragadeepika Pucha
Version : 2022 June 21
"""

####################################################################################################
####################################################################################################

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends import backend_pdf as pdf

from astropy.table import Table
from astropy.utils.data import download_file

from urllib.error import HTTPError

####################################################################################################
####################################################################################################

def get_image_cutout(ra_in, dec_in, layername = 'ls-dr9', pixel_scale = 0.262, cutout_size = 60.):
    """
    Function to get the image cutout from a given survey.
    Getting images for each of these surveys require a layername.
    To be consistent, we want to show the pixel scales to be similar to the ones from the survey.
    List of layernames and pixel scales for the surveys - 
    - GALAX - 'galex' - 1.5"/pixel
    - SDSS - 'sdss' - 0.396"/pixel
    - LS DR9 - 'ls-dr9' - 0.262"/pixel
    - WISE W1/W2 - 'unwise-neo6' - 2.75"/pixel
    - VLASS - 'vlass1.2' - 1.0"/pixel
    
    Parameters
    ----------
    ra_in, dec_in : float, float
        Coordinates of the sky location for the image cutout
        
    layername : str
        String that leads to the url of the survey
        
    pixel_scale : float
        Pixel scale for the given survey in arcseconds per pixel
        
    cutout_size : float
        Size of the cutout. Default = 60" (1 arc-minute in size)
        This is used along with the pixel_scale, to calculate the number of pixels.
        Note - The number of pixels is rounded to the nearest integer. 
        The final cutout size is approximately may not be the exact input value, but close to it
        
    Returns
    -------
        img : 2d array
            Array containing the image cutout
    """
    
    pixels = cutout_size/pixel_scale
    cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra=%.6f&dec=%.6f&layer='\
    %(ra_in,dec_in)+layername+'&pixscale=%g&size=%d'%(pixel_scale, pixels)
    try:
        img = plt.imread(download_file(cutout_url,cache=False,show_progress=False,timeout=120))
    except HTTPError as e:
        if (e.code == 500):
            img = np.zeros((int(pixels), int(pixels),3))
    
    return (img)

####################################################################################################
####################################################################################################

def get_multiple_image_cutouts(ra_in, dec_in,  layername = 'ls-dr9', \
                               pixel_scale = 0.262, cutout_size = 20.):
    """
    Function to get multiple image cutouts from a given survey.
    Getting images for each of these surveys require a layername.
    To be consistent, we want to show the pixel scales to be similar to the ones from the survey.
    List of layernames and pixel scales for the surveys - 
    - GALAX - 'galex' - 1.5"/pixel
    - SDSS - 'sdss' - 0.396"/pixel
    - LS DR9 - 'ls-dr9' - 0.262"/pixel
    - WISE W1/W2 - 'unwise-neo6' - 2.75"/pixel
    - VLASS - 'vlass1.2' - 1.0"/pixel
    
    Parameters
    ----------
    
    ra_in, dec_in : float, float
        Coordinates of the sky location for the image cutout
        
    layername : str
        String that leads to the url of the survey
        
    pixel_scale : float
        Pixel scale for the given survey in arcseconds per pixel
        
    cutout_size : float
        Size of the cutout. Default = 20"
        This is used along with the pixel_scale, to calculate the number of pixels.
        Note - The number of pixels is rounded to the nearest integer. 
        The final cutout size is approximately may not be the exact input value, but close to it.
        
    Returns
    -------
    imgs : list
        List of image cutouts of the input targets

    """
    imgs = []
    
    N = len(ra_in)
    
    for ii in range(N):
        img = get_image_cutout(ra_in[ii], dec_in[ii], layername = layername, \
                               pixel_scale = pixel_scale, cutout_size = cutout_size)
        imgs.append(img)
        
    return (imgs)

####################################################################################################
####################################################################################################

def get_multiwavelength_cutouts(ra_in, dec_in, cutout_size = 60.):
    """
    Function to get the list of image cutouts in order - 'GALEX', 'LS DR9', 'WISE', 'VLASS'
    Getting images for each of these surveys require a layername.
    To be consistent, we want to show the pixel scales to be similar to the ones from the survey.
    List of layernames and pixel scales for the surveys - 
    - GALAX - 'galex' - 1.5"/pixel
    - LS DR9 - 'ls-dr9' - 0.262"/pixel
    - WISE W1/W2 - 'unwise-neo6' - 2.75"/pixel
    - VLASS - 'vlass1.2' - 1.0"/pixel
    
    Parameters
    ----------
    
    ra_in, dec_in : float, float
        Coordinates of the sky location for the image cutouts
        
    cutout_size : float
        The size of the cutout. Defualt = 60" (1 arc-minute in size)
        This is used along with the pixel_scale, to calculate the number of pixels.
        Note - The number of pixels is rounded to the nearest integer. 
        The final cutout size is approximately may not be the exact input value, but close to it.
    
    Returns
    -------
    
    imgs : list
        List of image cutouts in order - 'GALEX', 'LS DR9', 'WISE', 'VLASS'
    """
    
    imgs = []
    
    layernames = np.array(['galex', 'ls-dr9', 'unwise-neo6', 'vlass1.2'])
    pixel_scales = np.array([1.5, 0.262, 2.75, 1.0])
    pixels = cutout_size/pixel_scales
    
    for index, layername in enumerate(layernames):
        img = get_image_cutout(ra_in, dec_in, layername = layername, \
                               pixel_scale = pixel_scales[index], cutout_size = cutout_size)
        imgs.append(img)
    
    return (imgs)

####################################################################################################
####################################################################################################

def plot_cutouts(imgs, Nplotmax = 80):
    """
    Function to plot a list of given image cutouts
    
    Parameters
    ----------
    imgs : list
        List of image cutouts to plot
    Nplotmax : int
        Maximum Number of image cutouts to plot from the given list. Default = 40
    
    """
    
    N = min([len(imgs), Nplotmax])
    Nrow = int(N/8) + 1
    
    fig = plt.figure(figsize = (16, 2*Nrow))
    
    for ii in range(N):
        ax = fig.add_subplot(Nrow, 8, ii+1)
        ax.imshow(imgs[ii])
        ax.set(xticks = [], yticks = [])
        
    plt.tight_layout()
    plt.close()
        
    return (fig)

####################################################################################################
####################################################################################################

def add_lines(z, ax = None, rest_frame = True, em_label = True,\
              abs_label = True, em_lines = None, abs_lines = None):
    """
    Function to add emission and/or absorption lines onto a plot. 
    
    If em_lines or abs_lines is given, plotting only the specified lines.
    If no lines are given, plotting all the lines that are present in x-range of the plot.
    
    Parameters
    ----------
    z : float
        Redshift value of the source
    
    ax : AxesSubplot
        The axis onto which the emission/absoption lines needs to be plotted.
        If ax = None, then the plotting function uses plt, rather than axis.
        
    rest_frame : bool
        Whether or not the plot is in rest-frame. Default is True.
    
    em_label : bool
        Whether or not to label the emission lines. Default is True.
        
    abs_label : bool
        Whether or not to label the absorption lines. Default is True.
    
    em_lines : list
        List of emission lines to label
    
    abs_lines : list
        List of absorption lines to label
    
    Returns
    -------
    None
    
    """
    
    # List of lines
    # This is the set of emission lines from the spZline files. 
    # All the wavelengths are in vaccuum wavelengths.

    # Emission Lines
    emission_lines = [
    {"name" : "Ly-alpha",       "lambda" : 1215.67,  "emission": True,  "label" : "Ly$\\alpha$"},
    {"name" : "N V 1240",       "lambda" : 1240.81,  "emission": True,  "label" : "N V"},
    {"name" : "C IV 1549",      "lambda" : 1549.48,  "emission": True,  "label" : "C IV" },
    {"name" : "He II 1640",     "lambda" : 1640.42,  "emission": True,  "label" : "He II"},
    {"name" : "C III] 1908",    "lambda" : 1908.734, "emission": True,  "label" : "C III]"},
    {"name" : "Mg II 2799",     "lambda" : 2800.315, "emission": True,  "label" : "Mg II" },
    {"name" : "[O II] 3725",    "lambda" : 3727.092, "emission": True,  "label" : " "},
    {"name" : "[O II] 3727",    "lambda" : 3729.875, "emission": True,  "label" : "[O II]"}, 
    {"name" : "[Ne III] 3868",  "lambda" : 3869.857, "emission": True,  "label" : "[Ne III]"},
    {"name" : "H-zeta",         "lambda" : 3890.151, "emission": True,  "label" : "H$\\zeta$"},
    {"name" : "[Ne III] 3970",  "lambda" : 3971.123, "emission": True,  "label" : "[Ne III]"},
    {"name" : "H-epsilon",      "lambda" : 3971.195, "emission": True,  "label" : "H$\\epsilon$"}, 
    {"name" : "H-delta",        "lambda" : 4102.892, "emission": True,  "label" : "H$\\delta$"},
    {"name" : "H-gamma",        "lambda" : 4341.684, "emission": True,  "label" : "H$\\gamma$"},
    {"name" : "[O III] 4363",   "lambda" : 4364.435, "emission": True,  "label" : "[O III]"},
    {"name" : "He II 4685",     "lambda" : 4686.991, "emission": True,  "label" : "He II"},
    {"name" : "H-beta",         "lambda" : 4862.683, "emission": True,  "label" : "H$\\beta$"},
    {"name" : "[O III] 4959",   "lambda" : 4960.294, "emission": True,  "label" : "[O III]" },
    {"name" : "[O III] 5007",   "lambda" : 5008.239, "emission": True,  "label" : "[O III]" },
    {"name" : "He II 5411",     "lambda" : 5413.025, "emission": True,  "label" : "He II"},
    {"name" : "[O I] 5577",     "lambda" : 5578.888, "emission": True,  "label" : "[O I]" },
    {"name" : "[N II] 5755",    "lambda" : 5756.186, "emission": True,  "label" : "[Ne II]" },
    {"name" : "He I 5876",      "lambda" : 5877.308, "emission": True,  "label" : "He I" },
    {"name" : "[O I] 6300",     "lambda" : 6302.046, "emission": True,  "label" : "[O I]" },
    {"name" : "[S III] 6312",   "lambda" : 6313.806, "emission": True,  "label" : "[S III]" },
    {"name" : "[O I] 6363",     "lambda" : 6365.535, "emission": True,  "label" : "[O I]" },
    {"name" : "[N II] 6548",    "lambda" : 6549.859, "emission": True,  "label" : "[N II]" },
    {"name" : "H-alpha",        "lambda" : 6564.614, "emission": True,  "label" : "H$\\alpha$" },
    {"name" : "[N II] 6583",    "lambda" : 6585.268, "emission": True,  "label" : "[N II]" },
    {"name" : "[S II] 6716",    "lambda" : 6718.294, "emission": True,  "label" : "[S II]" },
    {"name" : "[S II] 6730",    "lambda" : 6732.678, "emission": True,  "label" : "[S II]" },
    {"name" : "[Ar III] 7135",  "lambda" : 7137.758, "emission": True,  "label" : "[Ar III]" },]


    # Absorption lines
    absorption_lines = [
    {"name" : "H12",            "lambda" : 3751.22,  "emission": False, "label" : "H12"},
    {"name" : "H11",            "lambda" : 3771.70,  "emission": False, "label" : "H11"},
    {"name" : "H10",            "lambda" : 3798.98,  "emission": False, "label" : "H10"},
    {"name" : "H9",             "lambda" : 3836.48,  "emission": False, "label" : "H9"},
    {"name" : "H-zeta",         "lambda" : 3890.151, "emission": False, "label" : "H$\\zeta$" },
    {"name" : "K (Ca II 3933)", "lambda" : 3934.814, "emission": False, "label" : "K (Ca II)"},
    {"name" : "H (Ca II 3968)", "lambda" : 3969.623, "emission": False, "label" : "H (Ca II)"},
    {"name" : "H-epsilon",      "lambda" : 3971.195, "emission": False, "label" : "H$\\epsilon$"}, 
    {"name" : "H-delta",        "lambda" : 4102.892, "emission": False, "label" : "H$\\delta$" },
    {"name" : "G (Ca I 4307)",  "lambda" : 4308.952, "emission": False, "label" : "G (Ca I)"},
    {"name" : "H-gamma",        "lambda" : 4341.684, "emission": False, "label" : "H$\\gamma$"},
    {"name" : "H-beta",         "lambda" : 4862.683, "emission": False, "label" : "H$\\beta$"},
#   {"name" : "Mg I 5175",      "lambda" : 5176.441, "emission": False, "label" : "Mg I"},#Triplet
    {"name" : "Mg I 5183",      "lambda" : 5185.048, "emission": False, "label" : " "},
    {"name" : "Mg I 5172",      "lambda" : 5174.125, "emission": False, "label" : " "},
    {"name" : "Mg I 5167",      "lambda" : 5168.762, "emission": False, "label" : "Mg I"},
    {"name" : "D2 (Na I 5889)", "lambda" : 5891.582, "emission": False, "label" : " " },
    {"name" : "D1 (Na I 5895)", "lambda" : 5897.554, "emission": False, "label" : "D1,2 (Na I)" },
    {"name" : "H-alpha",        "lambda" : 6564.614, "emission": False, "label" : "H$\\alpha$"},
    ]
    
    if (ax == None):
        # If there is no axes given, plotting with the plt function
        ax = plt.gca()
    
    if (em_lines != None):
        # Choosing the emission lines listed by the user
        emission_lines = list(filter(lambda x: x['name'] in em_lines, emission_lines))
    
    if (abs_lines != None):
        # Choosing the absorption lines listed by the user
        absorption_lines = list(filter(lambda x: x['name'] in abs_lines, absorption_lines)) 
    
    xbounds = ax.get_xbound()   # Getting the x-range of the plot 
    # This is for selecting only those lines that are visible in the x-range of the plot
    
    for ii in range(len(emission_lines)):
        # If rest_frame = False, 
        # redshifting the emission lines to the observed frame of the source
        if (rest_frame == False):
            lam = emission_lines[ii]['lambda']*(1+z)
        else:
            lam = emission_lines[ii]['lambda']
        # Plotting the emission lines if they are within the x-range of the plot
        if (emission_lines[ii]['emission']) & (lam > xbounds[0]) & (lam < xbounds[1]):
            ax.axvline(lam, 0.95, 1.0, color = 'k', lw = 1.0)
            ax.axvline(lam, color = 'k', lw = 1.0, linestyle = ':')
            trans = ax.get_xaxis_transform()
            if (em_label == True):
                # Labeling the emission lines if em_label = True
                ax.annotate(emission_lines[ii]['label'], xy = (lam, 1.05), xycoords = trans, \
                         fontsize = 22, rotation = 90, color = 'k')
            
    for ii in range(len(absorption_lines)):
        # If rest_frame = False,
        # redshifting the absorption lines to the observed frame of the source
        if (rest_frame == False):
            lam = absorption_lines[ii]['lambda']*(1+z)
        else:
            lam = absorption_lines[ii]['lambda']
        # Plotting the absorption lines if they are within the x-range of the plot
        if (lam > xbounds[0]) & (lam < xbounds[1]):
            ax.axvline(lam, 0.2, 1.0, color = 'r', lw = 1.0, linestyle = ':')
            trans = ax.get_xaxis_transform()
            if (abs_label == True):
                # Labeling the absorption lines if abs_label = True
                ax.annotate(absorption_lines[ii]['label'], xy = (lam, 0.05), xycoords = trans, \
                         fontsize = 16, rotation = 90, color = 'r')

####################################################################################################
####################################################################################################