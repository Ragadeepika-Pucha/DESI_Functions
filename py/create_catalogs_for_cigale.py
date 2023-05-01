"""
This script generates the starting catalog for computing stellar masses for a given 
spectral production. Input: spectral production name: fuji|guadalupe|iron|...

Note: Run "source /global/cfs/cdirs/desi/software/desi_environment.sh master" 
before running this script from the terminal.

Author : Ragadeepika Pucha
Version : 2023 April 30
"""

####################################################################################################
import sys
from datetime import date
import time
import numpy as np

from astropy.table import Table, Column, join

from desispec.io import specprod_root

import warnings
warnings.filterwarnings('ignore')
####################################################################################################

## Starting time
start = time.time()

## Spectral production pipeline name given as input
specprod = str(sys.argv[1])

## File for README
readme = open(f'ReadMe-{specprod}.txt', 'w')

## Initial Setting up of the README file
readme.write(f'# This file contains information about how the {specprod} catalog for CIGALE is generated.\n')
readme.write(f'# Author : Ragadeepika Pucha\n')

## Current date
today = date.today()
readme.write(f'# Version : {today.strftime("%Y, %B %d")}\n\n')

readme.write('####################################################################################################\n\n')

## Specprod directory
specprod_dir = specprod_root(specprod)

## Zcatalog directory
zcat_dir = f'{specprod_dir}/zcatalog'
zcat_file = f'zall-pix-{specprod}.fits'
zcat = Table.read(f'{zcat_dir}/{zcat_file}')

## ReadMe Update
readme.write(f'* Started with {zcat_file} catalog ({len(zcat)} rows)\n\n')

## Applying the cuts
## zcat_primary = True
## coadd_fiberstatus = 0
## zwarn = 0 | zwarn = 4
## spectype = GALAXY | SPECTYPE = QSO

good_spec = (zcat['ZCAT_PRIMARY'])&(zcat['COADD_FIBERSTATUS'] == 0)&((zcat['ZWARN'] == 0)|(zcat['ZWARN'] == 4))&\
((zcat['SPECTYPE'] == 'GALAXY')|(zcat['SPECTYPE'] == 'QSO'))

zcat_sel = zcat[good_spec]['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'SPECTYPE',\
                           'DESI_TARGET', 'Z', 'TARGET_RA', 'TARGET_DEC']

## ReadMe Update
readme.write('* Applied the following cuts:\n\t- ZCAT_PRIMARY = True\n\t- COADD_FIBERSTATUS = 0\n\t'+\
             '- ZWARN = 0 | ZWARN = 4\n\t- SPECTYPE = GALAXY | SPECTYPE = QSO\n\n')

readme.write(f'* Number of primary spectra of GALAXY|QSO with good spectra and good redshifts = {len(zcat_sel)}/{len(zcat)}\n\n')


## LS DR9 Photometry
if (specprod == 'fuji'):
    dr = 'edr'
    ver = 'v2.0'
elif (specprod == 'guadalupe'):
    dr = 'dr1'
    ver = 'v2.0'
else:
    dr = 'dr1'
    ver = 'v1.0'
    
ls_dir = f'/global/cfs/cdirs/desi/public/{dr}/vac/lsdr9-photometry/{specprod}/{ver}/observed-targets'
ls_file = f'targetphot-{specprod}.fits'

## LS DR9 Photometry Catalog
ls_tab = Table.read(f'{ls_dir}/{ls_file}')

ls_columns = ['TARGETID', 'SURVEY', 'PROGRAM', 'RELEASE', \
              'FLUX_G', 'FLUX_R', 'FLUX_Z', \
              'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', \
              'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', \
              'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4', \
              'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', \
              'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4', \
              'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z']

## Valid DR9 photometry
dr9_phot = (ls_tab['RELEASE'] > 9000)

ls_tab_sel = ls_tab[dr9_phot][ls_columns]

## Join redshift catalog with DR9 photometry
zcat_ls = join(zcat_sel, ls_tab_sel, keys = ['TARGETID', 'SURVEY', 'PROGRAM'])

## Remove duplicates
_, ii = np.unique(zcat_ls['TARGETID'].data, return_index = True)
zcat_lsdr9 = zcat_ls[ii]

## ReadMe Update
readme.write('* Join with LS Targetphot VAC and remove any duplicates.\n Selected only candidates with valid LS DR9 photometry (RELEASE > 9000).\n\n')
readme.write(f'* This resulted in {len(zcat_lsdr9)} unique sources.\n\n')

## Add Spectra Path

hpx_dir = f'{specprod_dir}/healpix/'

def create_spectra_path(table):
    hpx_directory = np.array([hpx_dir]*len(table)).astype(object)
    survey = table['SURVEY'].astype(str).data.astype(object)
    program = table['PROGRAM'].astype(str).data.astype(object)
    hpx = table['HEALPIX'].data
    hpx_group = hpx//100
    hpx = hpx.astype(str).astype(object)
    hpx_group = hpx_group.astype(str).astype(object)
    slash = np.array(['/']*len(table)).astype(object)
    coadd = np.array(['coadd-']*len(table)).astype(object)
    hyphen = np.array(['-']*len(table)).astype(object)
    fits = np.array(['.fits']*len(table)).astype(object)

    path = hpx_directory+survey+slash+program+slash+hpx_group+slash+hpx+slash+coadd+survey+hyphen+program+hyphen+hpx+fits
    path = path.astype(str)
    
    column = Column(path, name = 'Spectra_Path')
    table.add_column(column)
    
    return (table)

## Final catalog
tfinal = create_spectra_path(zcat_lsdr9)

## Final catalog outfile name
outfile = f'{specprod}-forCIGALE.fits'

## Save the catalog
tfinal.write(outfile, overwrite = True)

## ReadMe Update
readme.write('* Added "Spectra_Path" column to the table.\n\n')
readme.write(f'* The final table is saved as {outfile}.\n\n')

end = time.time()
readme.write(f'* Time taken: {round(end-start, 2)} seconds\n\n')

readme.write('####################################################################################################\n\n')

readme.close()







