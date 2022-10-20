"""
This script consists of different functions related to Visual-Inspection of DESI spectra.
This script has the following functions:
    1) create_vi_files(table, specprod, survey, program, vi_folder)
    2) table_to_vifiles(table, vi_folder)

Author : Ragadeepika Pucha
Version : 2022 October 20

"""

####################################################################################################
####################################################################################################
import numpy as np
from astropy.table import Table

####################################################################################################
####################################################################################################

def create_vi_files(table, specprod, survey, program, vi_folder):
    """
    Function to create the files required to generate the VI html files.
    The generated files are within some "vi_folder" with the root folder:
    "/global/cfs/cdirs/desi/users/raga19/01_Dwarf_AGN_Project/desi_vi/"
    Each of the vi_folder needs to have three sub-folders:
     1) Files_for_VI -- where the *hpx*, *tgt* and *.sh files reside
     2) VI_Files -- html VI files go here
     3) VI_Results -- The final results of VI can be uploaded here
     
    Parameters
    ----------
    table : astropy table
        Table of sources whose spectra needs visual inspection
        
    specprod : str
        Spectral Release Name
        
    survey : str
        Survey name for the object. sv1|sv2|sv3|main|special
        
    program : str
        Observing program for the object. dark|bright|other|backup

    vi_folder : str
        Folder where the VI Files need to go
        
    Returns 
    -------
    None.
        
    """
    
    vi_root = '/global/cfs/cdirs/desi/users/raga19/01_Dwarf_AGN_Project/desi_vi'
    
    ## Files for the healpix and targetid lists
    hpx_file = open(f'{vi_root}/{vi_folder}/Files_for_VI/{specprod}_{survey}_{program}_hpx.txt', 'w')
    tgt_file = open(f'{vi_root}/{vi_folder}/Files_for_VI/{specprod}_{survey}_{program}_tgt.txt', 'w')
    
    for ii in range(len(table)):
        hpx_file.write(f'{table["HEALPIX"].data[ii]}\n')
        tgt_file.write(f'{table["TARGETID"].data[ii]}\n')
        
    hpx_file.close()
    tgt_file.close()
    
    ## bash scripts to run the VI files for the specific specprod, survey, and program
    vi_file = open(f'{vi_root}/{vi_folder}/Files_for_VI/{specprod}_{survey}_{program}.sh', 'w')
    
    vi_file.write('#!/bin/bash\n')
    vi_file.write('source /global/cfs/cdirs/desi/software/desi_environment.sh master\n\n')
    vi_file.write(f'OUTPUT_ROOT={vi_root}/{vi_folder}\n')
    vi_file.write('OUTPUT_DIR=${OUTPUT_ROOT}/VI_Files\n')
    vi_file.write('DATAPATH=${DESI_SPECTRO_REDUX}/'+specprod+'/healpix\n\n')
    
    vi_file.write(f'echo "{survey.upper()} {program.upper()}"\n')
    vi_file.write(f'HPX_FILE="{specprod}_{survey}_{program}_hpx.txt"\n')
    vi_file.write(f'TGT_FILE="{specprod}_{survey}_{program}_tgt.txt"\n\n')
    
    n = len(table)
    if (n > 1):
        vi_file.write('prospect_pages --datadir ${DATAPATH} \\'+'\n')
        vi_file.write('--dirtree_type healpix \\'+'\n')
        vi_file.write('--with_zcatalog \\'+'\n')
        vi_file.write('--template_dir /global/common/software/desi/cori/desiconda/20211217-2.0.0/'+
                      'code/redrock-templates/0.7.2/ \\'+'\n')
        vi_file.write('--outputdir ${OUTPUT_DIR} \\'+'\n')
        vi_file.write(' --pixel_list ${HPX_FILE} --target_list ${TGT_FILE} \\'+'\n')                          
        vi_file.write(f'--survey_program {survey} {program} \\'+'\n')
        vi_file.write(f'--titlepage_prefix {specprod}_{survey}_{program}_vi')
    else:
        hpx = table['HEALPIX'].data[0]
        tgt = table['TARGETID'].data[0]
        vi_file.write('prospect_pages --spectra_files ${DATAPATH}/'+
            f'{survey}/{program}/{hpx//100}/{hpx}/coadd-{survey}-{program}-{hpx}.fits \\'+'\n')
        vi_file.write('--zcat_files ${DATAPATH}/'+
            f'{survey}/{program}/{hpx//100}/{hpx}/redrock-{survey}-{program}-{hpx}.fits \\'+'\n')
        vi_file.write('--with_zcatalog \\'+'\n')
        vi_file.write(f'--targets {tgt} \\'+'\n')
        vi_file.write('--template_dir /global/common/software/desi/cori/desiconda/20211217-2.0.0/'+
                      'code/redrock-templates/0.7.2/ \\'+'\n')
        vi_file.write('--outputdir ${OUTPUT_DIR} \\'+'\n')
        vi_file.write(f'--titlepage_prefix {specprod}_{survey}_{program}_vi')
        
    vi_file.close()

####################################################################################################
####################################################################################################

def table_to_vifiles(table, vi_folder):
    """
    Function to create the files required to generate the VI html files from a given table.
    It also creates the overall run_vi file -- that can be used to get all the required vi files.
    The generated files are within some "vi_folder" with the root folder:
    "/global/cfs/cdirs/desi/users/raga19/01_Dwarf_AGN_Project/desi_vi/"
    Each of the vi_folder needs to have three sub-folders:
     1) Files_for_VI -- where the *hpx*, *tgt* and *.sh files reside
     2) VI_Files -- html VI files go here
     3) VI_Results -- The final results of VI can be uploaded here
     
    Parameters
    ----------
    table : astropy table
        Table of sources whose spectra needs visual inspection

    vi_folder : str
        Folder where the VI Files need to go
        
    Returns 
    -------
    None.
    """
    
    vi_root = '/global/cfs/cdirs/desi/users/raga19/01_Dwarf_AGN_Project/desi_vi'
    
    run_file = open(f'{vi_root}/{vi_folder}/Files_for_VI/run_vi.sh', 'w')
    run_file.write('#!/bin/bash\n\n')
    
    ## List of all unique specprods, surveys, and programs in the table
    
    specprods = np.unique(table['SPECPROD'].astype(str).data)
    surveys = np.unique(table['SURVEY'].astype(str).data)
    programs = np.unique(table['PROGRAM'].astype(str).data)
    
    for specprod in specprods:
        for survey in surveys:
            for program in programs:
                tab = table[(table['SPECPROD'].astype(str) == specprod)&\
                            (table['SURVEY'].astype(str) == survey)&\
                            (table['PROGRAM'].astype(str) == program)]
                if (len(tab) != 0):
                    create_vi_files(table = tab, specprod = specprod,\
                                    survey = survey, program = program, vi_folder=vi_folder)
                    run_file.write(f'source {specprod}_{survey}_{program}.sh\n')
    run_file.close()
    
####################################################################################################
####################################################################################################