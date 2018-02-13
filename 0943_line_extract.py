#		S.N. Kolwa (2017)
#		0943_line_extract.py
# 		Purpose: 
# 			- Create subcube for spatial region of interest i.e. radio galaxy
# 			- Extract subcubes for spectral regions isolated line species

#source-specific input: 
#1. wavelength-ranges for line extraction
#2. size of truncated field from full datacube of observation

# ---------
#  modules
# ---------

import matplotlib.pyplot as pl
import numpy as np 

from astropy.io import fits
import spectral_cube as sc
import mpdaf.obj as mpdo
import mpdaf

import warnings
from astropy.utils.exceptions import AstropyWarning

import sys
import time

start_time = time.time()

spec_feat 	= ['Lya','NV','CII','SiIV','NIV]','CIV','HeII','OIII]','CIII]','CII]']
lam1 		= [4600.,4818.,5015.,5276.,5640., 5842., 6135., 6468.,  6632.,  8994.]
lam2 		= [4990.,5026.,5468.,5792.,6020., 6334., 6828., 6626.,  8520.,  9302.]

#ignore those pesky warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

#import astrometry corrected, sky subtracted cube
fname		= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrocorr.fits"
cube		= mpdo.Cube(fname,mmap=True)

#radio galaxy and CGM subcube
rg 			= cube[:,155:300,60:240]

fname 	= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_glx_line.fits"
rg.write(fname)

spec 	= rg.sum( axis=(1,2) )
for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):
	print "Extracting line subcube for "+spec_feat+" ..."

	m1,m2 	= spec.wave.pixel( [lam1,lam2], nearest=True ) 
	emi 	= rg[m1:m2,:,:]

	emi.write('./out/'+spec_feat+'.fits')

	#duration of process
	elapsed = (time.time() - start_time)
	print "build time: %f s" % elapsed	