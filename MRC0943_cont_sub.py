# S.N. Kolwa 
# ESO (2017)
# MRC0943_cont_sub.py

# Purpose: 
# - Extract spatial subcube i.e. Radio Galaxy
# - Subtract the continuum around line feature
# - Output continuum-subtracted cube per spectral feature
# - rest-frame UV lines detected by MUSE 
# - Hardcoded: 
#   extracted and masked wavelength-ranges

import matplotlib.pyplot as pl
import numpy as np 

import astropy.units as u
import mpdaf.obj as mpdo
from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning

import time
import sys

start_time = time.time()

home = sys.argv[1]

spec_feat = ['Lya']
lam1 = [4680.]
lam2 = [4848.]
mask1 = [4714.]
mask2 = [4820.]

for spec_feat,lam1,lam2,mask1,mask2 in zip(spec_feat,lam1,lam2,mask1,mask2):
	#ignore those pesky warnings
	print spec_feat
	warnings.filterwarnings('ignore', category=UserWarning, append=True)
	warnings.simplefilter('ignore', category=AstropyWarning)
	
	#----------------
	# LOAD data cubes
	#----------------
	#import astrometry corrected, sky subtracted cube
	fname = home+"/DATA/MUSE_data/MRC0943-242/MRC0943_ZAP_astrocorr.fits"
	cube = mpdo.Cube(fname,mmap=True)
	
	#radio galaxy and CGM subcube
	rg = cube[:,190:200,120:130]

	fname = home+"/DATA/MUSE_data/MRC0943-242/MRC0943_glx_cont.fits"

	#------------------------------------
	#  CONTINUUM-SUBTRACT LINE EMISSION 
	#------------------------------------
	#extract continuum around ONE spectral line/feature
	spec 	= rg.sum( axis=(1,2) )
	m1,m2 	= spec.wave.pixel( [lam1,lam2], nearest=True ) 	
	emi 	= rg[m1:m2+1,:,:]								

	print 'Initialising empty cube onto which we write the continuum solution...'

	cont 		= emi.clone(data_init = np.empty, var_init = np.empty) 	# empty cube with same dim
	emi_copy 	= emi.copy()

	cont.write(home+'/DATA/MUSE_data/MRC0943-242/empty_cube.fits')	
	emi_copy.write(home+'/DATA/MUSE_data/MRC0943-242/copy.fits')										# copy of wavelength truncated subcube

	print 'Masking copy of sub-cube...'

	for sp in mpdo.iter_spe(emi_copy):
		sp.mask_region(lmin=mask1,lmax=mask2)

	print 'Calculating continuum subtraction solution...'

	#continuum solution 
	for sp,co in zip(mpdo.iter_spe(emi_copy),mpdo.iter_spe(cont)):
		co[:] = sp.poly_spec(1)

	print 'Subtracting continuum...'

	#continuum-subtract
	line = emi - cont

	print 'Writing continuum, subtracted and unsubtracted cubes to disk...'
				
	#duration of process
	print 'Extracted continuum-subtracted line subcube for '+spec_feat+ '...'
	elapsed = (time.time() - start_time)/60.

elapsed = (time.time() - start_time)/60.
print "Process complete. Total build time: %f mins" % elapsed	

