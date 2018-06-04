# S.N. Kolwa (2017)
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

import spectral_cube as sc
import astropy.units as u
import mpdaf.obj as mpdo
from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning

import time

start_time = time.time()

#wavelength ranges determined via visual inspection in QFitsView
spec_feat 		= [ 'Lya', 'NV', 'CII', 'SiIV', 'NIV]','CIV','HeII', 'OIII]','CIII]','CII]']
lam1 			= [ 4680., 4820., 5158., 5350., 5656., 5918., 6368.,  6480.,  7110., 9005.]
lam2			= [ 4848., 4914., 5332., 5664., 5984., 6234., 6494.,  6575.,  7888., 9260.]
mask1			= [ 4714., 4846., 5225., 5448., 5778., 6038., 6400.,  6508.,  7438., 9078.]
mask2			= [ 4820., 4890., 5265., 5552., 5880., 6120., 6468.,  6552.,  7530., 9170.]

for spec_feat,lam1,lam2,mask1,mask2 in zip(spec_feat,lam1,lam2,mask1,mask2):
	#ignore those pesky warnings
	print spec_feat
	warnings.filterwarnings('ignore', category=UserWarning, append=True)
	warnings.simplefilter('ignore', category=AstropyWarning)
	
	#----------------
	# LOAD data cubes
	#----------------
	#import astrometry corrected, sky subtracted cube
	fname		= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrocorr.fits"
	cube		= mpdo.Cube(fname,mmap=True)
	
	#radio galaxy and CGM subcube
	rg 			= cube[:,190:290,120:220]
	
	fname 	= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_glx_cont.fits"
	rg.write(fname)

	#------------------------------------
	#  CONTINUUM-SUBTRACT LINE EMISSION 
	#------------------------------------
	#extract continuum around ONE spectral line/feature
	spec 	= rg.sum( axis=(1,2) )
	m1,m2 	= spec.wave.pixel([lam1,lam2], nearest=True) 	
	emi 	= rg[m1:m2+1,:,:]								

	print 'Initialising empty cube onto which we write the continuum solution...'

	cont 		= emi.clone(data_init = np.empty, var_init = np.empty) 	# empty cube with same dim
	emi_copy 	= emi.copy()											# copy of wavelength truncated subcube

	print 'Masking copy of wavelength truncated cube...'

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
				
	line.write('./out/'+spec_feat+'_cs.fits')

	#duration of process
	print 'Extracted continuum-subtracted line subcube for '+spec_feat+ '...'
	elapsed = (time.time() - start_time)/60.

elapsed = (time.time() - start_time)/60.
print "Process complete. Total build time: %f mins" % elapsed	

