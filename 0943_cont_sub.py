# S.N. Kolwa (2017)
# 0943_cont_sub.py
# Purpose: 
# - Extract spatial subcube i.e. AGN host galaxy
# - Subtract the continuum around line feature
# - Output continuum-subtracted cube per spectral feature

import matplotlib.pyplot as pl
import numpy as np 

import spectral_cube as sc
import astropy.units as u

from astropy.io import fits

import mpdaf.obj as mpdo

import warnings
from astropy.utils.exceptions import AstropyWarning

import time

start_time = time.time()

spec_feat 	= [ 'Lya',  'NV', 'CII', 'SiIV', 'NIV]', 'CIV','HeII','OIII]','CIII]','CII]']
lam1 		= [ 4680., 4820., 5158., 5350., 5656., 5918., 6368.,  6480.,  7110., 9005.]
lam2		= [ 4848., 4914., 5332., 5664., 5984., 6234., 6494.,  6575.,  7888., 9260.]
mask1		= [ 4714., 4836., 5225., 5448., 5778., 6038., 6400.,  6495.,  7438., 9078.]
mask2		= [ 4820., 4890., 5265., 5552., 5880., 6120., 6468.,  6554.,  7530., 9170.]

for spec_feat,lam1,lam2,mask1,mask2 in zip(spec_feat,lam1,lam2,mask1,mask2):
	#ignore those pesky warnings
	warnings.filterwarnings('ignore', category=UserWarning, append=True)
	warnings.simplefilter('ignore', category=AstropyWarning)
	
	#----------------
	# LOAD data cubes
	#----------------
	##astrometry-corrected sky-subtracted cube
	cube_ 		= sc.SpectralCube.read("/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrom_corr.fits",hdu=1,format='fits')
	
	#radio galaxy region crop
	spec_cube 	= cube_[:,185:285,120:220]	
	#write out product under new fname
	spec_cube.write("/Users/skolwa/DATA/MUSE_data/0943-242/0943_spec_cube.fits", overwrite=True)
	
	##mpdaf load subcube to use MUSE-specific functions
	fname 		= "/Users/skolwa/DATA/MUSE_data/0943-242/0943_spec_cube.fits"
	cube 		= mpdo.Cube(fname)
	# cube.info() #query cube information
	
	#------------------------------------
	#  CONTINUUM-SUBTRACT LINE EMISSION 
	#------------------------------------
	#extract continuum around ONE spectral line/feature
	spec1 	= cube.sum(axis=(1,2))
	m1,m2 	= spec1.wave.pixel([lam1,lam2], nearest=True) 	#finds nearest cube-pixel to wavelengths lam1,lam2
	emi 	= cube[m1:m2+1,:,:]								#extract spectral slab
	emi.write('./out/'+spec_feat+'.fits')					#save it as fits file

	cont 		= emi.clone(data_init = np.empty, var_init = np.empty)
	emi_copy 	= emi.copy()

	for sp in mpdo.iter_spe(emi_copy):
		sp.mask_region(lmin=mask1,lmax=mask2)

	for sp,co in zip(mpdo.iter_spe(emi_copy),mpdo.iter_spe(cont)):
		co[:] = sp.poly_spec(1)

	line = emi - cont

	line.write('./out/'+spec_feat+'_cs.fits')

	# ap = line.subcube_circle_aperture(center=(50,44),radius=5,unit_center=None,unit_radius=None)	
	# spec = ap.sum(axis=(1,2))
	# img  = ap.sum(axis=0)

	# fig = pl.figure()
	# fig.add_subplot(211)
	# img.plot(scale='linear')
	# fig.add_subplot(212)
	# spec.plot(color='black')
	# pl.savefig('./out/'+spec_feat+'test_cs.png')
	# # pl.show()

#duration of process
elapsed = (time.time() - start_time)/60.
print "build time: %f mins" % elapsed	