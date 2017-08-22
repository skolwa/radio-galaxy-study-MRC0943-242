#S.N. Kolwa (2017)
#0943_line_extract.py
# Purpose: 
# - Extract spatial region of interest i.e. AGN host galaxy
# - Extract spectral region i.e. line emission 
# - No continuum-subtraction done on line spectra
# - Pure line-extraction 

import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np 

import spectral_cube as sc
import astropy.units as u
from astropy.io import fits
from wcsaxes import WCSAxes
from astropy.wcs import WCS
import mpdaf.obj as mpdo

import warnings
from astropy.utils.exceptions import AstropyWarning
import sys
import time

import itertools as it

start_time = time.time()

spec_feat 	= ['HeII','CIII]','CII]','NV','CII','SiIV','NIV]','OIII]','CIV','Lya']
lam1    		= [6370.,  7420.,  9055.,4805.,5200.,5415., 5778., 6470.,  6000.,4700.]
lam2    		= [6495.,  7545.,  9185.,4940.,5290.,5585., 5885., 6580.,  6150.,4835.]

for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):
	#ignore those pesky warnings
	warnings.filterwarnings('ignore', category=UserWarning, append=True)
	warnings.simplefilter('ignore', category=AstropyWarning)
	# print spec_feat,lam1,lam2
	
	#----------------
	# LOAD data cubes
	#----------------
	##spectral-cube load more durable than mpdaf load
	cube_ 		= sc.SpectralCube.read("/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrom_corr.fits",hdu=1,format='fits')
	spec_cube 	= cube_[:,185:285,120:220]	
	fname = "/Users/skolwa/DATA/MUSE_data/0943-242/0943_spec_cube.fits"
	spec_cube.write(fname, overwrite=True)
	
	##mpdaf load subcube to use MUSE specific functions
	cube 		= mpdo.Cube(fname)
	# cube.info()
	
	#------------------
	#  LINE EMISSION
	#------------------
	#isolate continuum around the spectral line feature
	m1,m2 	= cube.wave.pixel([lam1,lam2], nearest=True)
	emi 	= cube[m1:m2+1,:,:]
	
	emi.write('./out/'+spec_feat+'.fits')