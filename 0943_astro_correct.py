# 		S.N. Kolwa (2017)
# 		0943_astro_correct.py
# 		Purpose: 
# 			-Astrometry-correct using GAIA DR1

# ---------
#  modules
# ---------

import numpy as np
from math import*

import pyfits as pf

fname = '/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP.fits'

#obtain GAIA catalogue for std stars in field 
GAIA_std_stars = np.genfromtxt('/Users/skolwa/PHD_WORK/catalogues/MRC0943_GAIA.tab',dtype=None)

n = len(GAIA_std_stars)

gaia_ra 	= [ GAIA_std_stars[i][0] for i in range(n) ]
gaia_dec 	= [ GAIA_std_stars[i][1] for i in range(n) ]

#manually approximate the co-ordinates of standard stars
MUSE_std_stars =\
 [ (146.39072,-24.477007), (146.38229,-24.483344),\
 (146.38076,-24.484112), (146.37759,-24.486223),\
  (146.38471,-24.489334) ]

muse_ra = [ MUSE_std_stars[i][0] for i in range(n) ]
muse_dec = [ MUSE_std_stars[i][1] for i in range(n) ]

#determine relative offsets between MUSE and GAIA astrometry
ra_offset 	= [ gaia_ra[i] - muse_ra[i] for i in range(n) ]
dec_offset 	= [ gaia_dec[i] - muse_dec[i] for i in range(n) ]

#get average offsets
av_ra_offset = np.mean(ra_offset)
av_dec_offset = np.mean(dec_offset)

#astrometry correct
hdulist = pf.open(fname)

# print hdulist.info()

prihdr = hdulist[1].header

#import current crvals (deg)
crval1 = prihdr[29]
crval2 = prihdr[30]

#original crvals i.e. MUSE
print crval1,crval2

#gaia - muse = offset => gaia = muse + offset  
prihdr['crval1'] = crval1 + av_ra_offset
prihdr['crval2'] = crval2 + av_dec_offset

#updated crvals
print prihdr[29],prihdr[30]

hdulist.writeto('/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrocorr.fits',clobber=True)