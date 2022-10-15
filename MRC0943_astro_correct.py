# S.N. Kolwa
# ESO (2017)
# MRC0943_astro_correct.py

# Purpose: 
# -Astrometry-correct using GAIA DR2

import numpy as np
from math import*

from astropy.io import fits

home = sys.argv[1]
fname = home+'DATA/MUSE_data/0943-242/MRC0943_ZAP.fits'

#obtain GAIA catalogue for std stars in field 
GAIA_std_stars = np.genfromtxt(home+'PHD_WORK/catalogues/field_MRC0943_GAIA_DR2.txt', usecols=(0,1))

n = len(GAIA_std_stars)

gaia_ra 	= [ GAIA_std_stars[i][0] for i in range(n) ]
gaia_dec 	= [ GAIA_std_stars[i][1] for i in range(n) ]

#manually approximate the co-ordinates (in degrees) of standard stars (detected in DR2)
MUSE_std_stars = \
[ (146.38085, -24.484034), 
  (146.38478, -24.489314),
  (146.38231, -24.483294),
  (146.38934, -24.485921), 
  (146.39074, -24.477036), 
  (146.38821, -24.486365), 
  (146.37766, -24.486213), 
  (146.38769, -24.488350) 
]

muse_ra = [ MUSE_std_stars[i][0] for i in range(n) ]
muse_dec = [ MUSE_std_stars[i][1] for i in range(n) ]

#determine relative offsets between MUSE and GAIA astrometry
ra_offset 	= [ gaia_ra[i] - muse_ra[i] for i in range(n) ]
dec_offset 	= [ gaia_dec[i] - muse_dec[i] for i in range(n) ]

#get average offsets
av_ra_offset = np.mean(ra_offset)
av_dec_offset = np.mean(dec_offset)

#astrometry correct
hdulist = fits.open(fname)

#import current crvals (deg)
crval1 = hdr['crval1']
crval2 = hdr['crval1']

#gaia - muse = offset => gaia = muse + offset  
hdr['crval1'] = crval1 + av_ra_offset
hdr['crval2'] = crval2 + av_dec_offset

hdulist.writeto(home+'DATA/MUSE_data/0943-242/MRC0943_ZAP_astrocorr.fits',clobber=False)