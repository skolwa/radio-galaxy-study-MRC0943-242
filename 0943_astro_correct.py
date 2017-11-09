# S.N. Kolwa (2017)
# Purpose: 
# 0943_astro_correct.py
# -Sky-subtract datacube using ZAP
# -Astrometry-correct using GAIA DR1

import numpy as np
from math import*

import pyfits as pf
import zap

# zap.process('/Users/skolwa/DATA/MUSE_data/0943-242/DATACUBE_FINAL.fits',interactive=True)

#identify std stars in DS9 and write into MUSE catalogue
#cross-match MUSE and GAIA (web) to get this table:
numbers = np.loadtxt('/Users/skolwa/PHD_WORK/catalogues/field_MRC0943_GAIA.txt')

#offset between GAIA and MUSE co-ordinates in degrees
ra_offset = numbers[:,19]
dec_offset = numbers[:,20]

av_ra_offset = np.mean(ra_offset)
av_dec_offset = np.mean(dec_offset)

fname = '/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP.fits'

hdulist = pf.open(fname)

print hdulist.info()

prihdr = hdulist[1].header

#import current crvals (deg)
crval1 = prihdr[29]
crval2 = prihdr[30]

#original crvals i.e. MUSE
print crval1,crval2

prihdr['crval1'] = crval1 + av_ra_offset
prihdr['crval2'] = crval2 + av_dec_offset

#updated crvals
print prihdr[29],prihdr[30]

hdulist.writeto('/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrocorr.fits',clobber=True)
