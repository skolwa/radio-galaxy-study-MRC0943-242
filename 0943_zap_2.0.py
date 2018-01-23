# 		S.N. Kolwa (2017)
#		0943_zap_2.0.py
# 		Purpose: 
# 			-Sky-subtract datacube using Zurich Atmosphere Purge (ZAP) 2.0 (Soto et al 2016)

# ---------
#  modules
# ---------

import zap

#implement procedure
fname = '/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP.fits'
zap.process('/Users/skolwa/DATA/MUSE_data/0943-242/DATACUBE_FINAL.fits',outcubefits=fname,overwrite=True)