#S.N. Kolwa (2018)
#hzrgs_zap_2.0.py
#Purpose: 
# -Sky-subtract datacube using Zurich Atmosphere Purge (ZAP) 2.0 (Soto et al 2016)

import sys
import zap

zapcube 	= {
'4C19' : '4C19.71/4C19_71_ZAP.fits', 
'1320' : 'J0121+1320/J0121_1320_ZAP.fits',
'2422' : 'J0205+2422/J0205_2422_ZAP.fits',
'0943' : '0943-242/MRC0943_ZAP.fits', 
'4C04' : '4C04.11/4C04_11_ZAP.fits' }

cube 		= {
'4C19' : '4C19.71/DATACUBE_FINAL.fits', 
'1320' : 'J0121+1320/DATACUBE_FINAL.fits', 
'2422' : 'J0205+2422/DATACUBE_FINAL.fits',
'0943' : '0943-242/DATACUBE_FINAL.fits', 
'4C04' : '4C04.11/DATACUBE_FINAL.fits' }

#pick out 
key = sys.argv[1]
zapcube = [ zapcube[key] ]
cube = [ cube[key] ]

#implement procedure
for zapcube,cube in zip(zapcube,cube):
	zapcube = '/Users/skolwa/DATA/MUSE_data/'+zapcube
 	zap.process('/Users/skolwa/DATA/MUSE_data/'+cube,outcubefits=zapcube,overwrite=True)