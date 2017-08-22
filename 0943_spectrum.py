#S.N. Kolwa (2017)
#0943_line_fit.py
# Purpose:  
# - Plot the full MUSE spectrum of 0943-242's AGN host galaxy
# - Name the line identifiers
# - Line wavelengths are approximated from rest wavs and known redshift:
#   0943-242 redshift = 2.923

import matplotlib.pyplot as pl
import matplotlib.text as mpt
import numpy as np

import spectral_cube as sc
import astropy.units as u
import mpdaf.obj as mpdo

import matplotlib.ticker as tk

import warnings
from astropy.utils.exceptions import AstropyWarning

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

def obs_wav(z,lam_rest):
	lam_e = lam_rest*(1.+z)
	return lam_e

#0943 detected lines
wavs = [ 1215.7,1238.8,1242.8,1338.0,1402.8,1486.5,1548.2,1550.8,1640.4,1660.8,1666.1,1906.7,1908.7,2326.0 ]

wav_obs = [ obs_wav(2.923,wavs[i]) for i in range(len(wavs)) ]
ID = [ 'Lya', 'NV', 'NV', 'CII','SiIV', 'NIV]', 'CIV', 'CIV', 'HeII', 'OIII]', 'OIII]', 'CIII]', 'CIII]', 'CII]' ]

wavs_data = np.array( zip(wavs,wav_obs), dtype=[('ID', np.float64), ('lam_e', np.float64)] )

np.savetxt('./out/0943_spectrum.txt',wavs_data, fmt=['%.2f']+['%.2f'], header='Assuming Unshifted Line Centre from Vsys\nwav_e: rest-frame wavelength\nwav_o: redshifted wavelength\n\nwav_e   wav_o')

##load data cubes
##spectral-cube load more durable than mpdaf load
cube_ = sc.SpectralCube.read("/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_ZAP_astrom_corr.fits",hdu=1,format='fits')
# print cube

#extract sky-region of interest
spec_cube 	= cube_[:,185:285,120:220]		
spec_cube.write("/Users/skolwa/DATA/MUSE_data/0943-242/0943_spec_cube.fits", overwrite=True)

##mpdaf load subcube to use MUSE specific functions
cube = mpdo.Cube("/Users/skolwa/DATA/MUSE_data/0943-242/0943_spec_cube.fits")
# cube.info()

fig = pl.figure(figsize=(18,8))
muse_0943_spec = cube.subcube_circle_aperture(center=(50,45),radius=10,unit_center=None,unit_radius=None)
muse_spec = muse_0943_spec.sum(axis=(1,2))
muse_spec.plot(color='purple')

# fig = pl.figure()
# fig.add_subplot(111)
# muse_img = muse_0943_spec.sum(axis=0)
# muse_img.plot(scale='linear')

#define axes
ax = pl.gca()
data = ax.lines[0]

flux = data.get_ydata()
ymax = max(flux) + 5.e3
y_label = max(flux) + 1.e3

# label spectrum with known features
#assuming centre of line is at rest w.r.t. systemic velocity i.e. HeII
pl.plot( [wav_obs[0],wav_obs[0]],[-250.,ymax], color='red', ls='--',lw=0.3)
pl.text(4720.2,y_label,r'Ly$\alpha$',fontsize=10,rotation='90',va='bottom',color='red')

pl.plot( [wav_obs[1],wav_obs[1]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(4816.8,y_label,'NV doublet',fontsize=10,rotation='90',va='bottom')

pl.plot( [wav_obs[2],wav_obs[2]],[-250.,ymax], color='black', ls='--',lw=0.3)

pl.plot( [wav_obs[3],wav_obs[3]],[-250.,ymax], color='black', ls='--',lw=0.3 )
pl.text(5209.0,y_label,'CII',fontsize=10,rotation='90',va='bottom')

pl.plot( [wav_obs[4],wav_obs[4]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(5458.2,y_label,'SiIV',fontsize=10,rotation='90',va='bottom')

pl.plot( [wav_obs[5],wav_obs[5]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(5784.5,y_label, 'NIV]', rotation='90',fontsize=10,va='bottom')

pl.plot( [wav_obs[6],wav_obs[6]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(6028.6,y_label,'CIV doublet', rotation='vertical',fontsize=10,va='bottom')

pl.plot( [wav_obs[7],wav_obs[7]],[-250.,ymax], color='black', ls='--',lw=0.3)

pl.plot( [wav_obs[8],wav_obs[8]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(6382.3,y_label, 'HeII',rotation='vertical',fontsize=10,va='bottom')

pl.plot( [wav_obs[9],wav_obs[9]],[-250.,ymax], color='black', ls='--',lw=0.3)
pl.text(6468.3,y_label,'OIII] doublet', rotation='vertical',fontsize=10,va='bottom')

pl.plot( [wav_obs[10],wav_obs[10]],[-250.,ymax], color='black', ls='--',lw=0.3)

pl.plot( [wav_obs[11],wav_obs[11]],[-250.,ymax], color='black', ls='--',lw=0.2)
pl.text(7436.8,y_label,'CIII] doublet',rotation='90',fontsize=10,va='bottom')

pl.plot( [wav_obs[12],wav_obs[12]],[-250.,ymax], color='black', ls='--',lw=0.2)

pl.plot( [wav_obs[13],wav_obs[13]],[-250.,ymax], color='black',ls='--',lw=0.3)
pl.text(9074.9,y_label,'CII]',rotation='90',fontsize=10,va='bottom')

ax.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: '%d'%(x*1.e-3) ) )

pl.xlabel(r"Wavelength ($\AA$)")
pl.ylabel(r"Flux Density (10$^{-17}$ erg / s / cm$^2$ / $\AA$)")
pl.plot([0.,9300.],[0.,0.],color='black',ls='--')
pl.ylim([-250.,ymax])
pl.savefig("/Users/skolwa/PUBLICATIONS/0943_halo_environment/plots/0943_spectrum.eps")
pl.title("0943-242 Galaxy Spectrum")
pl.savefig("./out/0943-242 spectrum.png")
