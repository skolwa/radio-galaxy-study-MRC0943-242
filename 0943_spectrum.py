#			S.N. Kolwa (2017)
#			0943_line_fit.py
# 			Purpose:  
# 				- Plot the full MUSE spectrum of 0943-242's AGN host galaxy
# 				- Name the line identifiers
# 				- Line wavelengths are approximated from rest wavs and known redshift:
#   			0943-242 redshift = 2.923
#			Note: ** denotes sections of code that are individual to a source

# ---------
#  modules
# ---------

import matplotlib.pyplot as pl
import numpy as np

import spectral_cube as sc
import mpdaf.obj as mpdo

import matplotlib.ticker as tk

import warnings
from astropy.utils.exceptions import AstropyWarning

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

# -----------------------------------
#   Estimate Observed Wavelengths
# -----------------------------------
def obs_wav(z,lam_rest):
	lam_e = lam_rest*(1.+z)
	return lam_e

#rest wavelengths of identified lines **
wav_rest = \
[ 
('Lya', 1215.7), 
('NV',1238.8),('NV',1242.8),
('CII',1338.0),
('SiIV',1393.8),('SiIV',1402.8),
('NIV]',1486.5),
('CIV',1548.2),('CIV',1550.8),
('HeII',1640.4),
('OIII]',1660.8),('OIII]',1666.1),
('CIII]',1906.7),('CIII]',1908.7),
('CII]',2326.0) 
]

N = len(wav_rest)
print N

#estimated redshift from literature **
z_est = 2.923	

#redshifted wavelengths
line 		= [ wav_rest[i][0] for i in range(N) ]
wav_em 		= [ wav_rest[i][1] for i in range(N) ]
wav_obs 	= [ obs_wav(z_est,wav_rest[i][1]) for i in range(N) ]

print wav_obs

wavs_data = np.array( zip(line,wav_em,wav_obs), \
	dtype=[ ('line', '|S20'), ('wav_em',np.float64), ('wav_obs', np.float64)] )

np.savetxt('./out/0943_spectrum.txt', wavs_data, fmt=['%-10s']+['%.2f']+['%.2f'], \
	header='Assuming Unshifted Line Centre from Vsys\nwav_e: rest-frame wavelength\nwav_o: redshifted wavelength\n\nline     wav_e   wav_o')

# -------------------------
#   Estimated Wavelengths
# -------------------------
center = (83,108)
radius = 8

#load cube and extract region of interest **
fname 		= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_glx.fits"	
cube = mpdo.Cube(fname)

muse_0943 = cube.subcube_circle_aperture(center=center,\
	radius=radius,unit_center=None,unit_radius=None)

muse_spec = muse_0943.sum(axis=(1,2))

fig = pl.figure(figsize=(18,8))
fig.add_subplot(111)
muse_spec.plot(color='purple',lw=12)

#get axes data
ax = pl.gca()
data = ax.lines[0]

flux = data.get_ydata()
ymax = max(flux) + 5.e3 		#maximum y-limit of plot
y_label = max(flux) + 1.e3 		#y position of line labels 

# line labels and estimarted observed wavelengths
for i in range(N):
	pl.plot( [wav_obs[0],wav_obs[0]],[-250.,ymax], color='red', ls='--',lw=0.3)
	pl.text(wav_obs[0]-60.,y_label,r'Ly$\alpha$',fontsize=10,rotation='90',va='bottom',color='red')
	
	pl.plot( [wav_obs[1],wav_obs[1]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[1]-60.,y_label,'NV doublet',fontsize=10,rotation='90',va='bottom')
	
	pl.plot( [wav_obs[2],wav_obs[2]],[-250.,ymax], color='black', ls='--',lw=0.3)
	
	pl.plot( [wav_obs[3],wav_obs[3]],[-250.,ymax], color='black', ls='--',lw=0.3 )
	pl.text(wav_obs[3]-60.,y_label,'CII',fontsize=10,rotation='90',va='bottom')
	
	pl.plot( [wav_obs[4],wav_obs[4]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[4]-60.,y_label,'SiIV doublet',fontsize=10,rotation='90',va='bottom')

	pl.plot( [wav_obs[5],wav_obs[5]],[-250.,ymax], color='black', ls='--',lw=0.3 )
	
	pl.plot( [wav_obs[6],wav_obs[6]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[6]-60.,y_label, 'NIV]', rotation='90',fontsize=10,va='bottom')
	
	pl.plot( [wav_obs[7],wav_obs[7]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[7]-60.,y_label,'CIV doublet', rotation='vertical',fontsize=10,va='bottom')

	pl.plot( [wav_obs[8],wav_obs[8]],[-250.,ymax], color='black', ls='--',lw=0.3)
	
	pl.plot( [wav_obs[9],wav_obs[9]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[9]-60.,y_label, 'HeII',rotation='vertical',fontsize=10,va='bottom')
	
	pl.plot( [wav_obs[10],wav_obs[10]],[-250.,ymax], color='black', ls='--',lw=0.3)
	pl.text(wav_obs[10]-60.,y_label,'OIII] doublet', rotation='vertical',fontsize=10,va='bottom')
	
	pl.plot( [wav_obs[11],wav_obs[11]],[-250.,ymax], color='black', ls='--',lw=0.3)
	
	pl.plot( [wav_obs[12],wav_obs[12]],[-250.,ymax], color='black', ls='--',lw=0.2)
	pl.text(wav_obs[12]-60.,y_label,'CIII] doublet',rotation='90',fontsize=10,va='bottom')
	
	pl.plot( [wav_obs[13],wav_obs[13]],[-250.,ymax], color='black', ls='--',lw=0.2)
	
	pl.plot( [wav_obs[14],wav_obs[14]],[-250.,ymax], color='black',ls='--',lw=0.3)
	pl.text(wav_obs[14]-60.,y_label,'CII]',rotation='90',fontsize=10,va='bottom')

ax.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: '%d'%(x*1.e-3) ) )
ax.set_xlabel(r"Wavelength ($\AA$)")
pl.ylabel(r"Flux Density (10$^{-17}$ erg / s / cm$^2$ / $\AA$)")
pl.plot([0.,9300.],[0.,0.],color='black',ls='--')
pl.ylim([-250.,ymax])
# pl.title("0943-242 Galaxy Spectrum")
pl.savefig("./out/0943-242 spectrum.png")
pl.show()
