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
wav_rest1 = \
[ 
('Lya', 1215.7), 
('CII',1334.5),
('NIV]',1486.5),
('HeII',1640.4),
('CII]',2326.9) 
]

N1 = len(wav_rest1)

wav_rest2 = \
[
('NV',1238.8),('NV',1242.8),
('SiIV',1393.8),('SiIV',1402.8),
('CIV',1548.2),('CIV',1550.8),
('OIII]',1660.8),('OIII]',1666.1),
('CIII]',1906.7),('CIII]',1908.7),
]

N2 = len(wav_rest2)

#estimated redshift from literature **
z_est = 2.923	

#redshifted wavelengths
line1 		= [ wav_rest1[i][0] for i in range(N1) ]
line2		= [ wav_rest2[i][0] for i in range(N2) ]

wav_em1 	= [ wav_rest1[i][1] for i in range(N1) ]
wav_em2		= [ wav_rest2[i][1] for i in range(N2) ]

wav_obs1 	= [ obs_wav(z_est,wav_rest1[i][1]) for i in range(N1) ]
wav_obs2    = [ obs_wav(z_est,wav_rest2[i][1]) for i in range(N2) ] 

wavs_data1 = np.array( zip(line1,wav_em1,wav_obs1), \
	dtype=[ ('line', '|S20'), ('wav_em',np.float64), ('wav_obs', np.float64)] )

wavs_data2 = np.array( zip(line2,wav_em2,wav_obs2), \
	dtype=[ ('line', '|S20'), ('wav_em',np.float64), ('wav_obs', np.float64)] )

wavs_data = list(wavs_data1) + list(wavs_data2)

np.savetxt('./out/0943_spectrum.txt', wavs_data, fmt=['%-10s']+['%.2f']+['%.2f'], \
	header='Assuming a systemic redshift of '+`z_est`+'; \nwav_e: rest-frame (lab) wavelength\nwav_o: redshifted (observed8) wavelength\n\nline     wav_e   wav_o')

# -------------------------
#   Estimated Wavelengths8
# -------------------------
center = (83,108)
radius = 8

#load cube and extract region of interest **
fname 		= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_glx_line.fits"	
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

labels = [r'Ly$\alpha$', 'CII', 'NIV]', 'HeII', 'CII]']

#singlet lines
for j in range(N1):
	pl.plot( [wav_obs1[j],wav_obs1[j]],[-250.,ymax], color='k', ls='--',lw=0.3)
	pl.text(wav_obs1[j]-60.,y_label,labels[j],fontsize=10,rotation='90',va='bottom',color='k')

labels = ['CIII] doublet', 'SiIV doublet', 'NV doublet', 'OIII] doublet', 'CIV doublet' ]
n = len(labels)

#doublet lines
for i,j in zip(range(n),range(0,N2,2)):
 	pl.plot( [wav_obs2[j],wav_obs2[j]],[-250.,ymax], color='k', ls='--',lw=0.2)
	pl.text(wav_obs2[j]-60.,y_label,labels[i],rotation='90',fontsize=10,va='bottom')
	pl.plot( [wav_obs2[j+1],wav_obs2[j+1]],[-250.,ymax], color='k', ls='--',lw=0.2)
	
## plot spectrum
wav = data.get_xdata()
min_wav = min(wav)
max_wav = max(wav)

# observed wavelength
pl.xticks(np.arange(min_wav,max_wav+100.,400.))
ax.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: '%d'%(x*1.e-3) ) )
ax.set_xlabel(r"Observed Wavelength ($\AA$)")
pl.ylabel(r"Flux Density (10$^{-17}$ erg / s / cm$^2$ / $\AA$)")
pl.plot([0.,9300.],[0.,0.],color='black',ls='--')
pl.ylim([-250.,ymax])
# pl.title("MRC0943-242: R=8pix; ra,dec=(108,83)pix; MRC0943_glx_line.fits")
pl.savefig("./out/0943-242 spectrum observed.png")
pl.savefig("./out/0943-242 spectrum observed.pdf")

# rest wavelength
arr = np.arange(1200.,2500.,200.)
xticks = [ obs_wav(z_est,arr[i]) for i in range(len(arr)) ]
pl.xticks(xticks)
ax.xaxis.set_major_formatter( tk.FuncFormatter( lambda x,pos: '%d'%(x/3.923) ) )
pl.xlabel("Rest Wavelength ($\AA$)")
pl.savefig("./out/0943-242 spectrum rest.png")
pl.savefig("./out/0943-242 spectrum rest.pdf")



