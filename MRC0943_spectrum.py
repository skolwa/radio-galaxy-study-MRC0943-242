#S.N. Kolwa (2017)
#MRC0943_spectrum.py
#Purpose:  
#	- Plot the full MUSE spectrum of 0943-242's AGN host galaxy
#	- Name the line identifiers
#	- Line wavelengths are approximated from rest wavs and known redshift:
#	0943-242 redshift = 2.923
# Note: ** denotes sections of code that are individual to a source

import matplotlib.pyplot as pl
import numpy as np

import spectral_cube as sc
import mpdaf.obj as mpdo

import matplotlib.ticker as tk

import warnings
from astropy.utils.exceptions import AstropyWarning

from functions import*

# params = {'legend.fontsize': 18,
#           'legend.handlelength': 2}

# pl.rcParams.update(params)

# pl.rc('text', usetex=True)
pl.rc('font', **{'family':'sans-serif', 'sans-serif':['Computer Modern Sans serif']})

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

# -----------------------------------
#   Estimate Observed Wavelengths
# -----------------------------------
def obs_wav(z,lam_rest):
	lam_obs = lam_rest*(1.+z)
	return lam_obs

def rest_wav(z,lam_obs):
	lam_rest = lam_obs/(1.+z)
	return lam_rest

#rest wavelengths of identified lines **
wav_rest1 = \
[ 
('Lya', 1215.57), 
('CII', 1334.5),
('NIV]', 1486.5),
('HeII', 1640.4),
('CII]', 2326.9) 
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

np.savetxt('./out/spectra/0943_spectrum.txt', wavs_data, fmt=['%-10s']+['%.2f']+['%.2f'], \
	header='Assuming a systemic redshift of '+`z_est`+'; \nwav_e: rest-frame (lab) wavelength\nwav_o: redshifted (observed8) wavelength\n\nline     wav_e   wav_o')

# -------------------------
#   Estimated Wavelengths8
# -------------------------
radius = 3
center = (88,66)

#load cube and extract region of interest **
fname 		= "/Users/skolwa/DATA/MUSE_data/0943-242/MRC0943_glx_line.fits"	
cube 		= mpdo.Cube(fname)

#shift target pixel to center
center2 = (center[0]-1,center[1]-1)

cube = cube.subcube(center2,(2*radius+1),unit_center=None,unit_size=None)

#obtain spatially integrated spectrum
muse_spec = cube.sum(axis=(1,2))

wav 	=  muse_spec.wave.coord()  	#1.e-8 cm
flux 	=  muse_spec.data			#1.e-20 erg / s / cm^2 / Ang

fig = pl.figure(figsize=(18,8))
fig.add_subplot(111)
muse_spec.plot(color='purple',lw=12)

#get axes data
ax1 = pl.gca()
data = ax1.lines[0]

ymax = 1.5*max(flux)		#maximum y-limit of plot
y_label = 1.2*max(flux) 		#y position of line labels 

labels = [r'Ly$\alpha$', 'CII', 'NIV]', 'HeII', 'CII]']

#singlet lines
for j in range(N1):
	pl.plot( [wav_obs1[j],wav_obs1[j]],[-250.,ymax], color='k', ls='--',lw=0.6)
	pl.text(wav_obs1[j]-60.,y_label,labels[j],fontsize=12,rotation='90',va='bottom',color='k')


labels = ['NV doublet', 'SiIV doublet', 'CIV doublet', 'OIII] doublet', 'CIII] doublet'  ]
n = len(labels)

#doublet lines
for i,j in zip(range(n),range(0,N2,2)):
 	pl.plot( [wav_obs2[j],wav_obs2[j]],[-250.,ymax], color='k', ls='--',lw=0.4)
	pl.text(wav_obs2[j]-60.,y_label,labels[i],rotation='90',fontsize=12,va='bottom')
	pl.plot( [wav_obs2[j+1],wav_obs2[j+1]],[-250.,ymax], color='k', ls='--',lw=0.4)
	

min_wav = min(wav)
max_wav = max(wav)
pl.plot([0., 9400.],[0., 0.],color='black',ls='--')

arr = np.arange(1200., 2500., 200.)
xticks = [ obs_wav(z_est,arr[i]) for i in range(len(arr)) ]
pl.xticks(xticks)

ax1.set_xlabel("Observed Wavelength ($\AA$)", fontsize=15)
# ax2 = ax1.twiny()
# ax1.xaxis.set_major_formatter( tk.FuncFormatter( lambda x,pos: '%d'%(x/(1.+2.923)) ) )

ax1.set_xticks( np.arange(min_wav,max_wav+50.,400.) )
ax1.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: (x*1.e-3)) )
ax1.set_ylabel(r"Flux Density (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=15)
ax1.tick_params(direction='in', top=1)

ax2 = ax1.twiny()
ax2.set_xlabel(r"Rest Wavelength ($\AA$)", fontsize=15)
ax2.tick_params(direction='in', right=1, top=1)

rest_ticks = np.arange(1200.,2800.,200)

new_tick_locations = obs_wav(2.923, rest_ticks)

def tick_function(X):
    V = X/3.923
    return ["%.f" % z for z in V]

for tick in ax1.xaxis.get_major_ticks():
	tick.label.set_fontsize(14)

for tick in ax1.yaxis.get_major_ticks():
	tick.label.set_fontsize(14)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations), fontdict={'fontsize': '14'})

pl.ylim([-250.,ymax])
pl.savefig("./out/spectra/0943-242_spectrum_"+`radius`+"_pix.png")
pl.savefig("/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/0943-242_spectrum_"+`radius`+"_pix.pdf")
# pl.show()