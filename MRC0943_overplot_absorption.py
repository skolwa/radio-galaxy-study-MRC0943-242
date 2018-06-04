#S.N. Kolwa (2018)
#MRC0943_overplot_absorption.py

#Purpose: 
# -overplot and stack resonant line profiles: Lya, CIV, NV and SiIV
# -determine the relative velocity shifts between absorption features in each 


import matplotlib.pyplot as pl
import numpy as np 

import mpdaf.obj as mpdo
from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning

from functions import *

# from line_fit_0943 import subcube_mask

spec_feat_arr = [ 'HeII', 'Lya', 'CIV', 'NV', 'SiIV', 'CII']
wav_rest  = [ (1640.4,1640.4), (1215.57,1215.57), (1548.202, 1550.774), (1238.8, 1242.8), (1393.76, 1402.77), (1334.5,1334.5) ]

n = len(spec_feat_arr)
plot_colors = [ 'purple', 'orange', 'red', 'green', 'blue', 'magenta' ]

#pixel radius for extraction
radius = 3
center = (88,66)

wav_arr 	= [ [] for i in range(n) ]
flux_arr 	= [ [] for i in range(n) ]
vel_arr1 	= [ [] for i in range(n) ]
vel_arr2 	= [ [] for i in range(n) ]

z = 2.923 		#systemic redshift from literature
i = -1

for spec_feat in spec_feat_arr:

	fname 		= "./out/"+spec_feat+".fits"
	datacube 	= mpdo.Cube(fname,ext=1)

	#shift centre
	center1 = (center[0]-1,center[1]-1)

	sub = datacube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)

	subcube_mask(sub,radius)

	spec 	= sub.sum(axis=(1,2))

	# pl.figure()
	# img = arp.sum(axis=0)
	# img.plot(scale='linear')
	# pl.show()

	wav 	=  spec.wave.coord()  		#Ang
 	flux 	=  spec.data				#1.e-20 erg / s / cm^2 / Ang

 	# pl.plot(wav,flux,c='k',drawstyle='steps-mid')
 	# pl.show()

 	N = len(wav)
	i += 1

	normed_flux = [ flux[k]/max(flux) for k in xrange(N)  ]

	for j in range(N):
		wav_arr[i].append( wav[j] )
 		flux_arr[i].append( normed_flux[j] )

	c = 2.9979245800e5					#km/s

	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v

	vel_arr1[i] = [ vel(wav_arr[i][j],wav_rest[i][0],z) for j in xrange(N) ]  #LSR fine structure wav1
	vel_arr2[i] = [ vel(wav_arr[i][j],wav_rest[i][1],z) for j in xrange(N) ]  #LSR fine structure wav2

	# Lya absorber velocities from Gullberg et al (2016), Table 2
	lam = 1215.57

	wav_abs1 = lam*(1.+2.90689) 
	errs_abs1 = 0.00032
	ci_lim1 = (wav_abs1-errs_abs1,wav_abs1+errs_abs1) #conf interval limit

	wav_abs2 = lam*(1.+2.91864)
	errs_abs2 = 0.00002
	ci_lim2 = (wav_abs2-errs_abs2,wav_abs2+errs_abs2) #conf interval limit

	wav_abs3 = lam*(1.+2.92641)
	errs_abs3 = 0.00023
	ci_lim3 = (wav_abs3-errs_abs3,wav_abs3+errs_abs3) #conf interval limit

	wav_abs4 = lam*(1.+2.93254)
	errs_abs4 = 0.00007
	ci_lim4 = (wav_abs4-errs_abs4,wav_abs4+errs_abs4) #conf interval limit

	vel_abs1 = vel(wav_abs1,lam,z)
	vel_ci_lim1 = (vel(ci_lim1[0],lam,z),vel(ci_lim1[1],lam,z))

	vel_abs2 = vel(wav_abs2,lam,z)
	vel_ci_lim2 = (vel(ci_lim2[0],lam,z),vel(ci_lim2[1],lam,z))

	vel_abs3 = vel(wav_abs3,lam,z)
	vel_ci_lim3 = (vel(ci_lim3[0],lam,z),vel(ci_lim3[1],lam,z))

	vel_abs4 = vel(wav_abs4,lam,z)
	vel_ci_lim4 = (vel(ci_lim4[0],lam,z),vel(ci_lim4[1],lam,z))

for i,j in zip( xrange(2,len(flux_arr),1), xrange(2,len(plot_colors),1)):
	fig = pl.figure()
	ax = pl.gca()
	pl.plot([vel_ci_lim1[0],vel_ci_lim1[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim1[0],1.1,'1')
	pl.plot([vel_ci_lim2[0],vel_ci_lim2[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim2[0],1.1,'2')
	pl.plot([vel_ci_lim3[0],vel_ci_lim3[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim3[0],1.1,'3')
	pl.plot([vel_ci_lim4[0],vel_ci_lim4[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim4[0],1.1,'4')
	pl.ylim([-0.1,1.2])

	pl.plot(vel_arr1[0],flux_arr[0],c=plot_colors[0], drawstyle='steps-mid', lw=0.8, ls='--', label=r'HeII')
	pl.plot(vel_arr1[1],flux_arr[1],c=plot_colors[1], drawstyle='steps-mid', lw=0.8, ls='-.', label=r'Ly$\alpha$') #Lya 
	# pl.plot(vel_arr1[2],flux_arr[2],c='red', drawstyle='steps-mid', lw=0.8, ls='-', label='CIV')
	# pl.plot(vel_arr1[3],flux_arr[3],c='green', drawstyle='steps-mid', lw=0.8, ls='-', label='NV')
	pl.plot(vel_arr1[i],flux_arr[i],c=plot_colors[j], drawstyle='steps-mid', lw=1.2, label=spec_feat_arr[j])
	pl.xlim([-3000.,3000.])
	pl.ylim([-0.1,1.2])
	pl.legend()
	pl.xlabel('Velocity Offset (km/s)')
	pl.ylabel(r'Normalised Flux')
	# pl.savefig('./out/CIV_NV_abs_comparison_fsline1.png')
	pl.savefig('./out/'+spec_feat_arr[j]+'_abs_comparison_fsline1.png')
	# pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat_arr[j]+'_abs_comparison_fsline1.pdf')
	# pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/Lya_'+spec_feat_arr[j]+'1_CII.pdf')
	
for i,j in zip( xrange(2,len(flux_arr),1), xrange(2,len(plot_colors),1)):
	fig = pl.figure()
	ax = pl.gca()
	pl.plot([vel_ci_lim1[0],vel_ci_lim1[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim1[0],1.1,'1')
	pl.plot([vel_ci_lim2[0],vel_ci_lim2[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim2[0],1.1,'2')
	pl.plot([vel_ci_lim3[0],vel_ci_lim3[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim3[0],1.1,'3')
	pl.plot([vel_ci_lim4[0],vel_ci_lim4[1]],[0.9,1.1],c='k')
	pl.text(vel_ci_lim4[0],1.1,'4')
	pl.ylim([-0.1,1.2])

	pl.plot(vel_arr2[0],flux_arr[0],c=plot_colors[0], drawstyle='steps-mid', lw=0.8, ls='--', label=r'HeII')
	pl.plot(vel_arr2[1],flux_arr[1],c=plot_colors[1], drawstyle='steps-mid', lw=0.8, ls='-.', label=r'Ly$\alpha$') #Lya 
	# pl.plot(vel_arr2[5],flux_arr[5],c=plot_colors[5], drawstyle='steps-mid', lw=0.8, ls='-.', label='CII')
	# pl.plot(vel_arr2[2],flux_arr[2],c='red', drawstyle='steps-mid', lw=0.8, ls='-', label='CIV')
	# pl.plot(vel_arr2[3],flux_arr[3],c='green', drawstyle='steps-mid', lw=0.8, ls='-', label='NV')
	pl.plot(vel_arr2[i],flux_arr[i],c=plot_colors[j], drawstyle='steps-mid', lw=1.2, label=spec_feat_arr[j])
	pl.xlim([-3000.,3000.])
	pl.ylim([-0.1,1.2])
	pl.legend()
	pl.xlabel('Velocity Offset (km/s)')
	pl.ylabel(r'Normalised Flux')
	# pl.savefig('./out/CIV_NV_abs_comparison_fsline2.png')
	pl.savefig('./out/'+spec_feat_arr[j]+'_abs_comparison_fsline2.png')
	# pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat_arr[j]+'_abs_comparison_fsline2.pdf')
	# pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/Lya_'+spec_feat_arr[j]+'2_CII.pdf')
# pl.show()