# S.N. Kolwa (2018)
# MRC0943_line_fit_0.2.py

# Purpose:  
# - Non-resonant line profile fit i.e. Gaussian fits
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)
# - Flux axis in micro-Jy
# - Double Gaussian in HeII to account for blueshifted component

import matplotlib.pyplot as pl
import numpy as np 

import spectral_cube as sc
import mpdaf.obj as mpdo
import astropy.units as u
import matplotlib.ticker as tk
import lmfit.models as lm
from lmfit import Parameters

import warnings
from astropy.utils.exceptions import AstropyWarning

from functions import *

import sys
import os

#make code automated such that no fine tuning in initial guesses needed at all
#see guess function source code in lmfit

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat   = [ 'NIV]', 'HeII', 'OIII]','CIII]','CII]' ]

#pixel radius for extraction
radius = 3
center = (88,67)

# -----------------------------------------------
# 	   FIT to HeII for systemic kinematics
# -----------------------------------------------

fname 			= "./out/HeII.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

#shift centre
center1 = (center[0]-1,center[1]-1)

sub = datacube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)
var = varcube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)

cubes = [sub,var]
for cube in cubes:
	subcube_mask(cube,radius)

# pl.figure()
# var.sum(axis=0).plot()
# pl.show()

spec_HeII = sub.sum(axis=(1,2))
var_spec = var.sum(axis=(1,2))

wav 	=  spec_HeII.wave.coord()  		#1.e-8 cm
flux 	=  spec_HeII.data				#1.e-20 erg / s / cm^2 / Ang

wav_e_HeII 	= 1640.4	
z_est 		= 2.923 					# of source from literature

g_cen 		= wav_e_HeII*(1.+z_est)
g_blue		= 6415.

pars = Parameters()

#initial guesses for HeII (model for all other emisison lines)
amp = 1.e5
wid = 10.
cont = 100.

pars.add_many( \
	('g_cen1',g_blue, True,g_blue-1.,g_blue+1.),\
	('amp1', 0.1*amp, True, 0. ),\
	('wid1', wid, True, 0.),
	('g_cen2',g_cen, True,g_cen-1.,g_cen+1.),\
	('amp2', amp, True, 0. ),\
	('wid2', wid, True, 0.),\
	('cont', cont, True, 0.))	

var_flux    = var_spec.data
n 			= len(var_flux)
inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

mod 	= lm.Model(dgauss) 

fit 	= mod.fit(flux,pars,x=wav)

print fit.fit_report()

res 	= fit.params

g_blue 			= fit.params['g_cen1'].value
amp_blue		= fit.params['amp1'].value
wid_blue 		= fit.params['wid1'].value

wav_o_HeII 		= fit.params['g_cen2'].value
wav_o_err_HeII 	= fit.params['g_cen2'].stderr
amp_HeII 		= fit.params['amp2'].value
amp_HeII_err	= fit.params['amp2'].stderr
wid_HeII 		= fit.params['wid2'].value
wid_HeII_err	= fit.params['wid2'].stderr
cont_HeII 		= fit.params['cont'].value

delg 			= wav_o_err_HeII 		#error on gaussian centre for all lines

for spec_feat in spec_feat:
	print spec_feat
	#--------------------------
	#  Spectral Line Fitting
	#--------------------------	
	fname 			= "./out/"+spec_feat+".fits"
	datacube 		= mpdo.Cube(fname,ext=1)
	varcube  		= mpdo.Cube(fname,ext=2)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRUM for LINE FIT
	# -----------------------------------------------
	#shift centre
	center2 = (center[0]-1,center[1]-1)

	glx = datacube.subcube(center2,(2*radius+1),unit_center=None,unit_size=None)
	var_glx = varcube.subcube(center2,(2*radius+1),unit_center=None,unit_size=None)

	cubes = [glx,var_glx]
	for cube in cubes:
		subcube_mask(cube,radius)

	fig = pl.figure(figsize=(8,8))
	ax = pl.gca()

	spec   = glx.sum(axis=(1,2))
	var_spec = var_glx.sum(axis=(1,2))
	
	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v
	
	c = 2.9979245800e5 	#km/s
	
	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = obs frame (z=0)
	z = vel0_rest/c
	
	# -----------------------------------
	# 	  MODEL FIT to EMISSION LINE
	# -----------------------------------

	wav 	=  spec.wave.coord()  		#Ang
	flux 	=  spec.data				#1.e-20 erg / s / cm^2 / Ang

	# estimate central Gaussian wavelengths
	if spec_feat in 'HeII':			
		lam   = 1640.4
		g_cen = lam*(1.+z)

	elif spec_feat == 'CII]':
		lam   = 2326.9
		g_cen = lam*(1.+z)

	elif spec_feat == 'NIV]':
		lam   = 1486.5
		g_cen = lam*(1.+z)

	elif spec_feat == 'OIII]':
		lam1  = 1660.8
		lam2  = 1666.1
		g_cen1 = lam1*(1.+z)
		g_cen2 = lam2*(1.+z)

	elif spec_feat == 'CIII]':
		lam1  = 1906.7
		lam2  = 1908.7
		g_cen1 = lam1*(1.+z)
	  	g_cen2 = lam2*(1.+z)

	# SINGLET states
	if spec_feat == 'HeII':
		mod 	= lm.Model(dgauss) 

		pars 	= Parameters()

		pars.add_many( ('g_cen1', g_blue, False),
			('amp1', amp_blue, False),('wid1', wid_blue, False ),
			('amp2', amp_HeII, False),('wid2', wid_HeII, False ),
			('g_cen2', wav_o_HeII, False),
			('cont', cont_HeII, False) )	

		#flux variance
		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		fwhm		= 2*np.sqrt(2*np.log(2))*wid_HeII
		fwhm_err	= (wid_HeII_err/wid_HeII)*fwhm

		fwhm_kms	= (fwhm/wav_o_HeII)*c
		fwhm_kms_err = fwhm_kms*np.sqrt( (fwhm_err/fwhm)**2 + (wav_o_err_HeII/wav_o_HeII)**2 )

		pl.plot(wav, gauss(wav, amp_blue, wid_blue, g_blue, cont_HeII), ls='--', c='blue', label='blueshifted component')
		pl.plot(wav, gauss(wav, amp_HeII, wid_HeII, wav_o_HeII, cont_HeII ), ls='--', c='orange', label=r'HeII $\lambda$'+`lam`+r'$\AA$')
		pl.plot(wav,fit.best_fit,'r',label='best-fit model')
		pl.legend()

	if spec_feat in ('CII]','NIV]'):
		mod 	= lm.Model(gauss) 

		pars 	= Parameters()

		#HeII gaussian width (km/s)
		v = c * ( wid_HeII / wav_o_HeII ) 

		#initial guess Gaussian width for CIV lines
		guess_wid = g_cen * ( v / c )

		pars.add_many( ('g_cen', g_cen, True, g_cen-delg, g_cen+delg),
			('amp', amp_HeII, True, 10.),('wid', guess_wid, True, 0. ),
			('cont', cont_HeII, True, 0.) )	

		#flux variance
		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		print fit.fit_report()

		amp 		= fit.params['amp'].value
		amp_err		= fit.params['amp'].stderr
		wav_o 		= fit.params['g_cen'].value
		wav_o_err 	= fit.params['g_cen'].stderr
		wid 		= fit.params['wid'].value
		wid_err 	= fit.params['wid'].stderr	

		fwhm		= 2*np.sqrt(2*np.log(2))*wid
		fwhm_err	= (wid_err/wid)*fwhm

		fwhm_kms	= (fwhm/wav_o)*c
		fwhm_kms_err = fwhm_kms*np.sqrt( (fwhm_err/fwhm)**2 + (wav_o_err/wav_o)**2 )

		pl.plot(wav,fit.best_fit,'r',label='best-fit model')
		pl.legend()
	
	# DOUBLET states
	elif spec_feat == 'OIII]':
		mod 	= lm.Model(dgauss)

		pars 	= Parameters()
		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.0031912331406552*g_cen1') 
	
	elif spec_feat == 'CIII]':
		mod 	= lm.Model(dgauss)

		pars 	= Parameters()
		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.0010489327109666*g_cen1') 

	if spec_feat == 'OIII]' or spec_feat == 'CIII]':
		pars.add_many(  
			('amp1', amp_HeII, True, 0.,),
			 ('amp2', amp_HeII, True, 0.,), 
			 ('cont', cont_HeII, True, 0.) )

		#HeII gaussian width (km/s)
		v = c * ( wid_HeII / wav_o_HeII ) 

		#initial guess Gaussian width for CIV lines
		guess_wid = g_cen1 * ( v / c )

		pars.add('wid1', guess_wid, True, 0.)
		pars.add('wid2', expr='wid1')
		
		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)	

		print fit.fit_report()
		
		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)

		fwhm1		= 2*np.sqrt(2*np.log(2))*wid[0]
		fwhm1_err	= (wid_err[0]/wid[0])*fwhm

		fwhm2 		= 2*np.sqrt(2*np.log(2))*wid[1]
		fwhm2_err 	= (wid_err[1]/wid[1])*fwhm

		fwhm1_kms	= (fwhm1/wav_o[0])*c
		fwhm1_kms_err = fwhm1_kms*np.sqrt( (fwhm1_err/fwhm1)**2 + (wav_o_err[0]/wav_o[0])**2  )

		fwhm2_kms	= (fwhm2/wav_o[1])*c
		fwhm2_kms_err = fwhm2_kms*np.sqrt( (fwhm2_err/fwhm2)**2 + (wav_o_err[1]/wav_o[1])**2  )

		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value
	
		pl.plot(wav,fit.best_fit,'r',label='best-fit model')
		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont),color='orange',linestyle='--',label=spec_feat+r' $\lambda$'+`lam1`+r'$\AA$')
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont),color='blue',linestyle='--',label=spec_feat+r' $\lambda$'+`lam2`+r'$\AA$')

	chisqr = r'$\chi^2$: %1.2f' %fit.chisqr
	redchisqr = r'$\widetilde{\chi}^2$: %1.2f' %fit.redchi
	pl.text(0.15,0.9, redchisqr,ha='center', va='center',transform = ax.transAxes, fontsize=16)

	if spec_feat == 'CII]':
		pl.plot(wav,flux,drawstyle='steps-mid',color='k', label=spec_feat+r' $\lambda$'+`lam`+r'$\AA$')	#plot data
	else:
		pl.plot(wav,flux,drawstyle='steps-mid',color='k')	#plot data

	pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')	#fill grey between zero on flux axis and data
	pl.legend()


	# -----------------------------------------------
	# 	  Write fit parameters into text file
	# -----------------------------------------------
	
	filename = "./out/line-fitting/emission_line_fit/0943 fit.txt"

	if spec_feat 	== 'NIV]':
		wav_e = lam
		header = np.array( \
			["#wav0(Ang)          wav0_err       flux_peak(erg/s/cm^2/Ang)     flux_peak_err	sigma(Ang)    	   sigma_err   	  fwhm(km/s)   		fwhm_err      wav_rest(Ang) spec_feat"] )

		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 wid, wid_err, fwhm_kms, fwhm_kms_err, wav_e, spec_feat])
		with open(filename,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')	
	
	elif spec_feat 	== 'HeII':
		wav_e = lam
		gauss_fit_params = np.array([wav_o_HeII, wav_o_err_HeII, amp_HeII, amp_HeII_err,\
		 wid_HeII, wid_HeII_err, fwhm_kms, fwhm_kms_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'CII]':
		wav_e = lam
		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 wid, wid_err, fwhm_kms, fwhm_kms_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'OIII]':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array([wav_o[0], wav_o_err[0],\
			amp[0], amp_err[0], wid[0],\
			wid_err[0], fwhm1_kms, fwhm1_kms_err, wav_e1, spec_feat])

		gauss_fit_params2 = np.array([wav_o[1], wav_o_err[1],\
			amp[1], amp_err[1], wid[1],\
			wid_err[1], fwhm2_kms, fwhm2_kms_err, wav_e2, spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	elif spec_feat 	== 'CIII]':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array([wav_o[0], wav_o_err[0],\
			amp[0], amp_err[0], wid[0],\
			wid_err[0], fwhm1_kms, fwhm1_kms_err, wav_e1, spec_feat])

		gauss_fit_params2 = np.array([wav_o[1], wav_o_err[1],\
			amp[1], amp_err[1], wid[1],\
			wid_err[1], fwhm2_kms, fwhm2_kms_err, wav_e2, spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	# -----------------------------------------------
	#    	PLOT with velocity and uJy axes
	# -----------------------------------------------
	#LSR velocity
	vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	#offset velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat in ('HeII','NIV]','CII]'):	
		vel_meas = vel(wav_o,wav_e,z)		#central velocity of detected line
		vel_off = vel_meas - vel0			#offset velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z) - vel0 ) )
	
	#doublet state
	else:
		vel_meas = [vel(wav_o[0],wav_e1,z),vel(wav_o[1],wav_e2,z)]	
		vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 

		ax_up = ax.twiny()
		xticks_up = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e1,z) - vel0) ) 
		ax_up.xaxis.set_major_formatter(xticks_up)
	
	# define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)
	
	#central wavelength depending on singlet or doublet fit
	if spec_feat in ('HeII','NIV]','SiIV','CII]'):
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy
	else:
		wav_cent = wav_o[1]
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy	
	
	#need a better way of creating these tick!
	if maxf < 1.:
		flux0 = flux_cgs(wav_cent,0.)
		flux05 = flux_cgs(wav_cent,0.5e-6)
		flux1 = flux_cgs(wav_cent,1.e-6)
		major_yticks = [ flux0, flux05, flux1 ]
	
	elif maxf > 1. and maxf < 4.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		flux2 = flux_cgs(wav_cent,2.e-6)
		flux3 = flux_cgs(wav_cent,3.e-6)
		flux4 = flux_cgs(wav_cent,4.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	elif maxf > 4. and maxf < 5.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		flux2 = flux_cgs(wav_cent,2.e-6)
		flux3 = flux_cgs(wav_cent,3.e-6)
		flux4 = flux_cgs(wav_cent,4.e-6)
		flux5 = flux_cgs(wav_cent,5.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]
	
	elif maxf > 5. and maxf < 15.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,5.e-6)
		flux2 = flux_cgs(wav_cent,10.e-6)
		flux3 = flux_cgs(wav_cent,10.e-6)
		flux4 = flux_cgs(wav_cent,15.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	elif maxf > 15. and maxf < 30.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,10.e-6)
		flux2 = flux_cgs(wav_cent,20.e-6)
		flux3 = flux_cgs(wav_cent,30.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]
	
	elif maxf > 30.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,10.e-6)
		flux2 = flux_cgs(wav_cent,20.e-6)
		flux3 = flux_cgs(wav_cent,30.e-6)
		flux4 = flux_cgs(wav_cent,40.e-6)	
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	ax.set_yticks(major_yticks)
	
	#define y-axis as flux in Jy
	if maxf < 1.:
		yticks = tk.FuncFormatter( lambda x,pos: '%.1f'%( flux_Jy(wav_cent,x)*1.e6 ) 	)
		ax.yaxis.set_major_formatter(yticks)
	
	else:
		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6	 ) )
		ax.yaxis.set_major_formatter(yticks)
	
	#get wavelength corresponding to an offset velocity
	def offset_vel_to_wav(voff):
		v1 = -voff
		v2 = voff
		if spec_feat in ('HeII','NIV]','SiIV','CII]'):
			wav1 = wav_e*(1.+z)*(1.+(v1/c))
			wav2 = wav_e*(1.+z)*(1.+(v2/c))
			return wav1,wav2
		else:
			wav1 = wav_e2*(1.+z)*(1.+(v1/c))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c))
			return wav1,wav2

	def offset_vel_to_wav_up(voff):
		v1 = -voff
		v2 = voff
		wav1 = wav_e1*(1.+z)*(1.+(v1/c))
		wav2 = wav_e1*(1.+z)*(1.+(v2/c))
		return wav1,wav2
	
	#define x-axis as offset velocity from Vsys/zero-point (km/s)
	if spec_feat == 'CIII]' or spec_feat == 'OIII]':
		wav0 	= offset_vel_to_wav_up(+0.)
		wavlim1 = offset_vel_to_wav_up(1000.)
		wavlim2 = offset_vel_to_wav_up(2000.)
		wavlim3 = offset_vel_to_wav_up(3000.)
		wavlim4 = offset_vel_to_wav_up(4000.)
		wavlim5 = offset_vel_to_wav_up(5000.)
		wavlim6 = offset_vel_to_wav_up(6000.)
	
		major_ticks_up = [ wavlim4[0], wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
		wavlim1[1],  wavlim2[1], wavlim3[1], wavlim4[1], wavlim5[1], wavlim6[1] ]
	
		ax_up.set_xticks(major_ticks_up)

		for label in ax_up.xaxis.get_majorticklabels():
			label.set_fontsize(16)

	wav0 	= offset_vel_to_wav(+0.)
	wavlim1 = offset_vel_to_wav(1000.)
	wavlim2 = offset_vel_to_wav(2000.)
	wavlim3 = offset_vel_to_wav(3000.)
	wavlim4 = offset_vel_to_wav(4000.)

	major_ticks = [ wavlim4[0], wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1], wavlim3[1], wavlim4[1] ]

	ax.set_xticks(major_ticks)

	xmin = wavlim3[0] 
	xmax = wavlim3[1] 
	
	#define y-limits
	ymax = 1.2*max(flux)
	ymin = -0.01*max(flux)

	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(16)

	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(16)
	

	pl.plot([xmin,xmax],[0.,0.],ls='-',color='grey')	#zero flux density-axis
	ax.set_xlabel( r'Velocity (km/s)', fontsize=16)
	ax.set_ylabel( r'Flux Density ($\mu$Jy)', fontsize=16 )
	ax.set_ylim([ymin,ymax])
	ax.set_xlim([xmin,xmax])
	pl.savefig('./out/line-fitting/emission_line_fit/'+spec_feat+'_2_fit.png')
	pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_2_fit.pdf')


#align columns in text file for better readability
res = "./out/line-fitting/emission_line_fit/0943 fit.txt"
fit = open("./out/line-fitting/emission_line_fit/0943_fit_2.txt", "w")

with open(res, 'r') as f:
    for line in f:
        data = line.split()    # Splits on whitespace
        x = '{0[0]:<20}{0[1]:<20}{0[2]:<35}{0[3]:<20}{0[4]:<20}{0[5]:<20}{0[6]:<20}{0[7]:<20}{0[8]:<10}{0[9]:<5}'.format(data)
        fit.write(x[:]+'\n')	

#delete original file
os.system('rm ./out/line-fitting/emission_line_fit/0943\ fit.txt')