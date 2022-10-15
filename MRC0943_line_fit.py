# S.N. Kolwa
# ESO (2018)
# MRC0943_line_fit.py

# Purpose:  
# - Non-resonant line profile fit i.e. Gaussian fits
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)
# - Flux axis in micro-Jy
# - Double Gaussian in HeII to account for blueshifted component
# - Taken into account blueshifted component detected with highest S/N in HeII

import matplotlib.pyplot as pl
import numpy as np 

import spectral_cube as sc
import mpdaf.obj as mpdo
import astropy.units as u
import matplotlib.ticker as tk
import lmfit.models as lm
from lmfit import *

import warnings
from astropy.utils.exceptions import AstropyWarning

from functions import *

import sys
import os

params = {'legend.fontsize': 14,
          'legend.handlelength': 2}

pl.rcParams.update(params)

pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})
pl.rc('text', usetex=True)

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

home = sys.argv[1]

spec_feat   = [ 'HeII', 'NIV]', 'OIII]', 'CIII]', 'CII]']

#pixel radius for extraction
radius = 3
center = (88,66)

# -----------------------------------------------
# 	   FIT to HeII for systemic kinematics
# -----------------------------------------------

fname 			= "./out/HeII.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

#shift centre
center1 = (center[0]-1,center[1]-1)

#import subcube
cube_HeII = datacube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)
var_HeII = varcube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)

#mask cube to extract aperture of interest
cubes = [cube_HeII,var_HeII]
for cube in cubes:
	subcube_mask(cube,radius)

spec_HeII 		= cube_HeII.sum(axis=(1,2))
spec_HeII_copy 	= cube_HeII.sum(axis=(1,2))
var_spec_HeII 	= var_HeII.sum(axis=(1,2))

var_flux    = var_spec_HeII.data

# 1/sigma**2  => inverse noise
n 			= len(var_flux)
inv_noise 	= [ var_flux[i]**-1 for i in range(n) ] 

#mask non-continuum lines 
spec_HeII_copy.mask_region(6298.,6302.)	#skyline (?)
spec_HeII_copy.mask_region(6366.,6570.)	#HeII and OIII] doublet lines

wav_mask 	= spec_HeII_copy.wave.coord()
flux_mask 	= spec_HeII_copy.data

pars = Parameters()

pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))

mini = Minimizer(str_line_model, pars, nan_policy='omit')
out = mini.minimize(method='leastsq')

# report_fit(out)

grad = out.params['grad'].value
cut = out.params['cut'].value
cont = str_line(wav_mask,grad,cut)
cont_HeII = np.mean(cont)

# continuum subtract from copy of spectrum
spec_HeII_sub = spec_HeII - cont

#continuum subtracted data
wav 	=  spec_HeII_sub.wave.coord()    #1.e-8 cm
flux 	=  spec_HeII_sub.data            #1.e-20 erg / s / cm^2 / Ang

wav_e_HeII 	= 1640.4
z 			= 2.923 					
g_cen 		= wav_e_HeII*(1.+z)
g_blue 		= 6414.						

pars = Parameters()

pars.add_many( \
	('g_cen1', g_blue, True, g_blue-10., g_blue+25.),\
	('amp1', 2.e3, True, 0.),\
	('wid1', 12., True, 0., 50.),\
	('g_cen2', g_cen, True, g_cen-1., g_cen+3.),\
	('amp2', 2.e4, True, 0.),\
	('wid2', 10., True, 0.))	

mod 	= lm.Model(dgauss_nocont) 

fit 	= mod.fit(flux, pars, x=wav, weights=inv_noise)

print fit.fit_report()

res 	= fit.params

g_blue         = fit.params['g_cen1'].value
amp_blue       = fit.params['amp1'].value
wid_blue       = fit.params['wid1'].value

wav_o_HeII     = fit.params['g_cen2'].value
wav_o_HeII_err = fit.params['g_cen2'].stderr
amp_HeII       = fit.params['amp2'].value
amp_HeII_err   = fit.params['amp2'].stderr
wid_HeII       = fit.params['wid2'].value
wid_HeII_err   = fit.params['wid2'].stderr

delg           = wav_o_HeII_err 		#error on gaussian centre for all lines

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

	fig = pl.figure(figsize=(7,8))
	ax = pl.gca()

	spec   = glx.sum(axis=(1,2))
	var_spec = var_glx.sum(axis=(1,2))
	
	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v
	
	c = 2.9979245800e5 	#km/s
	
	vel0 = vel(wav_o_HeII, wav_e_HeII, 0.) 	#source frame = obs frame (z=0)
	z = vel0 / c
	z_err 	= ( wav_o_HeII_err / wav_o_HeII )*z
	
	# -----------------------------------
	# 	  MODEL FIT to EMISSION LINE
	# -----------------------------------

	wav 	=  spec.wave.coord()    #Ang
	flux 	=  spec.data            #1.e-20 erg / s / cm^2 / Ang

	# estimate central Gaussian wavelengths
	if spec_feat in 'HeII':			
		lam   = 1640.4
		g_cen = lam*(1.+z)

	elif spec_feat == 'CII]':
		lam   = 2326.9
		g_cen = lam*(1.+z)
		z_blue = 2.909
		g_cen_blue = lam*(1+z_blue) 		

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
	  	z_blue = 2.9097
	  	g_cen_bl = lam1*(1.+z_blue)

	#flux variance
	var_flux    = var_spec.data
	n 			= len(var_flux)
	inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

	# SINGLET states
	if spec_feat == 'HeII':
		mod 	= lm.Model(dgauss) 

		pars 	= Parameters()

		pars.add_many( ('g_cen1', g_blue, False),
			('amp1', amp_blue, False),('wid1', wid_blue, False ),
			('amp2', amp_HeII, False),('wid2', wid_HeII, False ),
			('g_cen2', wav_o_HeII, False),
			('cont', cont_HeII, False) )	

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		fwhm_kms	= convert_sigma_fwhm(wid_HeII, wid_HeII_err, wav_o_HeII, wav_o_HeII_err)

		vel1 = vel(wav_o_HeII, wav_e_HeII, z)
		vel1_err = abs( vel1*( np.sqrt( ( wav_o_HeII_err/wav_o_HeII )**2 + (z_err/z)**2 - 2*( wav_o_HeII_err/wav_o_HeII )*(z_err/z)\
		+ wav_o_HeII_err**2 + z_err**2 ) ) )

		pl.plot(wav, gauss(wav, amp_blue, wid_blue, g_blue, cont_HeII), ls='--', c='blue', label='Blueshifted component')
		pl.plot(wav, gauss(wav, amp_HeII, wid_HeII, wav_o_HeII, cont_HeII ), ls='--', c='orange', label=r'HeII $\lambda$'+`int(lam)`)
		pl.plot(wav,fit.best_fit,'r',label='Best fit')

	elif spec_feat == 'NIV]':
		mod 	= lm.Model(gauss) 

		pars 	= Parameters()

		#initial guess Gaussian width for CIV lines
		guess_wid = g_cen * ( wid_HeII / wav_o_HeII )

		pars.add_many( ('g_cen', g_cen, True, g_cen-delg, g_cen+delg),
			('amp', amp_HeII, True, 10.),('wid', guess_wid, True, 0. ),
			('cont', cont_HeII, True, 0.) )	

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		print fit.fit_report()

		amp 		= fit.params['amp'].value
		amp_err		= fit.params['amp'].stderr

		wav_o 		= fit.params['g_cen'].value
		wav_o_err 	= fit.params['g_cen'].stderr

		wid 		= fit.params['wid'].value
		wid_err 	= fit.params['wid'].stderr	

		fwhm_kms	= convert_sigma_fwhm(wid, wid_err, wav_o, wav_o_err)

		pl.plot(wav, fit.best_fit,'r',label='Best fit')

	elif spec_feat == 'CII]':
		mod 	= lm.Model(dgauss)

		#initial guess Gaussian width for CIV lines
		guess_wid = g_cen * ( wid_HeII / wav_o_HeII  )

		pars.add_many(	('g_cen1', g_cen_blue, True, g_cen_blue-delg, g_cen_blue+delg),\
			('amp1', amp_HeII, True, 0.),('wid1', guess_wid, True, 0. ),\
			('cont', cont_HeII, True, 0.), 
			('g_cen2', g_cen, True, g_cen-delg, g_cen+delg),
			('amp2', amp_HeII, True, 0.),('wid2', guess_wid, True, 0. ))	

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		print fit.fit_report()

		res 		= fit.params
		amp 		= (res['amp1'].value, res['amp2'].value) 
		amp_err		= (res['amp1'].stderr, res['amp2'].stderr)

		wav_o 		= (res['g_cen1'].value, res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr, res['g_cen2'].stderr)

		wid 		= (res['wid1'].value, res['wid2'].value)
		wid_err 	= (res['wid1'].stderr, res['wid2'].stderr)

		cont 		= res['cont'].value	

		fwhm_kms		= convert_sigma_fwhm(wid[0], wid_err[0], wav_o[0], wav_o_err[0])
		fwhm_kms_blue	= convert_sigma_fwhm(wid[1], wid_err[1], wav_o[1], wav_o_err[1])

		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont), color='blue', linestyle='--', label='Blueshifted '+spec_feat+r' $\lambda$'+`int(lam)`)
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont), color='orange', linestyle='--', label='Systemic '+spec_feat+r' $\lambda$'+`int(lam)`)
		pl.plot(wav, fit.best_fit,'r',label='Best fit')

	# DOUBLET lines
	elif spec_feat == 'OIII]':
		mod 	= lm.Model(dgauss)

		pars 	= Parameters()
		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.0031912331406552*g_cen1') 

		pars.add_many(  ('amp1', amp_HeII, True, 0.,),\
			('amp2', amp_HeII, True, 0.,),\
			('cont', cont_HeII, True, 0.) )

		#initial guess Gaussian width for CIV lines
		guess_wid = g_cen1 * ( wid_HeII / wav_o_HeII )

		pars.add('wid1', guess_wid, True, 0.)
		pars.add('wid2', expr='wid1')

		fit 		= mod.fit(flux,pars,x=wav,weights=inv_noise)

		print fit.fit_report()
		
		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)

		fwhm1_kms	= convert_sigma_fwhm(wid[0], wid_err[0], wav_o[0], wav_o_err[0])
		fwhm2_kms	= convert_sigma_fwhm(wid[1], wid_err[1], wav_o[1], wav_o_err[1])

		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value
	
		pl.plot(wav, fit.best_fit,'r',label='Best fit')
		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont),color='green',linestyle='--',label='Systemic '+spec_feat+r' $\lambda$'+`int(lam1)`)
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont),color='orange',linestyle='--',label='Systemic '+spec_feat+r' $\lambda$'+`int(lam2)`)
	
	elif spec_feat == 'CIII]':
		pars 	= Parameters()

		guess_wid1 = g_cen1 * ( wid_HeII / wav_o_HeII  )

		pars.add_many( ('g_cen1_bl', g_cen_bl, True, g_cen_bl-delg, g_cen_bl+delg),
			('amp1_bl', amp_HeII, True, 0.015*amp_HeII),('wid1_bl', guess_wid1, True, 0., 10. ),\
			('cont', cont_HeII, True, 0.) )	

		pars.add('g_cen2_bl', expr='1.0010489327109666*g_cen1_bl')
		pars.add('amp2_bl', expr='0.5*amp1_bl')
		pars.add('wid2_bl', expr='wid1_bl')

		pars.add_many( ('g_cen1', g_cen, True, g_cen-delg, g_cen+delg),
			('amp1', amp_HeII, True, 0.),('wid1', guess_wid1, True, 0. ))

		pars.add('g_cen2', expr='1.0010489327109666*g_cen1')
		pars.add('amp2', expr='0.5*amp1')
		pars.add('wid2', expr='wid1')

		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.0010489327109666*g_cen1') 

		pars.add('amp1', amp_HeII, True, 0.)
		pars.add('amp2', expr='0.5*amp1')

		def CIII_model(p):
			x = wav
			mod = dgauss_nocont(x, p['amp1_bl'], p['wid1_bl'], p['g_cen1_bl'], p['amp1'], p['wid1'], p['g_cen1'])\
			 + dgauss_nocont(x, p['amp2_bl'], p['wid2_bl'], p['g_cen2_bl'], p['amp2'], p['wid2'], p['g_cen2']) + cont
			data = flux
			weights = inv_noise
			return weights*(mod - data)

		#create minimizer
		mini = Minimizer(CIII_model, params=pars, nan_policy='omit')

		#solve with Levenberg-Marquardt
		fit = mini.minimize(method='leastsq')

		report_fit(fit)

		res 		= fit.params
		wav_o_bl 		= (res['g_cen1_bl'].value,res['g_cen2_bl'].value)
		wav_o_err_bl 	= (res['g_cen1_bl'].stderr,res['g_cen2_bl'].stderr)
		
		wid_bl 			= (res['wid1_bl'].value,res['wid2_bl'].value)
		wid_err_bl  	= (res['wid1_bl'].stderr,res['wid2_bl'].stderr)

		fwhm1_kms_bl	= convert_sigma_fwhm(wid_bl[0], wid_err_bl[0], wav_o_bl[0], wav_o_err_bl[0])
		fwhm2_kms_bl	= convert_sigma_fwhm(wid_bl[1], wid_err_bl[1], wav_o_bl[1], wav_o_err_bl[1])

		amp_bl 		= (res['amp1_bl'].value,res['amp2_bl'].value)
		amp_err_bl 	= (res['amp1_bl'].stderr,res['amp2_bl'].stderr)

		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)

		fwhm1_kms	= convert_sigma_fwhm(wid[0], wid_err[0], wav_o[0], wav_o_err[0])
		fwhm2_kms	= convert_sigma_fwhm(wid[1], wid_err[1], wav_o[1], wav_o_err[1])

		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value

		fn = dgauss_nocont(wav, amp_bl[0], wid_bl[0], wav_o_bl[0], amp[0], wid[0], wav_o[0])\
		+ dgauss_nocont(wav, amp_bl[1], wid_bl[1], wav_o_bl[1], amp[1], wid[1], wav_o[1]) + cont

		pl.plot(wav, gauss_nocont(wav,amp_bl[0],wid_bl[0],wav_o_bl[0]) + cont,color='blue',linestyle='--',label='Blueshifted '+spec_feat+r' $\lambda$'+`int(lam1)`)
		pl.plot(wav, gauss_nocont(wav,amp_bl[1],wid_bl[1],wav_o_bl[1]) + cont,color='blue',linestyle='-.',label='Blueshifted '+spec_feat+r' $\lambda$'+`int(lam2)`)
		
		pl.plot(wav, gauss_nocont(wav,amp[0],wid[0],wav_o[0]) + cont,color='green',linestyle='--',label='Systemic '+spec_feat+r' $\lambda$'+`int(lam1)`)
		pl.plot(wav, gauss_nocont(wav,amp[1],wid[1],wav_o[1]) + cont,color='orange',linestyle='-.',label='Systemic '+spec_feat+r' $\lambda$'+`int(lam2)`)

		pl.plot(wav, fn, c='red', label='Best fit')

	else:
		print "No data for ion"

	chisqr = r'$\chi^2$: %1.2f' %fit.chisqr
	redchisqr = r'$\chi_\nu^2$ = %1.2f' %fit.redchi
	pl.text(0.12,0.95, redchisqr, ha='center', va='center',transform = ax.transAxes, fontsize=18)

	pl.plot(wav,flux,drawstyle='steps-mid',color='k')	#plot data
	
	#fill grey between zero on flux axis and data
	pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')	
	pl.legend(frameon=False)

	# ------------------------------------------------------------
	#   Format best fit results to output them into a text file
	# ------------------------------------------------------------
	
	filename = "./out/line-fitting/emission_line_fit/0943 fit.txt"

	if spec_feat 	== 'HeII':
		wav_e = lam
		header = np.array( \
			["#wav0(Ang)          wav0_err       flux(10^{-17}erg/s/cm^2)     flux_err	sigma(Ang)    	   sigma_err   	  fwhm(km/s)   		fwhm_err      wav_rest spec_feat"] )

		gauss_fit_params = np.array(['%.2f'%wav_o_HeII, '%.2f'%wav_o_HeII_err, '%.2f'%(amp_HeII/1.e3), '%.2f'%(amp_HeII_err/1.e3),\
		 '%.2f'%wid_HeII, '%.2f'%wid_HeII_err, '%.2f'%fwhm_kms[0], '%.2f'%fwhm_kms[1], '%.2f'%wav_e, '%s'%spec_feat])

		with open(filename,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')	

	elif spec_feat 	== 'CII]':
		wav_e = lam

		gauss_fit_params2 = np.array(['%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%(amp[0]/1.e3), '%.2f'%(amp_err[0]/1.e3), '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm_kms_blue[0], '%.2f'%fwhm_kms_blue[1], '%.2f'%wav_e,'%s'%spec_feat+'_blue'])

		gauss_fit_params1 = np.array(['%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%(amp[1]/1.e3), '%.2f'%(amp_err[1]/1.e3), '%.2f'%wid[1],\
			'%.2f'%wid_err[1], '%.2f'%fwhm_kms[0], '%.2f'%fwhm_kms[1], '%.2f'%wav_e, '%s'%spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	elif spec_feat 	== 'NIV]':
		wav_e = lam
	
		gauss_fit_params = np.array(['%.2f'%wav_o, '%.2f'%wav_o_err, '%.2f'%(amp/1.e3), '%.2f'%(amp_err/1.e3),\
		 '%.2f'%wid, '%.2f'%wid_err, '%.2f'%fwhm_kms[0], '%.2f'%fwhm_kms[1], '%.2f'%wav_e, '%s'%spec_feat])

		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')

	elif spec_feat == 'OIII]':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array(['%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%(amp[0]/1.e3), '%.2f'%(amp_err[0]/1.e3), '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm1_kms[0], '%.2f'%fwhm1_kms[1], '%.2f'%wav_e1, '%s'%spec_feat])

		gauss_fit_params2 = np.array(['%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%(amp[1]/1.e3), '%.2f'%(amp_err[1]/1.e3), '%.2f'%wid[1],\
			'%.2f'%wid_err[1], '%.2f'%fwhm2_kms[0], '%.2f'%fwhm2_kms[1], '%.2f'%wav_e2, '%s'%spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')

	elif spec_feat == 'CIII]':
		wav_e1 = lam1
		wav_e2 = lam2

		# first doublet wavelength  
		gauss_fit_params1 = np.array(['%.2f'%wav_o_bl[0], '%.2f'%wav_o_err_bl[0],\
			'%.2f'%(amp_bl[0]/1.e3), '%.2f'%(amp_err_bl[0]/1.e3), '%.2f'%wid_bl[0],\
			'%.2f'%wid_err_bl[0], '%.2f'%fwhm1_kms_bl[0], '%.2f'%fwhm1_kms_bl[1], '%.2f'%wav_e1, '%s'%spec_feat+'_blue'])

		gauss_fit_params2 = np.array(['%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%(amp[0]/1.e3), '%.2f'%(amp_err[0]/1.e3), '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm1_kms[0], '%.2f'%fwhm1_kms[1], '%.2f'%wav_e1, '%s'%spec_feat])

		# second doublet wavelength  
		gauss_fit_params3 = np.array(['%.2f'%wav_o_bl[1], '%.2f'%wav_o_err_bl[1],\
			'%.2f'%(amp_bl[1]/1.e3), '%.2f'%(amp_err_bl[1]/1.e3), '%.2f'%wid_bl[1],\
			'%.2f'%wid_err_bl[1], '%.2f'%fwhm2_kms_bl[0], '%.2f'%fwhm2_kms_bl[1], '%.2f'%wav_e2, '%s'%spec_feat+'_blue'])

		gauss_fit_params4 = np.array(['%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%(amp[1]/1.e3), '%.2f'%(amp_err[1]/1.e3), '%.2f'%wid[1],\
			'%.2f'%wid_err[1], '%.2f'%fwhm2_kms[0], '%.2f'%fwhm2_kms[1], '%.2f'%wav_e2, '%s'%spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,
				gauss_fit_params2, gauss_fit_params3, gauss_fit_params4) ) )
			f.write('\n')

	
	# -----------------------------------------------
	#    	PLOT with velocity and uJy axes
	# -----------------------------------------------
	#LSR velocity
	vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	#singlet 
	if spec_feat == 'HeII':	
		vel_meas = vel(wav_o_HeII, wav_e_HeII, z)		#central velocity of detected line
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e, z) - vel0 ) )

	elif spec_feat == 'NIV]':
		vel_meas = vel(wav_o, wav_e, z)		
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e, z) - vel0 ) )

	elif spec_feat == 'CII]':
		vel_meas = vel(wav_o[1], wav_e, z)				
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e, z) - vel0 ) )
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e, z) - vel0 ) )
	
	#doublet 
	else:
		vel_meas = [vel(wav_o[0],wav_e1,z),vel(wav_o[1],wav_e2,z)]	
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e2, z) - vel0)	) 

		ax_up = ax.twiny()
		xticks_up = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x, wav_e1, z) - vel0) ) 
		ax_up.xaxis.set_major_formatter(xticks_up)
	
	# define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)
	
	if spec_feat == 'HeII':
		wav_cent = wav_o_HeII 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy

	elif spec_feat == 'NIV]':
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   

	elif spec_feat == 'CII]':
		wav_cent = wav_o[0]
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   

	else:
		wav_cent = wav_o[1]
		maxf = flux_Jy(wav_cent,max(flux))*1.e6 

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
	ymax = 1.4*max(flux)
	ymin = -0.01*max(flux)

	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)

	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	
	pl.plot([xmin,xmax],[0.,0.],ls='-',color='grey')	#zero flux density-axis
	ax.set_xlabel( r'Velocity (km/s)', fontsize=18)
	ax.set_ylabel( r'Flux Density ($\mu$Jy)', fontsize=18 )
	ax.set_ylim([ymin,ymax])
	ax.set_xlim([xmin,xmax])
	pl.savefig('./out/line-fitting/emission_line_fit/'+spec_feat+'_0.4_fit.png')
	if spec_feat in ('CIII]', 'CII]'):
		pl.savefig(home+'PUBLICATIONS/0943_absorption/plots/'+spec_feat+'_fit.pdf')

#align columns in text file for better readability
res = "./out/line-fitting/emission_line_fit/0943 fit.txt"
fit = open("./out/line-fitting/emission_line_fit/0943_fit_0.4.txt", "w")

with open(res, 'r') as f:
    for line in f:
        data = line.split()    # Splits on whitespace
        x = '{0[0]:<10}{0[1]:<10}{0[2]:<25}{0[3]:<10}{0[4]:<10}{0[5]:<10}{0[6]:<10}{0[7]:<10}{0[8]:<10}{0[9]:<5}'.format(data)
        fit.write(x[:]+'\n')	

#delete original file
os.system('rm ./out/line-fitting/emission_line_fit/0943\ fit.txt')