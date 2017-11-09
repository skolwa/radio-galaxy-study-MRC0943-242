#S.N. Kolwa (2017)
#0943_line_fit.py
# Purpose:  
# - Non-resonant line profile fit i.e. Gaussian and Lorentzian only
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)

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

import sys

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat   = [  'NV', 'CII', 'SiIV', 'NIV]','HeII','OIII]','CIII]','CII]']

for spec_feat in spec_feat:

	# -----------------------------------------------
	# 	 	FIT to HeII for systemic velocity
	# -----------------------------------------------

	fname 			= "./out/HeII_cs.fits"
	hdr 	 		= mpdo.Cube(fname,ext=0)
	datacube 		= mpdo.Cube(fname,ext=1)
	varcube  		= mpdo.Cube(fname,ext=2)
	
	HeII 			= datacube.subcube_circle_aperture(center=(84,45),radius=10,\
		unit_center	=None,unit_radius=None)
	spec_HeII 		= HeII.sum(axis=(1,2))
	
	wav 	=  spec_HeII.wave.coord()  		#Ang
	flux 	=  spec_HeII.data				#1.e-20 erg / s / cm^2 / Ang
	
	lorenz 	= lm.LorentzianModel(prefix='lorenz_')
	pars 	= lorenz.make_params()
	pars.add('lorenz_center',value=6433.,min=6430.,max=6436.)
	pars.add('lorenz_sigma',value=10.,min=5.,max=15.)
	
	gauss 	= lm.GaussianModel(prefix='gauss_')
	pars.update(gauss.make_params())
	
	pars['gauss_center'].set(6420.)
	
	#composite model
	mod 	= lorenz + gauss
	
	#model fit
	out 	= mod.fit(flux,pars,x=wav)
	
	#fit params
	report 			= out.fit_report(min_correl=0.1)
	res 			= out.params
	
	wav_o_HeII 		= res['lorenz_center'].value
	wav_o_err_HeII	= res['lorenz_center'].stderr
	amp_HeII 		= res['lorenz_amplitude'].value
	amp_err_HeII 	= res['lorenz_amplitude'].stderr
	fwhm_HeII 		= res['lorenz_fwhm'].value
	fwhm_err_HeII	= res['lorenz_fwhm'].stderr

	wav_e_HeII 		= 1640.4
	
	#--------------------------
	# LOAD spectral data cubes
	#--------------------------
	
	fname 				= "./out/"+spec_feat+"_cs.fits"
	hdr 	 			= mpdo.Cube(fname,ext=0)
	datacube 			= mpdo.Cube(fname,ext=1)
	varcube  			= mpdo.Cube(fname,ext=2)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRUM for LINE FIT
	# -----------------------------------------------
	glx 	= datacube.subcube_circle_aperture(center=(84,45),radius=4,\
			unit_center=None,unit_radius=None)
	
	# fig = pl.figure()
	# img 			= glx.sum(axis=0)
	# continuum 		= img.plot( scale='arcsinh' )
	# pl.colorbar(continuum,orientation = 'vertical')
	# pl.savefig('./out/line-fitting/'+spec_feat+'_img.png')
	
	fig 		= pl.figure()
	spec 		= glx.sum(axis=(1,2))
	
	wav_ax 		=  spec.wave.coord()  		#1.e-8 cm
	flux_ax 	=  spec.data				#1.e-20 erg / s / cm^2 / Ang

	maxf 		= max(flux_ax)
	
	fig,ax = pl.subplots()
	pl.plot(wav_ax,flux_ax,c='k',drawstyle='steps-mid')
	
	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v
	
	c = 2.9979245800e10
	
	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = obs frame (z=0)
	z = vel0_rest/c

	def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2):
		gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
		gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
		return gauss1 + gauss2
	
	# -----------------------------------
	# 	  MODEL FIT to EMISSION LINE
	# -----------------------------------
	#SINGLET states
	#Gaussian
	if spec_feat in ('CII', 'NIV]','SiIV'): 
		mod 	= lm.GaussianModel()	
		
		pars 	= mod.guess(flux_ax,x=wav_ax)
		out 	= mod.fit(flux_ax,pars,x=wav_ax)
		report 	= out.fit_report(min_correl=0.1)
	
		res 	= out.params
	
		wav_o 		= res['center'].value
		wav_o_err 	= res['center'].stderr
	
		fwhm 		= res['fwhm'].value
		fwhm_err	= res['fwhm'].stderr
	
		amp 		= res['amplitude'].value
		amp_err		= res['amplitude'].stderr
		
		pl.plot(wav_ax,out.best_fit,'r--')
	
	#Gaussian plus Lorentzian fit
	elif spec_feat == 'HeII': 
		lorenz 		= lm.LorentzianModel(prefix='lorenz_')
		pars['lorenz_center'].set(6433.)
		pars['lorenz_sigma'].set(10.)
		
		gauss 		= lm.GaussianModel(prefix='gauss_')
		pars.update(gauss.make_params())
		
		pars['gauss_center'].set(6420.)
	
		mod 	= lorenz + gauss
		out 	= mod.fit(flux_ax,pars,x=wav_ax)
		report 	= out.fit_report(min_correl=0.1)
	
		res 	= out.params
			
		comps 	= out.eval_components(x=wav)
	
		pl.plot(wav_ax,out.best_fit,'r--')
	
	#Lorentzian
	elif spec_feat == 'CII]':	
		mod 	= lm.LorentzianModel()	
		
		pars 	= mod.guess(flux_ax,x=wav_ax)
		out 	= mod.fit(flux_ax,pars,x=wav_ax)
		report 	= out.fit_report(min_correl=0.1)
	
		res 	= out.params
	
		wav_o 		= res['center'].value
		wav_o_err 	= res['center'].stderr
	
		fwhm 		= res['fwhm'].value
		fwhm_err	= res['fwhm'].stderr
	
		amp 		= res['amplitude'].value
		amp_err		= res['amplitude'].stderr
		
		pl.plot(wav_ax,out.best_fit,'r--')
	
	#DOUBLET states
	#double Gaussian
	elif spec_feat == 'NV':
		mod 	= lm.Model(dgauss)
	
		g_cen1 = 1238.8*(1.+z)
	 	g_cen2 = 1242.8*(1.+z)
	
		pars 	= Parameters()
		pars.add('g_cen1',value=g_cen1)
		pars.add('g_cen2',value=g_cen2,expr='1.0032289312237648*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add_many(  
			('amp1',maxf), ('wid1',10.),('amp2',maxf),('wid2',10.)
			)
	
		out 	= mod.fit(flux_ax,pars,x=wav_ax)

		res 		= out.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
	
		fwhm 		= (res['wid1'].value,res['wid2'].value)
		fwhm_err  	= (res['wid1'].stderr,res['wid2'].stderr)
	
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)
	
	elif spec_feat == 'OIII]':
		mod 	= lm.Model(dgauss)
	
		g_cen1 = 1660.8*(1.+z)
	 	g_cen2 = 1666.1*(1.+z)
	
		pars 	= Parameters()
		pars.add('g_cen1',value=g_cen1)
		pars.add('g_cen2',value=g_cen2,expr='1.0031912331406552*g_cen1') 

		pars.add_many(  
			('amp1',maxf), ('wid1',10.), ('amp2',maxf), ('wid2',10.)
			)
	
		out 	= mod.fit(flux_ax,pars,x=wav_ax)

		res 		= out.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
	
		fwhm 		= (res['wid1'].value,res['wid2'].value)
		fwhm_err  	= (res['wid1'].stderr,res['wid2'].stderr)
	
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)
	
	elif spec_feat == 'CIII]':
		mod 	= lm.Model(dgauss)
	
		g_cen1 = 1906.7*(1.+z)
	 	g_cen2 = 1908.7*(1.+z)
	
		pars 	= Parameters()
		pars.add('g_cen1',value=g_cen1)
		pars.add('g_cen2',value=g_cen2,expr='1.0010489327109666*g_cen1') 

		pars.add_many(  
			('amp1',maxf), ('wid1',10.), ('amp2',maxf), ('wid2',10.)
			)
	
		out 	= mod.fit(flux_ax,pars,x=wav_ax)
	
		res 		= out.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
	
		fwhm 		= (res['wid1'].value,res['wid2'].value)
		fwhm_err  	= (res['wid1'].stderr,res['wid2'].stderr)
	
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

	report 	= out.fit_report(min_correl=0.1)
	# print report

	pl.plot(wav_ax,out.best_fit,'r--')
	pl.fill_between(wav_ax,flux_ax,color='grey',interpolate=True,step='mid')

	# -----------------------------------------------
	# 	      MPDAF GAUSSIAN FIT PARAMETERS
	# -----------------------------------------------
	
	filename = "./out/line-fitting/0943 fit.txt"
	
	#rest (lab-frame) wavelengths of emission lines

	if spec_feat 	== 'NV':
		wav_e1 = 1238.8
		wav_e2 = 1242.8
		header = np.array( \
			["#wav0          err_wav0        flux_peak      err_flux_peak\
						  FWHM    	      err_FWHM     wav_rest spec_feat"] )

		gauss_fit_params1 = np.array([wav_o[0], wav_o_err[0],\
			amp[0], amp_err[0], fwhm[0],\
			fwhm_err[0], wav_e1, spec_feat])

		gauss_fit_params2 = np.array([wav_o[1], wav_o_err[1],\
			amp[1], amp_err[1], fwhm[1],\
			fwhm_err[1], wav_e2, spec_feat])

		with open(filename,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	elif spec_feat 	== 'HeII':
		wav_e = 1640.4
		wav_o = wav_o_HeII
		gauss_fit_params = np.array([wav_o, wav_o_err_HeII, amp_HeII,\
		 amp_err_HeII, fwhm_HeII, fwhm_err_HeII, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'CII]':
		wav_e = 2326.0
		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 fwhm, fwhm_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'NIV]':
		wav_e = 1486.5
		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 fwhm, fwhm_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'SiIV':
		wav_e = 1402.8
		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 fwhm, fwhm_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'CII':
		wav_e = 1338.0
		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 fwhm, fwhm_err, wav_e, spec_feat])
		with open(filename,'a') as f:
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')
	
	elif spec_feat 	== 'OIII]':
		wav_e1 = 1660.8
		wav_e2 = 1666.1

		gauss_fit_params1 = np.array([wav_o[0], wav_o_err[0],\
			amp[0], amp_err[0], fwhm[0],\
			fwhm_err[0], wav_e1, spec_feat])

		gauss_fit_params2 = np.array([wav_o[1], wav_o_err[1],\
			amp[1], amp_err[1], fwhm[1],\
			fwhm_err[1], wav_e2, spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	elif spec_feat 	== 'CIII]':
		wav_e1 = 1906.7
		wav_e2 = 1908.7

		gauss_fit_params1 = np.array([wav_o[0], wav_o_err[0],\
			amp[0], amp_err[0], fwhm[0],\
			fwhm_err[0], wav_e1, spec_feat])

		gauss_fit_params2 = np.array([wav_o[1], wav_o_err[1],\
			amp[1], amp_err[1], fwhm[1],\
			fwhm_err[1], wav_e2, spec_feat])

		with open(filename,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------
	
	#precise speed of light in km/s
	c   = 299792.458											
	
	vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):	
		vel_meas = vel(wav_o,wav_e,z)		#central velocity of detected line
		vel_off = vel_meas - vel0			#residual velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z) - vel0 ) )
	 	# print vel_meas
	 	# print vel_off
	
	#doublet state
	else:
		vel_meas = [vel(wav_o[0],wav_e1,z),vel(wav_o[1],wav_e2,z)]	#central velocity of line1
		vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 
	
	#define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)
	
	#convert flux units to Jy
	def flux_Jy(wav,flux):
		f = 3.33564095e4*flux*1.e-20*wav**2
		return f
	
	#define y-tick marks in reasonable units
	#recover flux in 10^-20 erg/s/cm^2 and convert to microJy
	def flux(wav,flux_Jy):
		f = flux_Jy/(3.33564095e4*wav**2)
		return f*1.e20
	
	#central wavelength depending on singlet or doublet fit
	if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy
		# print maxf
	
	else:
		wav_cent = wav_o[1]
		maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy	
	
	if maxf < 1.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,0.5e-6)
		major_yticks = [ flux0, flux1 ]
	
	elif maxf > 1. and maxf < 4.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,1.e-6)
		flux2 = flux(wav_cent,2.e-6)
		flux3 = flux(wav_cent,2.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]
	
	elif maxf > 4. and maxf < 5.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,1.e-6)
		flux2 = flux(wav_cent,2.e-6)
		flux3 = flux(wav_cent,3.e-6)
		flux4 = flux(wav_cent,4.e-6)
		flux5 = flux(wav_cent,5.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]
	
	elif maxf > 5. and maxf < 15.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,5.e-6)
		flux2 = flux(wav_cent,10.e-6)
		flux3 = flux(wav_cent,10.e-6)
		flux4 = flux(wav_cent,15.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	elif maxf > 15. and maxf < 30.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,10.e-6)
		flux2 = flux(wav_cent,20.e-6)
		flux3 = flux(wav_cent,30.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]
	
	elif maxf > 30.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,10.e-6)
		flux2 = flux(wav_cent,20.e-6)
		flux3 = flux(wav_cent,30.e-6)
		flux4 = flux(wav_cent,40.e-6)	
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
		if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
			wav1 = wav_e*(1.+z)*(1.+(v1/c))
			wav2 = wav_e*(1.+z)*(1.+(v2/c))
			return wav1,wav2
		else:
			wav1 = wav_e2*(1.+z)*(1.+(v1/c))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c))
			return wav1,wav2
	
	#define x-axis as offset velocity from Vsys/zero-point (km/s)
	wav0 		= offset_vel_to_wav(0.)
	wavlim05 	= offset_vel_to_wav(500.)
	wavlim1 	= offset_vel_to_wav(1000.)
	wavlim15 	= offset_vel_to_wav(1500.)
	wavlim2 	= offset_vel_to_wav(2000.)
	
	major_ticks = [ wavlim2[0], wavlim15[0], wavlim1[0], wavlim05[0], wav0[1],\
	wavlim05[1], wavlim1[1], wavlim15[1], wavlim2[1] ]
	ax.set_xticks(major_ticks)
	
	#define x-limits
	if spec_feat in ('CII','NV','HeII','SiIV','NIV]','OIII]','CIII]'):
		xmin = wavlim2[0] - 2.
		xmax = wavlim2[1] + 2.
	
	elif spec_feat in ('CII]'):
		xmin = wavlim15[0] - 2.
		xmax = wavlim15[1] + 2.
	
	#define y-limits
	ymax = 1.2*max(flux_ax)
	ymin = -0.1*max(flux_ax)
	
	#draw line representing central velocity of spectral feature
	if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
		pl.plot([wav_o,wav_o],[ymin,ymax],color='green',ls='--')
	
	else:
		pl.plot([wav_o[0],wav_o[0]],[ymin,ymax],color='green',ls='--')
		pl.plot([wav_o[1],wav_o[1]],[ymin,ymax],color='green',ls='--')
	
	#draw plot
	pl.title(spec_feat+' Fit')
	ax.set_xlabel( 'Velocity Offset (km/s)' )
	ax.set_ylabel( r'Flux Density ($\mu$Jy)' )
	ax.set_ylim([ymin,ymax])
	ax.set_xlim([xmin,xmax])
	pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis

	pl.savefig('./out/line-fitting/'+spec_feat+' Fit.eps')
	# print '----------------------------------------------------------'			