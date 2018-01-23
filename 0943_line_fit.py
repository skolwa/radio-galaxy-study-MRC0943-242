
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

spec_feat   = [ 'CII', 'NIV]', 'HeII','OIII]','CIII]','CII]']

#pixel radius for extraction
radius = 6
center = (84,108)

for spec_feat in spec_feat:
	# -----------------------------------------------
	# 	 	FIT to HeII for systemic velocity
	# -----------------------------------------------

	fname 			= "./out/HeII.fits"
	hdr 	 		= mpdo.Cube(fname,ext=0)
	datacube 		= mpdo.Cube(fname,ext=1)
	varcube  		= mpdo.Cube(fname,ext=2)
	
	HeII 			= datacube.subcube_circle_aperture(center=center,radius=radius,\
		unit_center	=None,unit_radius=None)
	spec_HeII 		= HeII.sum(axis=(1,2))

	var_glx = varcube.subcube_circle_aperture(center=(84,45),radius=6,\
			unit_center=None,unit_radius=None)
	var_spec = var_glx.sum(axis=(1,2))

	#continuum subtract the spectrum
	spec_HeII.mask_region(6366.,6570.)
	cont = spec_HeII.poly_spec(1)

	spec_HeII.unmask()

	line_HeII = spec_HeII - cont

	wav 	=  line_HeII.wave.coord()  		#1.e-8 cm
	flux 	=  line_HeII.data				#1.e-20 erg / s / cm^2 / Ang
	
	lorenz 	= lm.LorentzianModel(prefix='lorenz_')
	pars 	= lorenz.make_params()
	pars.add('lorenz_center',value=6433.,min=6432.,max=6434.)
	pars.add('lorenz_sigma',value=10.,min=5.,max=15.)
	
	# gauss 	= lm.GaussianModel(prefix='gauss_')
	# pars.update(gauss.make_params())
	
	# pars['gauss_center'].set(6420.,min=6410.,max=6430.) 		#blueshifted component
	
	#composite model
	mod 	= lorenz 
	
	#model fit
	out 	= mod.fit(flux,pars,x=wav)
	
	#fit params
	report 			= out.fit_report(min_correl=0.1)
	# print report

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
	
	fname 				= "./out/"+spec_feat+".fits"
	hdr 	 			= mpdo.Cube(fname,ext=0)
	datacube 			= mpdo.Cube(fname,ext=1)
	varcube  			= mpdo.Cube(fname,ext=2)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRUM for LINE FIT
	# -----------------------------------------------
	glx 	= datacube.subcube_circle_aperture(center=center,radius=radius,\
			unit_center=None,unit_radius=None)

	var_glx = varcube.subcube_circle_aperture(center=center,radius=radius,\
			unit_center=None,unit_radius=None)

	var_spec = var_glx.sum(axis=(1,2))
	
	## extract 2D image
	# fig = pl.figure()
	# img 			= glx.sum(axis=0)
	# continuum 		= img.plot( scale='arcsinh' )
	# pl.colorbar(continuum,orientation = 'vertical')
	# pl.savefig('./out/line-fitting/'+spec_feat+'_img.png')
	
	fig,ax = pl.subplots(1,1)
	spec   = glx.sum(axis=(1,2))
	
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
	if spec_feat in 'CII':				#Gaussian	
		spec.mask_region(5216.,5282.)
		cont 	= spec.poly_spec(1)
		spec.unmask()
	
		line 	= spec - cont
	
		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang

		mod 	= lm.GaussianModel()	
		
		lam = 1338.*(1.+z)

		pars = mod.guess(flux, x=wav)
		pars.update(mod.make_params())
		pars['center'].set(lam,min=lam-0.5,max=lam+0.5)

		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)
		report 	= fit.fit_report(min_correl=0.1)
		# print report

		res 		= fit.params
		wav_o 		= res['center'].value
		wav_o_err 	= res['center'].stderr
	
		fwhm 		= res['fwhm'].value
		fwhm_err	= res['fwhm'].stderr
	
		amp 		= res['amplitude'].value
		amp_err		= res['amplitude'].stderr

	elif spec_feat == 'NIV]':
		spec.mask_region(5452.,5558.)
		spec.mask_region(5572.,5584.)
		cont 	= spec.poly_spec(1)
		spec.unmask()
	
		line 	= spec - cont
	
		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang

		mod 	= lm.GaussianModel()	
		
		lam = 1486.5*(1.+z)

		pars = mod.guess(flux, x=wav)
		pars.update(mod.make_params())
		pars['center'].set(lam,min=lam-0.5,max=lam+0.5)

		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)

		report  = fit.fit_report(min_correl=0.1)
		# print report

		res 		= fit.params
		wav_o 		= res['center'].value
		wav_o_err 	= res['center'].stderr
	
		fwhm 		= res['fwhm'].value
		fwhm_err	= res['fwhm'].stderr
	
		amp 		= res['amplitude'].value
		amp_err		= res['amplitude'].stderr
			
	elif spec_feat == 'HeII':			#Gaussian and Lorentzian
		spec.mask_region(6366.,6570.)
		spec.mask_region(6298.,6302.)
		cont 	= spec.poly_spec(1)
		spec.unmask()

		line 	= spec - cont
	
		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang

		lam = 1640.4*(1.+z)

		lorenz 	= lm.LorentzianModel(prefix='lorenz_')
		pars 	= lorenz.make_params()
		pars.add('lorenz_center',value=6433.,min=lam-0.5,max=lam+0.5)
		pars.add('lorenz_sigma',value=10.,min=5.,max=15.)
		
		# gauss 	= lm.GaussianModel(prefix='gauss_')
		# pars.update(gauss.make_params())
		
		# pars['gauss_center'].set(6420.,min=6410.,max=6430.) 		#blueshifted component
		# pars['gauss_amplitude'].set(min=0.)
		
		#composite model
		mod 	= lorenz 
		
		#model fit
		fit 	= mod.fit(flux,pars,x=wav)
		
		#fit params
		report 			= fit.fit_report(min_correl=0.1)
		# print report
	
		res 			= fit.params
		comps 			= fit.eval_components(x=wav)
	
		# pl.plot(wav,comps['lorenz_'],'b--')
		# pl.plot(wav,comps['gauss_'],'b--')

	elif spec_feat == 'CII]':
		spec.mask_region(9070.,9194.)

		cont = spec.poly_spec(1)

		spec.unmask()

		line 	= spec - cont
	
		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang

		pl.plot(wav,flux,drawstyle='steps-mid',c='k')
		maxf 	= max(flux)
		mod 	= lm.LorentzianModel()	
		
		lam = 2326.*(1.+z)

		pars = mod.guess(flux, x=wav)
		pars.update(mod.make_params())
		pars['center'].set(lam,min=lam-0.5,max=lam+0.5)

		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)	
		report 	= fit.fit_report(min_correl=0.1)	
		# print report

		res 		= fit.params
		wav_o 		= res['center'].value
		wav_o_err 	= res['center'].stderr
	
		fwhm 		= res['fwhm'].value
		fwhm_err	= res['fwhm'].stderr
	
		amp 		= res['amplitude'].value
		amp_err		= res['amplitude'].stderr
		
	#DOUBLET states
	#double Gaussian
	elif spec_feat == 'OIII]':
		spec.mask_region(6485.,6580.)
		cont 	= spec.poly_spec(1)
		spec.unmask()

		line 	= spec - cont

		mod 	= lm.Model(dgauss)
	
		g_cen1 = 1660.8*(1.+z)
	 	g_cen2 = 1666.1*(1.+z)
	
		pars 	= Parameters()
		pars.add('g_cen1',value=g_cen1,min=g_cen1-0.5,max=g_cen1+0.5)
		pars.add('g_cen2',value=g_cen2,expr='1.0031912331406552*g_cen1',min=g_cen2-0.5,max=g_cen2+0.5) 

		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang

		pars.add_many( ('amp1',1.e3,True,1.e2,5.e4), ('wid1',10.,True,5.,20.),\
			('amp2',2.e3,True,1.e2,5.e4), ('wid2',10.,True,5.,20.) )
	
		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)
		report 	= fit.fit_report(min_correl=0.1)
		# print report

		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
	
		fwhm 		= (res['wid1'].value,res['wid2'].value)
		fwhm_err  	= (res['wid1'].stderr,res['wid2'].stderr)
	
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)
	
	elif spec_feat == 'CIII]':
		spec.mask_region(7435.,7542.)
		spec.mask_region(8262.,8520.)

		cont = spec.poly_spec(1)

		spec.unmask()

		line = spec - cont
		mod 	= lm.Model(dgauss)
	
		g_cen1 = 1906.7*(1.+z)
	 	g_cen2 = 1908.7*(1.+z)
	
		pars 	= Parameters()
		pars.add('g_cen1',value=g_cen1,min=g_cen1-0.5,max=g_cen1+0.5)
		pars.add('g_cen2',value=g_cen2,expr='1.0010489327109666*g_cen1',min=g_cen2-0.5,max=g_cen2+0.5) 

		pars.add_many(  
			('amp1',2.e5,True,0.,5.e5), ('wid1',10.,True,5.,20.),\
			 ('amp2',2.e5,True,0.,5.e5), ('wid2',10.,True,5.,20.)
			)
		
		wav 	=  line.wave.coord()  		#Ang
		flux 	=  line.data				#1.e-20 erg / s / cm^2 / Ang
		var_flux    = var_spec.data

		n 			= len(var_flux)
		inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ]

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise)	
		report  = fit.fit_report(min_correl=0.1)
		# print report
	
		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
	
		fwhm 		= (res['wid1'].value,res['wid2'].value)
		fwhm_err  	= (res['wid1'].stderr,res['wid2'].stderr)
	
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

	pl.plot(wav,fit.best_fit,'r--')						#plot model
	pl.plot(wav,flux,drawstyle='steps-mid',color='k')	#plot data
	pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')	#fill grey between zero on flux axis and data
	# pl.show()

	# -----------------------------------------------
	# 	  Write fit parameters into text file
	# -----------------------------------------------
	
	filename = "./out/line-fitting/0943 fit.txt"

	if spec_feat 	== 'CII':
		wav_e = 1338.0
		header = np.array( \
			["#wav0          err_wav0        flux_peak      err_flux_peak\
						  FWHM    	      err_FWHM     wav_rest spec_feat"] )

		gauss_fit_params = np.array([wav_o, wav_o_err, amp, amp_err,\
		 fwhm, fwhm_err, wav_e, spec_feat])
		with open(filename,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '  '.join(map(str,gauss_fit_params)) ) 
			f.write('\n')	
	
	elif spec_feat 	== 'HeII':
		wav_e = 1640.4
		wav_o = wav_o_HeII
		gauss_fit_params = np.array([wav_o, wav_o_err_HeII, amp_HeII,\
		 amp_err_HeII, fwhm_HeII, fwhm_err_HeII, wav_e, spec_feat])
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
	
	elif spec_feat 	== 'CII]':
		wav_e = 2326.0
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
	#    	PLOT with velocity and uJy axes
	# -----------------------------------------------
	
	#precise speed of light in km/s
	c   = 299792.458											
	
	vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat in ('HeII','NIV]','CII]','CII'):	
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
	
	# define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)
	
	#convert flux units to Jy
	def flux_Jy(wav,flux):
		f = 3.33564095e4*flux*1.e-20*wav**2
		return f
	
	#define y-tick marks in reasonable units
	#recover flux in 10^-20 erg/s/cm^2 and convert to microJy
	def flux_cgs(wav,flux_Jy):
		f = flux_Jy/(3.33564095e4*wav**2)
		return f*1.e20
	
	#central wavelength depending on singlet or doublet fit
	if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy
		# print maxf
	
	else:
		wav_cent = wav_o[1]
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy	
	
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
		major_yticks = [ flux0, flux1, flux2, flux3 ]
	
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
	# wavlim05 	= offset_vel_to_wav(500.)
	wavlim1 	= offset_vel_to_wav(1000.)
	# wavlim15 	= offset_vel_to_wav(1500.)
	wavlim2 	= offset_vel_to_wav(2000.)
	# wavlim25    = offset_vel_to_wav(2500.)
	wavlim3 	= offset_vel_to_wav(3000.)
	
	# major_ticks = [ wavlim3[0], wavlim25[0], wavlim2[0], wavlim15[0], wavlim1[0], wavlim05[0], wav0[1],\
	# wavlim05[1], wavlim1[1], wavlim15[1], wavlim2[1], wavlim25[1], wavlim3[1] ]
	major_ticks = [ wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
	 wavlim1[1], wavlim2[1], wavlim3[1] ]
	ax.set_xticks(major_ticks)

	xmin = wavlim3[0] - 2.
	xmax = wavlim3[1] + 2.
	
	#define y-limits
	ymax = 1.2*max(flux)
	ymin = -0.1*max(flux)
	
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
	pl.savefig('./out/line-fitting/'+spec_feat+'_fit.eps')
	# print '----------------------------------------------------------'			