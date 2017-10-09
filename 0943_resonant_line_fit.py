#S.N. Kolwa (2017)
#0943_abs_line_fit.py
# Purpose:  
# - Resonant line profile fit i.e. Gaussian/Lorentzian and Voigt
# - Flux in Jy 
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)

import matplotlib.pyplot as pl
import numpy as np 

import spectral_cube as sc
import mpdaf.obj as mpdo

import warnings
from astropy.utils.exceptions import AstropyWarning

import astropy.units as u
import matplotlib.ticker as tk

import lmfit.models as lm
from lmfit import CompositeModel
 
from scipy.special import wofz

import PyAstronomy.modelSuite as ms

import scipy as sp

import sys

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat = ['Lya','CIV']
lam1	  = [4700.,6000.]
lam2      = [4835.,6150.]

# -----------------------------------------------
# 	 	FIT to HeII for systemic velocity
# -----------------------------------------------
fname 			= "./out/HeII.fits"
cube 			= sc.SpectralCube.read(fname,hdu=1,formt='fits')
cube.write(fname, overwrite=True)
cube 			= mpdo.Cube(fname)
HeII 			= cube.subcube_circle_aperture(center=(52,46),radius=14,\
	unit_center	=None,unit_radius=None)
spec_HeII 		= HeII.sum(axis=(1,2))
spec_HeII.plot(color='black')

ax 			= pl.gca()				
data 		= ax.lines[0]
wav 		= data.get_xdata()			
flux_ax    	= data.get_ydata()	

lorenz 	= lm.LorentzianModel(prefix='lorenz_')
pars 	= lorenz.make_params()
pars['lorenz_center'].set(6433.)
pars['lorenz_sigma'].set(10.)

gauss 	= lm.GaussianModel(prefix='gauss_')
pars.update(gauss.make_params())

pars['gauss_amplitude'].set(4810.)
pars['gauss_center'].set(6420.)
pars['gauss_sigma'].set(10.)

#composite model
mod 	= lorenz + gauss

#model fit
out 	= mod.fit(flux_ax,pars,x=wav)

#fit params
params 			= out.fit_report(min_correl=0.1)
res 			= out.params

wav_o_HeII 		= res['lorenz_center'].value
wav_o_err_HeII	= res['lorenz_center'].stderr
height_HeII 	= res['lorenz_height'].value
height_err_HeII = res['lorenz_height'].stderr
fwhm_HeII 		= res['lorenz_fwhm'].value
fwhm_err_HeII	= res['lorenz_fwhm'].stderr

wav_e_HeII 		= 1640.4

# -----------------------------------------------
# 	 			CIV and Ly-alpha
# -----------------------------------------------
for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):
	# --------------------------
	# LOAD spectral data cubes
	# --------------------------
	fname = "./out/"+spec_feat+"_cs.fits"
	cube 		= sc.SpectralCube.read(fname,hdu=1,formt='fits')
	cube.write(fname, overwrite=True)
	cube 			= mpdo.Cube(fname)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRAL REGION OF LINE
	# -----------------------------------------------
	#extend this to pixel binning (tesselation map)
	glx		= cube.subcube_circle_aperture(center=(50,46),radius=6,\
			unit_center=None,unit_radius=None)
	spec 			= glx.sum(axis=(1,2))

	pl.figure()
	img 			= glx.sum(axis=0)
	ax 				= img.plot( scale='arcsinh' )
	pl.colorbar(ax,orientation = 'vertical')
	pl.savefig('./out/line-fitting/'+spec_feat+'_img.png')

	pl.figure()
	spec.plot( color='black' )
	ax 			= pl.gca()				
	data 		= ax.lines[0]
	wav 		= data.get_xdata()			
	flux_ax  	= data.get_ydata()

	c   		= 2.99792458e10			#speed of light: cm/s
	e 			= 4.8e-10				#electron charge: esu
	me 			= 9.1e-28				#electron mass: g

	#gaussian models
	def gauss(x, amp, wid, g_cen):
		gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 /(2*wid**2))
		return gauss

	def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2):
		gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
		gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
		return gauss1 + gauss2

# Voigt-Hjerting function, as approximate by Tepper-Garcia 2006:
	def H(a, x):
	    """ The H(a, u) function of Tepper Garcia 2006
	    """
	    P = x ** 2
	    H0 = sp.e ** (-(x ** 2))
	    Q = 1.5 / x ** 2
	    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
	    return H

	if spec_feat == 'Lya':
		f 			= 0.4162				#HI oscillator strength
		gamma 		= 6.265e8				#gamma of HI line
		nu0			= c / (1215.57*1.e-8) 	#Hz 

		def voigt_profile1(x,z1,f0_1,b1,N1):  			
			nu 		= c / ( (x / (1.+z1)) * 1.e-8 )	
			nud     = ( nu0 / c ) * b1
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return f0_1*(np.exp( -tau ) - 1.)

		def voigt_profile2(x,z2,f0_2,b2,N2):  			
			nu 		= c / ( (x / (1.+z2)) * 1.e-8 )	
			nud     = ( nu0 / c ) * b2
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return f0_2*(np.exp( -tau ) - 1.)

		def voigt_profile3(x,z3,f0_3,b3,N3):  			
			nu 		= c / ( (x / (1.+z3)) * 1.e-8 )	
			nud     = ( nu0 / c ) * b3
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return f0_3*(np.exp( -tau ) - 1.)

		def voigt_profile4(x,z4,f0_4,b4,N4):  			
			nu 		= c / ( (x / (1.+z4)) * 1.e-8 )	
			nud     = ( nu0 / c ) * b4
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return f0_4*(np.exp( -tau ) - 1.)

		# pl.plot( wav,gauss(wav, 5.e5, 15., 4769.),c='b' )
		# pl.plot( wav,voigt_profile1(wav,2.905,4.e3,84.e5,5.e13),c='r' )
		# pl.plot( wav,voigt_profile2(wav,2.918,11.e3,40.e5,10**19.5),c='r' )
		# pl.plot( wav,voigt_profile3(wav,2.928,12.e3,140.e5,3.7e13),c='r' )
		# pl.plot( wav,voigt_profile4(wav,2.933,7.e3,33.e5,1.7e13),c='r')

		mod 	= lm.Model(dgauss) \
		 + lm.Model(voigt_profile1) + lm.Model(voigt_profile2) + \
		 lm.Model(voigt_profile3) + lm.Model(voigt_profile4)

		pars 	= mod.make_params( amp1=2.e3, wid1=35., g_cen1=4772.,\
			amp2=6.e5, wid2=20., g_cen2=4765.,\
		 z1=2.905, f0_1=4.e3, b1=86.e5, N1=5.e13,\
		 z2=2.919, f0_2=12.e3, b2=65.e5, N2=10**19.5, \
		 z3=2.928, f0_3=5.e3, b3=100.e5, N3=3.7e13, 
		 z4=2.934, f0_4=10.e3, b4=35.e5, N4=2.e13 )

		# test_fit = mod.eval(pars,x=wav)

		# pl.plot(wav,test_fit,'r--')

		fit 	= mod.fit(flux_ax,pars,x=wav,fit_kws={'nan_policy':'omit'})

		comps 	= fit.eval_components(x=wav)

		# pl.plot(wav,comps['dgauss'],'b--')
		# pl.plot(wav,comps['voigt_profile1'],'m--')
		# pl.plot(wav,comps['voigt_profile2'],'m--')
		# pl.plot(wav,comps['voigt_profile3'],'m--')
		# pl.plot(wav,comps['voigt_profile4'],'m--')

		print fit.fit_report()

		wav_o 		= fit.params['g_cen2'].value
		wav_o_err 	= fit.params['g_cen2'].stderr

		pl.fill_between(wav,flux_ax,color='grey',interpolate=True,step='mid')

		pl.title(spec_feat+' Fit')
		pl.plot(wav,fit.best_fit,'r--')
		# pl.savefig('./out/line-fitting/'+spec_feat+' component Fit.eps')	

	elif spec_feat == 'CIV':
		f1 			= 0.097
		f2 			= 0.194
		gamma1 		= 2.70e8				#gamma of CIV line
		gamma2  	= 2.69e8
		nu01		= c / (1550.774*1.e-8)	#rest-frequency: Hz 
		nu02		= c / (1548.202*1.e-8) 

		def voigt_profile1(x,z1,f0_1,b1,N1):  			
			nu 		= c / ( (x / (1.+z1)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b1
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return f0_1*(np.exp( -tau ) - 1.)

		def voigt_profile2(x,z2,f0_2,b2,N2):  			
			nu 		= c / ( (x / (1.+z2)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b2
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return f0_2*(np.exp( -tau ) - 1.)

		# pl.plot(wav,gauss(wav,2e4,12.,6070.))
		# pl.plot(wav,gauss(wav,2e4,12.,6080.))
		# pl.plot(wav,voigt_profile1(wav,2.913,600.,95.e5,3.e14),c='m')
		# pl.plot(wav,voigt_profile2(wav,2.926,1200.,100.e5,7.e14),c='m')

		mod 	= lm.Model(dgauss) + lm.Model(voigt_profile1) + lm.Model(voigt_profile2)

		pars 	= mod.make_params(amp1=1.4e4, wid1=12., g_cen1=6070., \
		amp2=2.8e4, wid2=12., g_cen2=6080.,\
		z1=2.913,f0_1=600.,b1=95.e5,N1=6.e15,\
		z2=2.926,f0_2=1200.,b2=100.e5,N2=2.e14)

		# test_fit = mod.eval(pars,x=wav)

		# pl.plot(wav,test_fit,'r--')

		fit 	= mod.fit(flux_ax,pars,x=wav)

		comps 	= fit.eval_components(x=wav)

		# pl.plot(wav,comps['dgauss'],'b--')
		# pl.plot(wav,comps['voigt_profile1'],'m--')
		# pl.plot(wav,comps['voigt_profile2'],'m--')

		# print fit.fit_report()

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

		pl.fill_between(wav,flux_ax,color='grey',interpolate=True,step='mid')

		pl.title(spec_feat+' Fit')
		pl.plot(wav,fit.best_fit,'r--')
	  	# pl.savefig('./out/line-fitting/'+spec_feat+' component Fit.eps')
	 	
	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------							
	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v
	
	#systemic velocity and redshift based on HeII line
	c = 2.9979245800e5
	wav_e_HeII = 1640.4
	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = observer frame (z=0)
	z = vel0_rest/c

	# print "Systemic (HeII) Velocity (Vsys): %f km/s" %vel0_rest
	
	vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat == 'Lya':	
		wav_e = 1215.7
		vel_meas = vel(wav_o,wav_e,z)		#central velocity of detected line
		vel_off = vel_meas - vel0			#residual velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z) - vel0 ) )
	
	#doublet state
	elif spec_feat == 'CIV':
		wav_e1 = 1548.2
		wav_e2 = 1550.8
		vel_meas = [vel(wav_o1,wav_e1,z),vel(wav_o2,wav_e2,z)]	
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

	#central wavelength (on doublet we select longest rest wavelength)
	if spec_feat == 'Lya':
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy

	elif spec_feat == 'CIV':
		wav_cent = wav_o2
		maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy	
	
	if maxf > 50. :
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,20.e-6)
		flux2 = flux(wav_cent,40.e-6)
		flux3 = flux(wav_cent,60.e-6)
		flux4 = flux(wav_cent,80.e-6)
		flux5 = flux(wav_cent,100.e-6)
		flux6 = flux(wav_cent,120.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5, flux6 ]
	
	elif maxf < 50.:
		flux0 = flux(wav_cent,0.)
		flux1 = flux(wav_cent,5.e-6)
		flux2 = flux(wav_cent,10.e-6)
		flux3 = flux(wav_cent,15.e-6)
		flux4 = flux(wav_cent,20.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	ax.set_yticks(major_yticks)
	
	#define y-axis as flux in Jy
	if spec_feat in 'Lya':
		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6 ) )
		ax.yaxis.set_major_formatter(yticks)
		ymin = flux(wav_cent,-0.5e-6)
		ymax = flux(wav_cent,120.e-6)
		
	elif spec_feat == 'CIV':
		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)
		ymin = flux(wav_cent,-0.5e-6)
		ymax = flux(wav_cent,20.e-6)
	
	#get wavelengths corresponding to offset velocities
	def offset_vel_to_wav(voff):
		v1 = -voff
		v2 = voff
		if spec_feat =='Lya':
			wav1 = wav_e*(1.+z)*(1.+(v1/c))
			wav2 = wav_e*(1.+z)*(1.+(v2/c))
			return wav1,wav2

		elif spec_feat == 'CIV':
			wav1 = wav_e2*(1.+z)*(1.+(v1/c))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c))
			return wav1,wav2
	
	#define x-axis as in offset velocity units (km/s)
	wav0 = offset_vel_to_wav(0.)
	wavlim1 = offset_vel_to_wav(1000.)
	wavlim2 = offset_vel_to_wav(2000.)
	
	major_ticks = [ wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1] ]
	ax.set_xticks(major_ticks)
	
	#fontsize of ticks
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(10)
	
	#define x-limits
	xmin = wavlim2[0] - 2.
	xmax = wavlim2[1] + 2.
	
	#draw line representing central velocity of spectral feature
	if spec_feat == 'Lya':
		pl.plot([wav_o,wav_o],[ymin,ymax],color='green',ls='--')

	elif spec_feat == 'CIV':
		pl.plot([wav_o1,wav_o1],[ymin,ymax],color='green',ls='--')
		pl.plot([wav_o2,wav_o2],[ymin,ymax],color='green',ls='--')
	
	#draw plot
	pl.title(spec_feat+' Fit')
	ax.set_xlabel( 'Velocity Offset (km/s)' )
	ax.set_ylabel( r'Flux Density ($\mu$Jy)' )
	ax.set_ylim([ymin,ymax])
	ax.set_xlim([xmin,xmax])
	pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis
	pl.savefig('out/line-fitting/'+spec_feat+' Fit.eps')