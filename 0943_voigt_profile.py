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

import sys

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

# spec_feat = ['Lya','CIV']
# lam1	  	= [4700.,6000.]
# lam2      = [4835.,6150.]

# spec_feat = ['CIV']
# lam1 		= [6000.]
# lam2 		= [6150.]

spec_feat 	= ['Lya']
lam1 		= [4700.]
lam2 		= [4835.]

# # -----------------------------------------------
# # 	 	FIT to HeII for systemic velocity
# # -----------------------------------------------
# fname 			= "./out/HeII.fits"
# cube 			= sc.SpectralCube.read(fname,hdu=1,formt='fits')
# cube.write(fname, overwrite=True)
# cube 			= mpdo.Cube(fname)
# HeII 			= cube.subcube_circle_aperture(center=(52,46),radius=14,\
# 	unit_center	=None,unit_radius=None)
# spec_HeII 		= HeII.sum(axis=(1,2))
# spec_HeII.plot(color='black')

# ax 			= pl.gca()				
# data 		= ax.lines[0]
# wav 		= data.get_xdata()			
# flux_ax    	= data.get_ydata()	

# lorenz 	= lm.LorentzianModel(prefix='lorenz_')
# pars 	= lorenz.make_params()
# pars['lorenz_center'].set(6433.)
# pars['lorenz_sigma'].set(10.)

# gauss 	= lm.GaussianModel(prefix='gauss_')
# pars.update(gauss.make_params())

# pars['gauss_amplitude'].set(4810.)
# pars['gauss_center'].set(6420.)
# pars['gauss_sigma'].set(10.)

# #composite model
# mod 	= lorenz + gauss

# #model fit
# out 	= mod.fit(flux_ax,pars,x=wav)

# #fit params
# params 			= out.fit_report(min_correl=0.1)
# res 			= out.params

# wav_o_HeII 		= res['lorenz_center'].value
# wav_o_err_HeII	= res['lorenz_center'].stderr
# height_HeII 	= res['lorenz_height'].value
# height_err_HeII = res['lorenz_height'].stderr
# fwhm_HeII 		= res['lorenz_fwhm'].value
# fwhm_err_HeII	= res['lorenz_fwhm'].stderr

# wav_e_HeII 		= 1640.4

# -----------------------------------------------
# 	 			CIV and Ly-alpha
# -----------------------------------------------
for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):
	#--------------------------
	# LOAD spectral data cubes
	#--------------------------
	fname = "./out/"+spec_feat+".fits"
	cube 		= sc.SpectralCube.read(fname,hdu=1,formt='fits')
	cube.write(fname, overwrite=True)
	cube 			= mpdo.Cube(fname)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRAL REGION OF LINE
	# -----------------------------------------------
	#extend to pixel binning
	if spec_feat == 'Lya':
		glx 	= cube.subcube_circle_aperture(center=(50,46),radius=6,\
			unit_center=None,unit_radius=None)
		spec 			= glx.sum(axis=(1,2))

	elif spec_feat == 'CIV':
		glx		= cube.subcube_circle_aperture(center=(52,46),radius=5,\
			unit_center=None,unit_radius=None)
		spec 			= glx.sum(axis=(1,2))

	# pl.figure()
	# img 			= glx.sum(axis=0)
	# ax 				= img.plot( scale='arcsinh' )
	# pl.colorbar(ax,orientation = 'vertical')
	# pl.savefig('./out/line-fitting/'+spec_feat+'_img.png')

	pl.figure()
	spec.plot( color='black' )
	ax 			= pl.gca()				
	data 		= ax.lines[0]
	wav 		= data.get_xdata()			
	flux_ax  	= data.get_ydata()

	#measure continuum flux
	cont_ind = list(enumerate(wav))
	ind = []

	for i in range(len(cont_ind)):
		if (cont_ind[i][1] < 4730. or cont_ind[i][1] > 4810.):
			ind.append(cont_ind[i][0])
	
	N = 0
	f_cont = []
	for i in ind:
		N+=1
		f_cont.append(flux_ax[i])

	f_cont = sum(f_cont)/N

	z 			= 2.923
	c   		= 2.99792458e10		#speed of light: cm/s
	f 			= 0.4162			#HI oscillator strength
	e 			= 4.8e-10			#electron charge: esu
	gamma 		= 6.265e8			#HI
	me 			= 9.1e-28			#electron mass: g	 

	#model functions
	def gauss(x, amp, wid, g_cen):
		gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 /(2*wid**2))
		return gauss

	def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2):
		gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
		gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
		return gauss1 + gauss2

	def rest_freq(x,z):
		return c/(x*1.e-8/(1.+z))		#rest freq in /s

	def voigt_function(a,u):
		z = a + 1j*u
		return np.real(wofz(z))

	def voigt_profile1(x,lam0_1,b1,N1):  		#exp(-tau)
		nu0 	= c/(lam0_1*1.e-8/(1.+z))
		nud     = (nu0/c)*b1
		a 		= gamma / (4.*np.pi*nud)
		u 		= (rest_freq(wav,2.92) - nu0)/nud
		tau 	= N1 * np.sqrt(np.pi) * e**2 * f * voigt_function(a,abs(u)) / nud / me / c
		return np.exp(-tau)

	def voigt_profile2(x,lam0_2,b2,N2):  		#exp(-tau)
		nu0 	= c/(lam0_2*1.e-8/(1.+z))
		nud     = (nu0/c)*b2
		a 		= gamma / (4.*np.pi*nud)
		u 		= (rest_freq(wav,2.92) - nu0)/nud
		tau 	= N2 * np.sqrt(np.pi) * e**2 * f * voigt_function(a,abs(u)) / nud / me / c
		return np.exp(-tau)

	def voigt_profile3(x,lam0_3,b3,N3):  		#exp(-tau)
		nu0 	= c/(lam0_3*1.e-8/(1.+z))
		nud     = (nu0/c)*b3
		a 		= gamma / (4.*np.pi*nud)
		u 		= (rest_freq(wav,2.92) - nu0)/nud
		tau 	= N3 * np.sqrt(np.pi) * e**2 * f * voigt_function(a,abs(u)) / nud / me / c
		return np.exp(-tau)

	def voigt_profile4(x,lam0_4,b4,N4):  		#exp(-tau)
		nu0 	= c/(lam0_4*1.e-8/(1.+z))
		nud     = (nu0/c)*b4
		a 		= gamma / (4.*np.pi*nud)
		u 		= (rest_freq(wav,2.92) - nu0)/nud
		tau 	= N4 * np.sqrt(np.pi) * e**2 * f * voigt_function(a,abs(u)) / nud / me / c
		return np.exp(-tau)

	if spec_feat == 'Lya':
		# pl.figure()
		# pl.plot(wav, gauss(wav, 5.e5, 15., 4769.))
		pl.plot( wav, f_cont*f_cont*voigt_profile1(wav,4752.,84.e5,5.e13) + gauss(wav,5.e5, 15., 4769.) + f_cont*f_cont*voigt_profile2(wav,4768.,64.e5,1.3e15) + f_cont*f_cont*voigt_profile3(wav,4777.,140.e5,3.7e13) + f_cont*f_cont*voigt_profile4(wav,4784.,33.e5,1.7e13) - f_cont*f_cont)
		# pl.plot( wav,f_cont*voigt_profile2(wav,4768.,64.e5,1.3e15))
		# pl.plot( wav,f_cont*voigt_profile3(wav,4777.,140.e5,3.7e13))
		# pl.plot( wav,f_cont*voigt_profile4(wav,4784.,33.e5,1.7e13))	

		# mod 	= lm.Model(gauss) + lm.Model(voigt_profile1) + lm.Model(voigt_profile2) + lm.Model(voigt_profile3) + lm.Model(voigt_profile4) 

		# pars 	= mod.make_params( amp=5.e5, g_cen=4769, wid=15., lam0_1=4752., b1=84.e5, N1=5.e13, lam0_2=4768., b2=64.e5, N2=1.3e15, lam0_3=4777., b3=140.e5, N3=3.7e13, lam0_4=4784., b4=33.e5,N4=1.7e13)

		# # pars 	= mod.make_params( lam0_1=4749., b1=84.e5, N1=5.e13, lam0_2=4764., b2=64.e5, N2=1.e19, lam0_3=4774., b3=140.e5, N3=3.7e13 )

		# # # # # pars 	= mod.make_params(amp=5.e5, g_cen=4769, wid=15., v_cen1=4749., lam0_1=4749., b1=88.e3, N1=1.e9, lam0_2=4764., b2=58.e3, N2=1.e15, lam0_3=4774., b3=109.e3, N3=1.e10, lam0_4=4780., b4=23.e3, N4=1.e9 )

		# res 	= mod.fit(flux_ax,pars,x=wav)

		# comps 	= res.eval_components(x=wav)

		# pl.plot(wav,comps['gauss'],'b--')
		# pl.plot(wav,comps['voigt_profile1'],'m--')
		# pl.plot(wav,comps['voigt_profile2'],'m--')
		# pl.plot(wav,comps['voigt_profile3'],'m--')
		# pl.plot(wav,comps['voigt_profile4'],'m--')

		# pl.plot(wav,res.best_fit,'r--')

		# print res.fit_report(min_correl=0.1)

		pl.show()

	# 	wav_o 		= res.params['g_cen'].value
	# 	wav_o_err 	= res.params['g_cen'].stderr

	# 	wav_abs1	= res.params['v_cen1'].value
	# 	wav_abs2 	= res.params['v_cen2'].value
	# 	wav_abs3 	= res.params['v_cen3'].value

	# 	# print wav_o1,wav_o2

	# 	pl.fill_between(wav,flux_ax,color='grey',interpolate=True,step='mid')

	# 	ymin = 5.*min(flux_ax)
	# 	ymax = 1.2*max(flux_ax)
	# 	pl.title(spec_feat+' Fit')

	# 	pl.savefig('./out/line-fitting/'+spec_feat+' Fit.eps')		

	# elif spec_feat == 'CIV':
	# 	print 'line: '+spec_feat
	# 	# pl.plot(wav,gauss(wav,3.5e4,12.,6070.))
	# 	# pl.plot(wav,gauss(wav,3.5e4,12.,6080.))
	# 	# pl.plot(wav,voigt1(wav,6067.,0.03,0.02,),c='m')
	# 	# pl.plot(wav,voigt1(wav,6077.,0.03,0.02,),c='m')

	#  	mod 	= lm.Model(dgauss) + (lm.Model(voigt1) + lm.Model(voigt2))

	# 	pars 	= mod.make_params(amp1=3.5e4, g_cen1=6070., wid1=12., \
	# 		amp2=3.5e4, g_cen2=6080., wid2=12.,\
	# 		v_cen1=6067., alpha1=0.03, gamma1=0.02,\
	# 		v_cen2=6077., alpha2=0.03, gamma2=0.02)

	# 	res 	= mod.fit(flux_ax,pars,x=wav)

	# 	comps 	= res.eval_components(x=wav)

	# 	# pl.plot(wav,comps['dgauss'],'b--')
	# 	# pl.plot(wav,comps['voigt1'],'m--')
	# 	# pl.plot(wav,comps['voigt2'],'m--')

	# 	# print res.fit_report(min_correl=0.1)

	# 	wav_o1 		= res.params['g_cen1'].value
	# 	wav_o_err1 	= res.params['g_cen1'].stderr	

	# 	wav_o2 		= res.params['g_cen2'].value
	# 	wav_o_err2 	= res.params['g_cen2'].stderr	

	# 	pl.fill_between(wav,flux_ax,color='grey',interpolate=True,step='mid')
	# 	ymin = -0.8*min(flux_ax)
	# 	ymax = 1.2*max(flux_ax)
	# 	pl.title(spec_feat+' Fit')
	# 	pl.plot(wav,res.best_fit,'r--')
	#  	pl.savefig('./out/line-fitting/'+spec_feat+' Fit.eps')	
	 	
	# # -----------------------------------------------
	# #    PLOT Velocity-Integrated Line Profiles
	# # -----------------------------------------------
	# #precise speed of light in km/s										
	
	# #radial velocity (Doppler)
	# def vel(wav_obs,wav_em,z):
	# 	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	# 	return v
	
	# #systemic velocity and redshift based on HeII line
	# wav_e_HeII = 1640.4
	# vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = observer frame (z=0)
	# z = vel0_rest/c

	# # print "Systemic (HeII) Velocity (Vsys): %f km/s" %vel0_rest
	
	# vel0 = vel(wav_o_HeII,wav_e_HeII,z)
	
	# #residual velocity and velocity offset scale (Ang -> km/s)
	# #singlet state
	# if spec_feat == 'Lya':	
	# 	wav_e = 1215.7
	# 	vel_meas = vel(wav_o,wav_e,z)		#central velocity of detected line
	# 	vel_off = vel_meas - vel0			#residual velocity
	#  	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z) - vel0 ) )
	
	# #doublet state
	# elif spec_feat == 'CIV':
	# 	wav_e1 = 1548.2
	# 	wav_e2 = 1550.8
	# 	vel_meas = [vel(wav_o1,wav_e1,z),vel(wav_o2,wav_e2,z)]	
	# 	vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
	# 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 
	
	# #define x-axis as velocity (km/s)
	# ax.xaxis.set_major_formatter(xticks)
	
	# #convert flux units to Jy
	# def flux_Jy(wav,flux):
	# 	f = 3.33564095e4*flux*1.e-20*wav**2
	# 	return f
	
	# #define y-tick marks in reasonable units
	# #recover flux in 10^-20 erg/s/cm^2 and convert to microJy
	# def flux(wav,flux_Jy):
	# 	f = flux_Jy/(3.33564095e4*wav**2)
	# 	return f*1.e20

	# #central wavelength depending on singlet or doublet fit
	# if spec_feat == 'Lya':
	# 	wav_cent = wav_o 
	# 	maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy

	# elif spec_feat == 'CIV':
	# 	wav_cent = wav_o2
	# 	maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy	
	
	# if maxf > 80. :
	# 	flux0 = flux(wav_cent,0.)
	# 	flux1 = flux(wav_cent,20.e-6)
	# 	flux2 = flux(wav_cent,40.e-6)
	# 	flux3 = flux(wav_cent,60.e-6)
	# 	flux4 = flux(wav_cent,80.e-6)
	# 	flux5 = flux(wav_cent,100.e-6)
	# 	flux6 = flux(wav_cent,120.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5, flux6 ]
	
	# elif maxf < 30.:
	# 	flux0 = flux(wav_cent,0.)
	# 	flux1 = flux(wav_cent,5.e-6)
	# 	flux2 = flux(wav_cent,10.e-6)
	# 	flux3 = flux(wav_cent,15.e-6)
	# 	flux4 = flux(wav_cent,20.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]
	
	# ax.set_yticks(major_yticks)
	
	# #define y-axis as flux in Jy
	# if spec_feat in 'Lya':
	# 	yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6 ) )
	# 	ax.yaxis.set_major_formatter(yticks)
		
	# elif spec_feat == 'CIV':
	# 	yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
	# 	ax.yaxis.set_major_formatter(yticks)
	
	# #get wavelengths corresponding to offset velocities
	# def offset_vel_to_wav(voff):
	# 	v1 = -voff
	# 	v2 = voff
	# 	if spec_feat =='Lya':
	# 		wav1 = wav_e*(1.+z)*(1.+(v1/c))
	# 		wav2 = wav_e*(1.+z)*(1.+(v2/c))
	# 		return wav1,wav2

	# 	elif spec_feat == 'CIV':
	# 		wav1 = wav_e2*(1.+z)*(1.+(v1/c))
	# 		wav2 = wav_e2*(1.+z)*(1.+(v2/c))
	# 		return wav1,wav2
	
	# #define x-axis as in offset velocity units (km/s)
	# wav0 = offset_vel_to_wav(0.)
	# wavlim05 = offset_vel_to_wav(500.)
	# wavlim1 = offset_vel_to_wav(1000.)
	# wavlim15 = offset_vel_to_wav(1500.)
	# wavlim2 = offset_vel_to_wav(2000.)
	
	# major_ticks = [ wavlim2[0], wavlim15[0], wavlim1[0], wavlim05[0], wav0[1],\
	#  wavlim05[1], wavlim1[1], wavlim15[1], wavlim2[1] ]
	# ax.set_xticks(major_ticks)
	
	# #fontsize of ticks
	# for tick in ax.xaxis.get_major_ticks():
	#     tick.label.set_fontsize(10)
	
	# #define x-limits
	# xmin = wavlim2[0] - 2.
	# xmax = wavlim2[1] + 2.
	
	# #draw line representing central velocity of spectral feature
	# if spec_feat == 'Lya':
	# 	pl.plot([wav_o,wav_o],[ymin,ymax],color='green',ls='--')

	# elif spec_feat == 'CIV':
	# 	pl.plot([wav_o1,wav_o1],[ymin,ymax],color='green',ls='--')
	# 	pl.plot([wav_o2,wav_o2],[ymin,ymax],color='green',ls='--')
	
	# #draw plot
	# pl.title(spec_feat+' Fit')
	# ax.set_xlabel( 'Velocity Offset (km/s)' )
	# ax.set_ylabel( r'Flux Density ($\mu$Jy)' )
	# # ax.set_ylabel(r'Flux Density (10$^{-20}$ erg / s / cm$^{2}$ / $\AA$)')
	# ax.set_ylim([ymin,ymax])
	# ax.set_xlim([xmin,xmax])
	# pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis
	# pl.savefig('out/line-fitting/'+spec_feat+' Fit.eps')