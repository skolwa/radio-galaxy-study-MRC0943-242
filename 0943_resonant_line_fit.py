#S.N. Kolwa (2017)
#0943_abs_line_fit.py
# Purpose:  
# - Resonant line profile fit i.e. Gaussian/Lorentzian and Voigt
# - Flux in Jy 
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)

import matplotlib.pyplot as pl
import numpy as np 


import mpdaf.obj as mpdo
from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning

import astropy.units as u
import matplotlib.ticker as tk

import lmfit.models as lm
from lmfit import CompositeModel
from lmfit import Parameters
 
from scipy.special import wofz

import scipy as sp
import sys

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat = ['Lya','CIV','NV','SiIV']

# spec_feat = ['NV']

#pixel radius for extraction
radius = 4
center = (84,108)

# -----------------------------------------------
# 	 	FIT to HeII for systemic velocity
# -----------------------------------------------
fname 		= "./out/HeII.fits"
hdu 		= fits.open(fname)
hdr 	 	= mpdo.Cube(fname,ext=0)
datacube 	= mpdo.Cube(fname,ext=1)
varcube  	= mpdo.Cube(fname,ext=2)

HeII 		= datacube.subcube_circle_aperture(center=center,radius=radius,\
	unit_center	=None,unit_radius=None)
spec_HeII 	= HeII.sum(axis=(1,2))

#continuum subtract the spectrum
spec_HeII.mask_region(6366.,6570.)
cont = spec_HeII.poly_spec(1)
spec_HeII.unmask()

line_HeII 	= spec_HeII - cont
wav 		=  line_HeII.wave.coord()  		#1.e-8 cm
flux 		=  line_HeII.data				#1.e-20 erg / s / cm^2 / Ang

lorenz 	= lm.LorentzianModel(prefix='lorenz_')
pars 	= lorenz.make_params()
pars.add('lorenz_center',value=6433.,min=6432.,max=6434.)
pars.add('lorenz_sigma',value=10.,min=5.,max=15.)

# gauss 	= lm.GaussianModel(prefix='gauss_')
# pars.update(gauss.make_params())

# pars['gauss_center'].set(6420.)
# pars['gauss_sigma'].set(10.)

#composite model
mod 	= lorenz 

#model fit
fit 	= mod.fit(flux,pars,x=wav)

# print fit.fit_report()

res 	= fit.params

wav_o_HeII 		= res['lorenz_center'].value
wav_o_err_HeII	= res['lorenz_center'].stderr

height_HeII 	= res['lorenz_height'].value
height_err_HeII = res['lorenz_height'].stderr

fwhm_HeII 		= res['lorenz_fwhm'].value
fwhm_err_HeII	= res['lorenz_fwhm'].stderr

wav_e_HeII 		= 1640.4

# -----------------------------------------------
# 	      resonant lines: CIV and Lya
# -----------------------------------------------

for spec_feat in spec_feat:
	# print '-----'
	# print spec_feat
	# print '-----'
	# --------------------------
	#     LOAD data cubes
	# --------------------------

	fname = "./out/"+spec_feat+".fits"
	datacube 			= mpdo.Cube(fname,ext=1)
	varcube				= mpdo.Cube(fname,ext=2)
	
	# -----------------------------------------------
	# 		EXTRACT SPECTRAL REGION OF LINE
	# -----------------------------------------------

	glx		= datacube.subcube_circle_aperture(center=center,radius=radius,\
			unit_center=None,unit_radius=None)
	spec 	= glx.sum(axis=(1,2))

	var_glx = varcube.subcube_circle_aperture(center=center,radius=radius,\
			unit_center=None,unit_radius=None)
	var_spec = var_glx.sum(axis=(1,2))

	fig,ax = pl.subplots(1,1)
	var_flux    = var_spec.data

	n 			= len(var_flux)

	inv_noise 	= [ 1./np.sqrt(var_flux[i]) for i in range(n) ] 

	#gaussian models
	def gauss(x, amp, wid, g_cen):
		gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 /(2*wid**2))
		return gauss

	def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2):
		gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
		gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
		return gauss1 + gauss2

	# Voigt-Hjerting function
	def H(a, x):
	    """ The H(a, u) function of Tepper Garcia 2006
	    """
	    P = x ** 2
	    H0 = sp.e ** (-(x ** 2))
	    Q = 1.5 / x ** 2
	    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
	    return H

	#radial velocity (Doppler)
	def vel(wav_obs,wav_em,z):
		v = c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v

	c = 2.9979245800e10

	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = obs frame (z=0)
	z = vel0_rest/c

	#filename into which fit parameters are written
	filename = "./out/line-fitting/0943_resonant_fit.txt"

	e 			= 4.8e-10				#electron charge: esu
	me 			= 9.1e-28				#electron mass: g

	if spec_feat == 'Lya':
		spec.mask_region(4706.,4906.)
		cont = spec.poly_spec(1)
		spec.unmask()

		line = spec - cont

		wav 		= line.wave.coord()		
		flux  		= line.data

		f 			= 0.4162				#HI oscillator strength
		gamma 		= 6.265e8				#gamma of HI line
		lam			= 1215.57				# rest wavelength
		nu0			= c / (lam*1.e-8) 		#Hz 

		def voigt_profile1(x,z1,b1,N1):  			
			nu 		= c / ( (x / (1.+z1) ) * 1.e-8 )	
			nud     = ( nu0 / c ) * b1
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile2(x,z2,b2,N2):  			
			nu 		= c / ( (x / (1.+z2) ) * 1.e-8 )	
			nud     = ( nu0 / c ) * b2
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile3(x,z3,b3,N3):  			
			nu 		= c / ( (x / (1.+z3) ) * 1.e-8 )	
			nud     = ( nu0 / c ) * b3
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile4(x,z4,b4,N4):  			
			nu 		= c / ( (x / (1.+z4) ) * 1.e-8 )	
			nud     = ( nu0 / c ) * b4
			a 		= gamma / ( 4. * np.pi * nud )
			u 		= ( nu - nu0 ) / nud
			tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
			return np.exp( -tau )

		mod 	= lm.Model(gauss)*lm.Model(voigt_profile1)*lm.Model(voigt_profile2)*lm.Model(voigt_profile3)*lm.Model(voigt_profile4)

		g_cen = lam*(1.+z)

		pars = Parameters()

		# fn = gauss(wav,35.*max(flux),10.,g_cen)*voigt_profile1(wav,2.906,100.e5,2.e14)*voigt_profile2(wav,2.919,60.e5,1.e19)*voigt_profile3(wav,2.926,80.e5,5.e13)*voigt_profile4(wav,2.934,35.e5,2.e13) 

		# pl.plot(wav,fn)

		pars.add_many( ('amp' , 35*max(flux), True, 10*max(flux),100*max(flux) ), ('wid' , 15., True, 10., 30.), ('g_cen' , g_cen, True, g_cen-0.5, g_cen+0.5),\
		 ('z1', 2.907, True, 2.906, 2.908), ('b1', 86.e5, True, 1.e5, 200.e5), ('N1', 1.e14, True, 1.e12, 1.e20),\
		 ('z2', 2.919, True, 2.918, 2.92), ('b2', 65.e5, True, 1.e5, 200.e5), ('N2', 1.e19, True, 1.e12, 1.e20), \
		 ('z3', 2.926, True, 2.925, 2.927), ('b3', 100.e5, True, 1.e5, 200.e5), ('N3', 2.e13, True, 1.e12, 1.e20), 
		 ('z4', 2.934, True, 2.933, 2.935), ('b4', 35.e5, True, 1.e5, 200.e5), ('N4', 2.e12, True, 1.e12, 1.e20) )

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		comps 	= fit.eval_components(x=wav)

		pl.plot(wav,comps['gauss'],'b--')

		print fit.fit_report()

		wav_o 		= fit.params['g_cen'].value
		wav_o_err 	= fit.params['g_cen'].stderr

		pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
		pl.title(spec_feat+' Fit')
		pl.plot(wav,fit.best_fit,'r--')
		pl.plot(wav,flux,drawstyle='steps-mid',color='k')
		pl.show()
		
	elif spec_feat == 'CIV':
		spec.mask_region(6005.,6150.)
		cont 	= spec.poly_spec(1)
		spec.unmask()

		line = spec - cont

		wav 		= line.wave.coord()		
		flux  		= line.data

		f1 			= 0.190
		f2 			= 0.0948
		gamma1 		= 2.69e8				#gamma of CIV line
		gamma2  	= 2.70e8
		lam1		= 1548.202 
		lam2		= 1550.774
		nu01		= c / (lam1*1.e-8)		#rest-frequency: Hz 
		nu02		= c / (lam2*1.e-8) 

		def voigt_profile1(x,z1,b1,N1):  			
			nu 		= c / ( (x / (1.+z1)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b1
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile2(x,z2,b2,N2):  			
			nu 		= c / ( (x / (1.+z2)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b2
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile3(x,z3,b3,N3):  			
			nu 		= c / ( (x / (1.+z3)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b3
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile4(x,z4,b4,N4):  			
			nu 		= c / ( (x / (1.+z4)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b4
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return np.exp( -tau )

		mod 	= lm.Model(dgauss)*lm.Model(voigt_profile1)*lm.Model(voigt_profile2)*lm.Model(voigt_profile3)*lm.Model(voigt_profile4)

		g_cen1 = lam1*(1.+z)
		g_cen2 = lam2*(1.+z)

		pars = Parameters()

		# fn = dgauss(wav,1.2e4,12.,g_cen1,3.2e4,12.,g_cen2) + voigt_profile1(wav,2.920,2.e3,100.e5,1.e14)\
		#  + voigt_profile2(wav,2.920,2.e3,100.e5,1.e14) + voigt_profile3(wav,2.906,2.e3,100.e5,1.e13) + voigt_profile4(wav,2.906,2.e3,100.e5,1.e13)

		# pl.plot(wav,fn)

		pars.add_many( ('amp1', 60.*max(flux), True, 20.*max(flux), 200.*max(flux)), ('wid1', 15., True, 10., 20.),
		('amp2', 40.*max(flux), True, 20.*max(flux), 200.*max(flux)), ('wid2', 15., True, 10.,20.) )

		pars.add('g_cen1',g_cen1,True,g_cen1-0.5,g_cen1+0.5)
		pars.add('g_cen2',g_cen2,True,expr='1.0016612819257436*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add('z1',2.919,True,2.918,2.92)
		pars.add('z2',True,expr='z1')

		pars.add('z3',2.907,True,2.906,2.908)
		pars.add('z4',True,expr='z3')

		pars.add('N1',1.e13,True,1.e10,1.e22)
		pars.add('N2',True,expr='N1')

		pars.add('b1',100.e5,True,20.e5,400.e5)
		pars.add('b2',True,expr='b1')

		pars.add('N3',1.e12,True,1.e10,1.e22)
		pars.add('N4',True,expr='N3')

		pars.add('b3',100.e5,True,20.e5,400.e5)
		pars.add('b4',True,expr='b3')

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		comps 	= fit.eval_components(x=wav)

		pl.plot(wav,comps['dgauss'],'b--')
		pl.plot(wav,comps['voigt_profile1'],'m--')
		pl.plot(wav,comps['voigt_profile2'],'m--')
		pl.plot(wav,comps['voigt_profile3'],'m--')
		pl.plot(wav,comps['voigt_profile4'],'m--')

		print fit.fit_report()

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

		pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
		pl.plot(wav,fit.best_fit,'r--')
		pl.plot(wav,flux,drawstyle='steps-mid',color='k')
		pl.show()

	elif spec_feat == 'NV':
		spec.mask_region(4840.,4908.)
		cont 	= spec.poly_spec(1)
		spec.unmask()

		line = spec - cont

		wav 		= line.wave.coord()		
		flux  		= line.data

		f1 			= 0.0777
		f2 			= 0.156
		gamma1 		= 3.37e8			#gamma of NV line
		gamma2  	= 3.40e8
		lam1		= 1238.8
		lam2		= 1242.8
		nu01		= c / (lam1*1.e-8)	#rest-frequency: Hz 
		nu02		= c / (lam2*1.e-8) 

		def voigt_profile1(x,z1,b1,N1):  			
			nu 		= c / ( (x / (1.+z1)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b1
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile2(x,z2,b2,N2):  			
			nu 		= c / ( (x / (1.+z2)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b2
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile3(x,z3,b3,N3):  			
			nu 		= c / ( (x / (1.+z3)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b3
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return np.exp( -tau )

		def voigt_profile4(x,z4,b4,N4):  			
			nu 		= c / ( (x / (1.+z4)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b4
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return np.exp( -tau )

		mod 	= lm.Model(dgauss)*lm.Model(voigt_profile1)*lm.Model(voigt_profile2)*lm.Model(voigt_profile3)*lm.Model(voigt_profile4)

		g_cen1 = lam1*(1.+z)
	 	g_cen2 = lam2*(1.+z)

	 	pars = Parameters()

		pars.add('g_cen1',value=g_cen1,min=g_cen1-0.5,max=g_cen1+0.5)
		pars.add('g_cen2',value=g_cen2,expr='1.003228931223765*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add_many( 
			('amp1', 16.e3, True, 2.e3, 100.e3),('wid1', 10.,True, 2., 20.),
			('amp2', 16.e3, True, 2.e3, 100.e3),('wid2', 10.,True, 2., 20.) )

		pars.add('z1',2.916,True,2.915,2.917)
		pars.add('z2',True,expr='z1')

		pars.add('z3',2.906,True,2.905,2.907 )
		pars.add('z4',True,expr='z3')

		pars.add('N1',1.e13,True,1.e12,1.e16)
		pars.add('N2',True,expr='N1')

		pars.add('b1',40.e5,True,50.e5,200.e5)
		pars.add('b2',True,expr='b1')

		pars.add('N3',1.e13,True,1.e12,1.e16)
		pars.add('N4',True,expr='N3')

		pars.add('b3',30.e5,True,50.e5,200.e5)
		pars.add('b4',True,expr='b3')

		fn = dgauss(wav,1.e3,4.,g_cen1,1.e3,8.,g_cen2)*voigt_profile1(wav, 2.917, 40.e5, 1.e13)*voigt_profile2(wav, 2.917, 40.e5, 1.e13)*voigt_profile3(wav, 2.906, 30.e5, 1.e13)*voigt_profile4(wav, 2.906, 30.e5, 1.e13)
		pl.plot(wav,fn)

		# fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		# print fit.fit_report()

		# comps 	= fit.eval_components(x=wav)

		# pl.plot(wav,comps['dgauss'], 'b--')

		# wav_o1 		= fit.params['g_cen1'].value
		# wav_o_err1 	= fit.params['g_cen1'].stderr	

		# wav_o2 		= fit.params['g_cen2'].value
		# wav_o_err2 	= fit.params['g_cen2'].stderr	

		# pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
		# pl.plot(wav,fit.best_fit,'r--')
		pl.plot(wav,flux,drawstyle='steps-mid',color='k')
		pl.show()

	elif spec_feat == 'SiIV':
		spec.mask_region(5452.,5588.)
		cont 	= spec.poly_spec(1)
		spec.unmask()

		line = spec - cont

		wav 		= line.wave.coord()		
		flux  		= line.data

		f1 			= 0.513
		f2 			= 0.254
		gamma1 		= 8.80e8			#gamma of SiIV line
		gamma2  	= 8.63e8
		lam1		= 1393.76
		lam2		= 1402.77
		nu01		= c / (lam1*1.e-8)	#rest-frequency: Hz 
		nu02		= c / (lam2*1.e-8) 

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

		def voigt_profile3(x,z3,f0_3,b3,N3):  			
			nu 		= c / ( (x / (1.+z3)) * 1.e-8 )	
			nud     = ( nu01 / c ) * b3
			a 		= gamma1 / ( 4. * np.pi * nud )
			u 		= ( nu - nu01 ) / nud
			tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
			return f0_3*(np.exp( -tau ) - 1.)

		def voigt_profile4(x,z4,f0_4,b4,N4):  			
			nu 		= c / ( (x / (1.+z4)) * 1.e-8 )	
			nud     = ( nu02 / c ) * b4
			a 		= gamma2 / ( 4. * np.pi * nud )
			u 		= ( nu - nu02 ) / nud
			tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
			return f0_4*(np.exp( -tau ) - 1.)

		mod 	= lm.Model(dgauss) + lm.Model(voigt_profile1) + lm.Model(voigt_profile2)\
		 + lm.Model(voigt_profile3) + lm.Model(voigt_profile4)

		g_cen1 = lam1*(1.+z)
	 	g_cen2 = lam2*(1.+z)

	 	pars = Parameters()

		pars.add('g_cen1',value=g_cen1,min=g_cen1-0.5,max=g_cen1+0.5)
		pars.add('g_cen2',value=g_cen2,expr='1.0064645276087705*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add_many( 
			('amp1', 1.e3, False, 2.e3, 4.e3),('wid1', 12.,True, 2., 20.),
			('amp2', 5.5e3, False, 2.e3, 8.e3),('wid2', 12.,True, 2., 20.),
			('f0_1', 4.e3, True, 200.,1.e5),('f0_2', 4.e3, True, 200.,1.e5),
			('f0_3', 2.e3, True, 200.,1.e5),('f0_4', 2.e3, True, 200.,1.e5))

		pars.add('z1',True,2.920,2.919,2.921)
		pars.add('z2',True,expr='z1')

		pars.add('z3',True,2.912,2.911,2.913)
		pars.add('z4',True,expr='z3')

		pars.add('N1',1.e12,True,1.e10,1.e14)
		pars.add('N2',True,expr='N1')

		pars.add('b1',100.e5,True,50.e5,200.e5)
		pars.add('b2',True,expr='b1')

		pars.add('N3',2.e12,True,1.e10,1.e14)
		pars.add('N4',True,expr='N3')

		pars.add('b3',30.e5,True,50.e5,200.e5)
		pars.add('b4',True,expr='b3')

		# fn = dgauss(wav,1.e3,8.,g_cen1,5.5e3,14.,g_cen2) + \
		# voigt_profile1(wav, 2.920, 4.e3, 100.e5, 2.e12) + \
		# voigt_profile2(wav, 2.920, 4.e3, 100.e5, 2.e12) + \
		# voigt_profile3(wav, 2.911, 2.e3, 50.e5, 2.e12) + \
		# voigt_profile4(wav, 2.911, 2.e3, 50.e5, 2.e12)
		# pl.plot(wav,fn)

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		# print fit.fit_report()

		comps 	= fit.eval_components(x=wav)

		pl.plot(wav,comps['dgauss'], 'b--')
		pl.plot(wav,comps['voigt_profile1'],'m--')
		pl.plot(wav,comps['voigt_profile2'],'m--')
		pl.plot(wav,comps['voigt_profile3'],'m--')
		pl.plot(wav,comps['voigt_profile4'],'m--')

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

		pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
		pl.plot(wav,fit.best_fit,'r--')
		pl.plot(wav,flux,drawstyle='steps-mid',color='k')

	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------						
	
	#systemic velocity and redshift based on HeII line
	c = 2.9979245800e5
	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = observer frame (z=0)
	z = vel0_rest/c

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

	elif spec_feat == 'NV':
		wav_e1		= 1238.8
		wav_e2		= 1242.8
		vel_meas = [vel(wav_o1,wav_e1,z),vel(wav_o2,wav_e2,z)]	
		vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 

	elif spec_feat == 'SiIV':
		wav_e1   = 1393.76
		wav_e2   = 1402.77
		vel_meas = [vel(wav_o1,wav_e1,z),vel(wav_o2,wav_e2,z)]	
		vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 
	
	#define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)		#format of tickmarks (i.e. unit conversion)
	
	#convert erg/s/cm^2/Ang to Jy
	def flux_Jy(wav,flux):
		f = 3.33564095e4*flux*1.e-20*wav**2
		return f
	
	#define y-tick marks in reasonable units
	#recover flux in 10^-20 erg/s/cm^2 and convert to microJy
	def flux_cgs(wav,flux_Jy):
		f = flux_Jy/(3.33564095e4*wav**2)
		return f*1.e20

	#central wavelength (on doublet we select longest rest wavelength)
	if spec_feat == 'Lya':
		wav_cent = wav_o 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy

	else:
		wav_cent = wav_o2
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy	
	
	if maxf < 60. and maxf > 20.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,10.e-6)
		flux2 = flux_cgs(wav_cent,20.e-6)
		flux3 = flux_cgs(wav_cent,30.e-6)
		flux4 = flux_cgs(wav_cent,40.e-6)
		flux5 = flux_cgs(wav_cent,50.e-6)
		flux6 = flux_cgs(wav_cent,60.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5, flux6 ]

		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)

	elif maxf < 20. and maxf > 10.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,5.e-6)
		flux2 = flux_cgs(wav_cent,10.e-6)
		flux3 = flux_cgs(wav_cent,15.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]

		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)

	elif  maxf > 3. and maxf < 10.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,2.e-6)
		flux2 = flux_cgs(wav_cent,4.e-6)
		flux3 = flux_cgs(wav_cent,6.e-6)
		flux4 = flux_cgs(wav_cent,8.e-6)
		flux5 = flux_cgs(wav_cent,10.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]

		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)

	elif maxf < 3.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		flux2 = flux_cgs(wav_cent,2.e-6)
		flux3 = flux_cgs(wav_cent,3.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]

		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)

	elif spec_feat == 'SiIV':
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		flux2 = flux_cgs(wav_cent,2.e-6)
		major_yticks = [ flux0, flux1, flux2 ]

		yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
		ax.yaxis.set_major_formatter(yticks)
	
	ax.set_yticks(major_yticks)  	#location of tickmarks	
	
	#get wavelengths corresponding to offset velocities
	def offset_vel_to_wav(voff):
		v1 = -voff
		v2 = voff
		if spec_feat =='Lya':
			wav1 = wav_e*(1.+z)*(1.+(v1/c))
			wav2 = wav_e*(1.+z)*(1.+(v2/c))
			return wav1,wav2

		else:
			wav1 = wav_e2*(1.+z)*(1.+(v1/c))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c))
			return wav1,wav2

	#define x-axis as in offset velocity units (km/s)
	wav0 	= offset_vel_to_wav(0.)
	wavlim1 = offset_vel_to_wav(1000.)
	wavlim2 = offset_vel_to_wav(2000.)
	wavlim3 = offset_vel_to_wav(3000.)
	
	major_ticks = [ wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1] ]

	ax.set_xticks(major_ticks)
	
	#fontsize of ticks
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(10)

	#define x-limits
	xmin = wavlim3[0] - 2.
	xmax = wavlim3[1] + 2.

	y = max(flux)
	
	#draw line representing central velocity of spectral feature
	if spec_feat == 'Lya':
		pl.plot([wav_o,wav_o],[-0.1*y,1.1*y],color='green',ls='--')

	else:
		pl.plot([wav_o1,wav_o1],[-0.1*y,1.1*y],color='green',ls='--')
		pl.plot([wav_o2,wav_o2],[-0.1*y,1.1*y],color='green',ls='--')
	
	#draw plot
	pl.title(spec_feat+' Fit')
	ax.set_xlabel( 'Velocity Offset (km/s)' )
	ax.set_ylabel( r'Flux Density ($\mu$Jy)' )
	ax.set_xlim([xmin,xmax])
	pl.savefig('out/line-fitting/'+spec_feat+'_components.eps')
	ax.set_ylim([-0.1*y,1.1*y])
	pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis
	pl.savefig('out/line-fitting/'+spec_feat+'_fit.eps')