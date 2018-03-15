#S.N. Kolwa (2017)
#0943_abs_line_fit.py
# Purpose:  
# - Resonant line profile fit i.e. Gaussian and Voigt
# - Flux in uJy 
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

# spec_feat = [ 'SiIV' ]
spec_feat = [ 'Lya', 'CIV', 'NV', 'SiIV']

#pixel radius for extraction
radius = 3
center = (84,106)

#define Gaussian models
def gauss(x, amp, wid, g_cen, cont):
	gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 / (2*wid**2))
	return gauss + cont

def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2,cont):
	gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
	gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
	return gauss1 + gauss2 + cont

# -----------------------------------------------
# 	   	  HeII for systemic kinematics
# -----------------------------------------------
fname 			= "./out/HeII.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

HeII 			= datacube.subcube_circle_aperture(center=center,radius=radius,\
	unit_center=None,unit_radius=None)
spec_HeII 		= HeII.sum(axis=(1,2))

var_glx = varcube.subcube_circle_aperture(center=center,radius=radius,\
	unit_center=None,unit_radius=None)
var_spec = var_glx.sum(axis=(1,2))

wav 	=  spec_HeII.wave.coord()  		#1.e-8 cm
flux 	=  spec_HeII.data				#1.e-20 erg / s / cm^2 / Ang

wav_e_HeII 	= 1640.4
z 			= 2.923

g_cen 		= wav_e_HeII*(1.+z)

pars = Parameters()

pars.add_many( \
	('g_cen', g_cen, True, g_cen-2., g_cen+2.),\
	('amp', 2.e4, True, 0.),\
	('wid', 10., True, 0.),\
	('cont', 10., True, 0.))	

mod 	= lm.Model(gauss) 

fit 	= mod.fit(flux,pars,x=wav)

# print fit.fit_report()

res 	= fit.params

amp_HeII		= fit.params['amp'].value
wav_o_HeII 		= fit.params['g_cen'].value
wav_o_err_HeII 	= fit.params['g_cen'].stderr
amp 			= fit.params['amp'].value
amp_err			= fit.params['amp'].stderr
wid_HeII 		= fit.params['wid'].value
wid_err			= fit.params['wid'].stderr
cont_HeII 		= fit.params['cont'].value

# pl.plot(wav,fit.best_fit,'r',label='model')
# pl.plot(wav,flux,drawstyle='steps-mid',color='k')
# pl.show()

# -----------------------------------------------
# 	   			CONSTANTS 	
# -----------------------------------------------

e 			= 4.8e-10				#electron charge: esu
me 			= 9.1e-28				#electron mass: g

# -----------------------------------------------
# 	   	     Voigt-Hjerting Models  	
# -----------------------------------------------
# Voigt-Hjerting function
def H(a, x):
    """ The H(a, u) function of Tepper Garcia 2006
    """
    P = x ** 2
    H0 = sp.e ** (-(x ** 2))
    Q = 1.5 / x ** 2
    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
    return H

#Voigt models for Lya (singlet)
def voigt_profile1(x,z1,b1,N1):	#absorber 1	
	nu 		= c / ( (x / (1.+z1) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b1
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile2(x,z2,b2,N2):	#absorber 2			
	nu 		= c / ( (x / (1.+z2) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b2
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile3(x,z3,b3,N3):	#absorber 3 			
	nu 		= c / ( (x / (1.+z3) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b3
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile4(x,z4,b4,N4):	#absorber 4			
	nu 		= c / ( (x / (1.+z4) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b4
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	return np.exp( -tau )

#Voigt models for CIV, NV, SiIV (doublets)
def voigt_profile1_1(x,z1,b1,N1):  	#absorber 1 at z = z1		
	nu 		= c / ( (x / (1.+z1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile2_1(x,z2,b2,N2): 	#absorber 2 at z = z1 			
	nu 		= c / ( (x / (1.+z2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile1_2(x,z3,b3,N3):  #absorber 1 at z = z2		
	nu 		= c / ( (x / (1.+z3)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b3
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile2_2(x,z4,b4,N4): #absorber 2 at z = z2 			
	nu 		= c / ( (x / (1.+z4)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b4
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	return np.exp( -tau )

fname 		= "./out/Lya.fits"

#radial velocity (Doppler)
def vel(wav_obs,wav_em,z):
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v

c = 2.9979245800e10		#cm/s
	
vel0 = vel(wav_o_HeII,wav_e_HeII,0.) 	# source frame = obs frame 
z = vel0/c 								# systemic redshift

#file into which fit parameters are written
filename = "./out/line-fitting/0943_resonant_fit.txt"

fname 			= "./out/Lya.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

Lya 			= datacube.subcube_circle_aperture(center=center,radius=radius,\
	unit_center=None,unit_radius=None)
spec_Lya 		= Lya.sum(axis=(1,2))

var_glx = varcube.subcube_circle_aperture(center=center,radius=radius,\
	unit_center=None,unit_radius=None)
var_spec = var_glx.sum(axis=(1,2))

var_flux    = var_spec.data

n 			= len(var_flux)
inv_noise 	= [ var_flux[i]**-1 for i in range(n) ] 

wav 	=  spec_Lya.wave.coord()  		#1.e-8 cm
flux 	=  spec_Lya.data				#1.e-20 erg / s / cm^2 / Ang
	
mod 	= lm.Model(gauss)*lm.Model(voigt_profile1)*lm.Model(voigt_profile2)*lm.Model(voigt_profile3)*lm.Model(voigt_profile4)

g_cen = 1215.57*(1.+z)

f 			= 0.4162				#HI oscillator strength
lam 		= 1215.57
gamma 		= 6.265e8				#gamma of HI line
nu0			= c / (lam*1.e-8) 		#Hz 

pars = Parameters()

z1 = 2.907
z2 = 2.919
z3 = 2.926
z4 = 2.934

pars.add_many( ('amp' , 5.e4, True, 0. ), ('wid', wid_HeII, True, 0.), ('g_cen', g_cen, True, g_cen-1., g_cen+1.),\
 ('z1', z1, True, z1-0.001, z1+0.001), ('b1', 80.e5, True, 10.e5, 200.e5 ), ('N1', 1.e14, True, 1.e10),\
 ('z2', z2, True, z2-0.001, z2+0.001), ('b2', 60.e5, True, 10.e5, 200.e5 ), ('N2', 1.e20, True, 1.e10), \
 ('z3', z3, True, z3-0.001, z3+0.001), ('b3', 140.e5, True, 10.e5, 200.e5 ), ('N3', 1.e14, True, 1.e10), 
 ('z4', z4, True, z4-0.001, z4+0.001), ('b4', 30.e5, True, 10.e5, 200.e5 ), ('N4', 1.e13, True, 1.e10), ('cont', cont_HeII, True, 0.) )

fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

# print fit.fit_report()

#fit parameters to pass down
wav_o 		= fit.params['g_cen'].value
wav_o_err 	= fit.params['g_cen'].stderr

N1			= fit.params['N1'].value
z1 			= fit.params['z1'].value
b1			= fit.params['b1'].value

N2			= fit.params['N2'].value
z2 			= fit.params['z2'].value
b2			= fit.params['b2'].value

N3			= fit.params['N3'].value
z3 			= fit.params['z3'].value
b3			= fit.params['b3'].value

N4			= fit.params['N4'].value
z4 			= fit.params['z4'].value
b4			= fit.params['b4'].value

cont 		= fit.params['cont'].value

for spec_feat in spec_feat:

	#From Cashman et al (2017)
	if spec_feat == 'Lya':
		f 			= 0.4162				
		lam 		= 1215.57
		gamma 		= 6.265e8				
		nu0			= c / (lam*1.e-8) 		
	
	elif spec_feat == 'CIV':
		f1 			= 0.190
		f2 			= 0.0948
		gamma1 		= 2.69e8				
		gamma2  	= 2.70e8
		lam1		= 1548.202 
		lam2		= 1550.774
		nu01		= c / (lam1*1.e-8)		
		nu02		= c / (lam2*1.e-8) 
	
	elif spec_feat == 'NV':
		f1 			= 0.0777
		f2 			= 0.156
		gamma1 		= 3.37e8				
		gamma2  	= 3.40e8
		lam1		= 1238.8
		lam2		= 1242.8
		nu01		= c / (lam1*1.e-8)
		nu02		= c / (lam2*1.e-8) 
	
	elif spec_feat == 'SiIV':
		f1 			= 0.513
		f2 			= 0.254
		gamma1 		= 8.80e8				
		gamma2  	= 8.63e8
		lam1		= 1393.76
		lam2		= 1402.77
		nu01		= c / (lam1*1.e-8)	
		nu02 		= c / (lam2*1.e-8)

	print '----'
	print spec_feat
	print '----'
	# --------------------------
	#     LOAD data cubes
	# --------------------------

	fname 		= "./out/"+spec_feat+".fits"
	datacube 	= mpdo.Cube(fname,ext=1)
	varcube		= mpdo.Cube(fname,ext=2)
	
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
	inv_noise 	= [ var_flux[i]**-1 for i in range(n) ] 

	c_kms = 2.9979245800e5

	if spec_feat == 'Lya':
		wav 		= spec.wave.coord()		
		flux  		= spec.data

		mod 	= lm.Model(gauss)*lm.Model(voigt_profile1)*lm.Model(voigt_profile2)*lm.Model(voigt_profile3)*lm.Model(voigt_profile4)

		g_cen = lam*(1.+z)

		pars = Parameters()

		pars.add_many( ('amp' , amp_HeII, True, 10.), ('wid', wid_HeII, True, 0.), ('g_cen', g_cen, True, g_cen-1.,g_cen+1.),
		 ('z1', z1, True, z1-0.001, z1+0.001), ('b1', b1, True, 10.e5, 200.e5), ('N1', N1, True, 1.e10),
		 ('z2', z2, True, z2-0.001, z2+0.001), ('b2', b2, True, 10.e5, 200.e5), ('N2', N2, True, 1.e10), 
		 ('z3', z3, True, z3-0.001, z3+0.001), ('b3', b3, True, 10.e5, 200.e5), ('N3', N3, True, 1.e10), 
		 ('z4', z4, True, z4-0.001, z4+0.001), ('b4', b4, True, 10.e5, 200.e5), ('N4', N4, True, 1.e10), ('cont', cont_HeII, True, 0.) )

		fit = mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		comps = fit.eval_components(x=wav)

		pl.plot(wav, comps['gauss'],'b--',label=`lam`+r'$\AA$')

		print fit.fit_report()

		wav_o 		= fit.params['g_cen'].value
		wav_o_err 	= fit.params['g_cen'].stderr
		
	elif spec_feat == 'CIV':
		wav 		= spec.wave.coord()		
		flux  		= spec.data

		mod 	= lm.Model(dgauss)*lm.Model(voigt_profile1_1)*lm.Model(voigt_profile1_2)*lm.Model(voigt_profile2_1)*lm.Model(voigt_profile2_2)

		g_cen1 = lam1*(1.+z)
		g_cen2 = lam2*(1.+z)

		pars = Parameters()

		v = c_kms * ( wid_HeII / wav_o_HeII ) #HeII gaussian width (km/s)

		guess_wid1 = g_cen1 * ( v / c_kms )
		guess_wid2 = g_cen2 * ( v / c_kms )

		pars.add_many( 
			('amp1', 5.e3, True, 2.e3),
		('amp2', 5.e3, True, 2.e3), 
		('cont', cont_HeII, 10.) )

		#estimated absorber redshifts
		z1_ = z2 		#absorber 2 in Lya
		z3_ = z1 		#absorber 1 in Lya

		pars.add('wid1', guess_wid1, True, 0.)
		pars.add('wid2', guess_wid2, True, 0.)

		# pars.add('wid1', guess_wid1, True, 0.)
		# pars.add('wid2', guess_wid2, True, expr='wid1')

		pars.add('g_cen1',g_cen1,True,g_cen1-1.,g_cen1+1.)
		pars.add('g_cen2',g_cen2,True,expr='1.0016612819257436*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add('z1',z1_,True,z1_-0.001,z1_+0.001)
		pars.add('z2',True,expr='z1')

		pars.add('z3',z3_,True,z3_-0.001,z3_+0.001)
		pars.add('z4',True,expr='z3')

		pars.add('N1',1.e14,True,1.e10)
		pars.add('N2',True,expr='N1')

		pars.add('b1',b2,True,0.)
		pars.add('b2',True,expr='b1')

		pars.add('N3',1.e14,True,1.e10)
		pars.add('N4',True,expr='N3')

		pars.add('b3',b1,True,0.)
		pars.add('b4',True,expr='b3')

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value
	
		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont),color='orange',linestyle='--',label=`lam1`+r'$\AA$')
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont),color='blue',linestyle='--',label=`lam2`+r'$\AA$')
		pl.legend()

		print fit.fit_report()

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

	elif spec_feat == 'NV':
		wav 		= spec.wave.coord()		
		flux  		= spec.data

		mod 	= lm.Model(dgauss)*lm.Model(voigt_profile1_1)*lm.Model(voigt_profile2_1)

		g_cen1 = lam1*(1.+z)
	 	g_cen2 = lam2*(1.+z)

	 	pars = Parameters()

	 	v = c_kms * ( wid_HeII / wav_o_HeII ) #HeII gaussian width (km/s)

		guess_wid1 = g_cen1 * ( v / c_kms )
		guess_wid2 = g_cen2 * ( v / c_kms )

		pars.add('g_cen1',g_cen1,False,min=g_cen1-0.5,max=g_cen1+0.5)
		pars.add('g_cen2',g_cen2,expr='1.003228931223765*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add_many( 
			('amp1', 600., True, 0.),
			('amp2', 1200., True, 0.), 
			('cont', cont_HeII, 10.) )

		z_ = z2

		pars.add('wid1', guess_wid1, True, 0.)
		pars.add('wid2', guess_wid2, True, 0.)

		pars.add('z1', z_, True, z_-0.001, z_+0.001)
		pars.add('z2', True, expr='z1')

		pars.add('N1', 1.e14, True, 1.e10)
		pars.add('N2', True,expr='N1')

		pars.add('b1', 400.e5, True, 0.)
		pars.add('b2', True, expr='b1')

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value
	
		print fit.fit_report()

		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont),color='orange',linestyle='--',label=`lam1`+r'$\AA$')
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont),color='blue',linestyle='--',label=`lam2`+r'$\AA$')
		pl.legend()

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

		# pl.plot(wav,fit.best_fit)
		# pl.plot(wav,flux,ls='steps-mid')
		# pl.show()

	elif spec_feat == 'SiIV':
		wav 		= spec.wave.coord()		
		flux  		= spec.data

		mod 	= lm.Model(dgauss)*lm.Model(voigt_profile1_1)*lm.Model(voigt_profile2_1)

		g_cen1 = lam1*(1.+z)
	 	g_cen2 = lam2*(1.+z)

	 	pars = Parameters()

	 	v = c_kms * ( wid_HeII / wav_o_HeII ) #HeII gaussian width (km/s)

		guess_wid1 = g_cen1 * ( v / c_kms )
		guess_wid2 = g_cen2 * ( v / c_kms )

		pars.add('g_cen1',value=g_cen1,min=g_cen1-1.,max=g_cen1+1.)
		pars.add('g_cen2',value=g_cen2,expr='1.0064645276087705*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add_many( 
			('amp1', 8.e3, True, 0.), ('wid1', guess_wid1, True, 0.),
			('amp2', 5.e3, True, 0.), ('wid2', guess_wid2, True, 0.), 
			('cont', cont_HeII, 10.,))

		z_ = z2

		pars.add('wid1', guess_wid1, True, 0.)
		pars.add('wid2', guess_wid2, True, 0.)

		pars.add('z1', z_, True, z_-0.001, z_+0.001)
		pars.add('z2', True, expr='z1')

		pars.add('N1', 1.e15, True, 1.e10)
		pars.add('N2', True,expr='N1')

		pars.add('b1', 400.e5, True, 0.)
		pars.add('b2', True, expr='b1')

		fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		res 		= fit.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		cont 		= res['cont'].value
	
		print fit.fit_report()

		pl.plot(wav,gauss(wav,amp[0],wid[0],wav_o[0],cont),color='orange',linestyle='--',label=`lam1`+r'$\AA$')
		pl.plot(wav,gauss(wav,amp[1],wid[1],wav_o[1],cont),color='blue',linestyle='--',label=`lam2`+r'$\AA$')
		pl.legend()

		wav_o1 		= fit.params['g_cen1'].value
		wav_o_err1 	= fit.params['g_cen1'].stderr	

		wav_o2 		= fit.params['g_cen2'].value
		wav_o_err2 	= fit.params['g_cen2'].stderr	

		# pl.plot(wav,fit.best_fit)
		# pl.plot(wav,flux,ls='steps-mid')
		# pl.show()

	chisqr = r'$\chi^2$: %1.2f' %fit.chisqr
	redchisqr = r'$\widetilde{\chi}^2$: %1.2f' %fit.redchi
	pl.text(0.15, 0.9, redchisqr, ha='center', va='center', transform=ax.transAxes, fontsize=14)
	pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
	pl.title(spec_feat+' Fit')
	pl.plot(wav,fit.best_fit,'r',label='model')
	pl.legend()
	pl.plot(wav,flux,drawstyle='steps-mid',color='k')

	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------						
	#radial velocity (Doppler) in km/s
	def vel(wav_obs,wav_em,z):
		v = c_kms*((wav_obs/wav_em/(1.+z)) - 1.)
		return v

	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	

	z_kms = vel0_rest/c_kms

	vel0 = vel(wav_o_HeII,wav_e_HeII,z_kms)

	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat == 'Lya':	
		wav_e = 1215.7

	elif spec_feat == 'SiIV':
		wav_e1  = 1393.76
		wav_e2  = 1402.77 
	
	elif spec_feat == 'CIV':
		wav_e1 	= 1548.2
		wav_e2 	= 1550.8

	elif spec_feat == 'NV':
		wav_e1	= 1238.8
		wav_e2	= 1242.8

	if spec_feat in ('CIV', 'NV', 'SiIV'):
		vel_meas = [ vel(wav_o1,wav_e1,z_kms),vel(wav_o2,wav_e2,z_kms) ]	
		vel_off = [ vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z_kms) - vel0)	) 

	else:
		vel_meas = vel(wav_o,wav_e,z_kms)		#central velocity of detected line
		vel_off = vel_meas - vel0			#residual velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z_kms) - vel0 ) )

	#define x-axis as velocity (km/s)
	ax.xaxis.set_major_formatter(xticks)		#format of tickmarks (i.e. unit conversion)
	
	# #convert erg/s/cm^2/Ang to Jy
	# def flux_Jy(wav,flux):
	# 	f = 3.33564095e4*flux*1.e-20*wav**2
	# 	return f
	
	# #define y-tick marks in reasonable units
	# #recover flux in 10^-20 erg/s/cm^2 and convert to microJy
	# def flux_cgs(wav,flux_Jy):
	# 	f = flux_Jy/(3.33564095e4*wav**2)
	# 	return f*1.e20

	# #central wavelength (on doublet we select longest rest wavelength)
	# if spec_feat == 'Lya':
	# 	wav_cent = wav_o 
	# 	maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy

	# else:
	# 	wav_cent = wav_o2
	# 	maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy	

	# if maxf > 0. and maxf < 1.:
	# 	flux0 = flux_cgs(wav_cent,0.)
	# 	flux1 = flux_cgs(wav_cent,1.e-6)
	# 	major_yticks = [ flux0, flux1 ]

	# elif maxf > 1. and maxf < 3.:
	# 	flux0 = flux_cgs(wav_cent,0.)
	# 	flux1 = flux_cgs(wav_cent,1.e-6)
	# 	flux2 = flux_cgs(wav_cent,2.e-6)
	# 	flux3 = flux_cgs(wav_cent,3.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3 ]

	# elif  maxf > 3. and maxf < 10.:
	# 	flux0 = flux_cgs(wav_cent,0.)
	# 	flux1 = flux_cgs(wav_cent,2.e-6)
	# 	flux2 = flux_cgs(wav_cent,4.e-6)
	# 	flux3 = flux_cgs(wav_cent,6.e-6)
	# 	flux4 = flux_cgs(wav_cent,8.e-6)
	# 	flux5 = flux_cgs(wav_cent,10.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]

	# elif maxf > 10. and maxf < 20.:
	# 	flux0 = flux_cgs(wav_cent,0.)
	# 	flux1 = flux_cgs(wav_cent,5.e-6)
	# 	flux2 = flux_cgs(wav_cent,10.e-6)
	# 	flux3 = flux_cgs(wav_cent,15.e-6)
	#	flux4 = flux_cgs(wav_cent,20.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]

	# elif maxf > 20. and maxf < 60.:
	# 	flux0 = flux_cgs(wav_cent,0.)
	# 	flux1 = flux_cgs(wav_cent,10.e-6)
	# 	flux2 = flux_cgs(wav_cent,20.e-6)
	# 	flux3 = flux_cgs(wav_cent,30.e-6)
	# 	flux4 = flux_cgs(wav_cent,40.e-6)
	# 	flux5 = flux_cgs(wav_cent,50.e-6)
	# 	flux6 = flux_cgs(wav_cent,60.e-6)
	# 	major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5, flux6 ]

	# yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) )
	# ax.yaxis.set_major_formatter(yticks)
	# ax.set_yticks(major_yticks)  	#location of tickmarks	

	#get wavelengths corresponding to offset velocities
	def offset_vel_to_wav(voff):
		v1 = -voff
		v2 = voff
		if spec_feat =='Lya':
			wav1 = wav_e*(1.+z)*(1.+(v1/c_kms))
			wav2 = wav_e*(1.+z)*(1.+(v2/c_kms))
			return wav1,wav2

		else:
			wav1 = wav_e2*(1.+z)*(1.+(v1/c_kms))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c_kms))
			return wav1,wav2

	#x-axis in velocity (km/s)
	wav0 	= offset_vel_to_wav(0.)
	wavlim1 = offset_vel_to_wav(1000.)
	wavlim2 = offset_vel_to_wav(2000.)
	wavlim3 = offset_vel_to_wav(3000.)
	
	major_ticks = [ wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1], wavlim3[1] ]

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
	
	# draw plot
	ax.set_xlabel( 'Velocity Offset (km/s)' )
	# ax.set_ylabel( r'Flux Density ($\mu$Jy)' )
	ax.set_ylabel(r'Flux Density (10$^{-20}$ erg/s/cm$^2$/$\AA$)')
	ax.set_xlim([xmin,xmax])
	pl.savefig('out/line-fitting/'+spec_feat+'_components.png')
	ax.set_ylim([-0.1*y,1.1*y])
	pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis
	pl.savefig('out/line-fitting/'+spec_feat+'_fit.png')
	# pl.show()