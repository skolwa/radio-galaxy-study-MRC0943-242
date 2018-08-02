#S.N. Kolwa (2018)
#MRC0943_resonant_line_fit_0.7.py
# Purpose:  
# - Resonant line profile fit i.e. Gaussian and Voigt
# - Flux in uJy 
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)

# - fits 4 Lya absorption profiles
# - fits 3 CIV, NV absorbers
# - double gaussian to account for blueshifted source
# - absorbers are in 4sigma agreement with Lya absorbers along L.O.S. 
# Smoothed composite model on plot by factor of 2 (finer grid)
# Corner plots using MCMC analysis
# 1D spectrum continuum subtracted

import matplotlib.pyplot as pl
import numpy as np 

import mpdaf.obj as mpdo
from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning

import astropy.units as u
import matplotlib.ticker as tk

from lmfit import *
 
from scipy.signal import gaussian,fftconvolve

from functions import *

import scipy as sp
from time import time

import corner
import os

params = {'legend.fontsize': 18,
          'legend.handlelength': 2}

t_start = time()

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )
warnings.simplefilter  ('ignore' , 	category=RuntimeWarning          )

spec_feat = ['Lya', 'NV',  'SiIV', 'CIV', 'HeII']

#pixel radius for extraction
radius = 3
center = (88,66)

# --------------------------------------------------
# 	  HeII fit to estimate ideal emission model
# --------------------------------------------------
fname 			= "./out/HeII.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

#shift centre
center1 = (center[0]-1,center[1]-1)

cube_HeII = datacube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)
var_HeII = varcube.subcube(center1,(2*radius+1),unit_center=None,unit_size=None)

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

#define continuum function
def str_line_model(p):
	x = wav_mask
	mod = p['grad']*x + p['cut']		#y = mx+c
	data = flux_mask
	weights = inv_noise
	return weights*(mod-data)

pars = Parameters()

pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))

mini = Minimizer(str_line_model, pars, nan_policy='omit')
out = mini.minimize(method='leastsq')

# report_fit(out)

grad = out.params['grad'].value
cut = out.params['cut'].value
cont = str_line(wav_mask,grad,cut)

# continuum subtract from copy of spectrum
spec_HeII_sub = spec_HeII - cont

#continuum subtracted data
wav 	=  spec_HeII_sub.wave.coord()  		#1.e-8 cm
flux 	=  spec_HeII_sub.data				#1.e-20 erg / s / cm^2 / Ang

wav_e_HeII 	= 1640.4
z 			= 2.923 					#estimate from literature

g_cen 		= wav_e_HeII*(1.+z)
g_blue 		= 6414.						#estimate from visual look at spectrum

pars = Parameters()

pars.add_many( \
	('g_cen1', g_blue, True, g_blue-10., g_blue+25.),\
	('amp1', 2.e3, True, 0.),\
	('wid1', 12., True, 0.,50.),\
	('g_cen2', g_cen, True, g_cen-1., g_cen+3.),\
	('amp2', 2.e4, True, 0.),\
	('wid2', 10., True, 0.))	

def HeII_model(p):
	x = wav
	mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'],p['amp2'],p['wid2'],p['g_cen2'])
	data = flux
	weights = inv_noise
	return weights*(mod - data)

#create minimizer
mini = Minimizer(HeII_model, pars, nan_policy='omit')

#solve with Levenberg-Marquardt
out = mini.minimize(method='leastsq')

print "------------------------"
print "  HeII fit parameters  "
print "------------------------"
report_fit(out)

pl.figure(figsize=(12,12))
res = mini.emcee(params=pars,steps=2000,thin=20,burn=200)

figure = corner.corner(res.flatchain, labels=out.var_names, truths=list(out.params.valuesdict().values()))
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/HeII_corner_plot_0.7.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/HeII_corner_plot_0_7.pdf')

amp_blue		= out.params['amp1'].value
amp_blue_err	= out.params['amp1'].stderr
wid_blue		= out.params['wid1'].value
wid_blue_err	= out.params['wid1'].stderr
wav_o_blue 		= out.params['g_cen1'].value
wav_o_blue_err 	= out.params['g_cen1'].stderr

fwhm_kms_blue_HeII 	= convert_sigma_fwhm(wid_blue, wid_blue_err, wav_o_blue, wav_o_blue_err)

z_blue 			= wav_o_blue/wav_e_HeII - 1.   #redshift of blue absorber

amp_HeII		= out.params['amp2'].value
amp_HeII_err	= out.params['amp2'].stderr
wav_o_HeII 		= out.params['g_cen2'].value
wav_o_err_HeII 	= out.params['g_cen2'].stderr
wid_HeII 		= out.params['wid2'].value
wid_HeII_err	= out.params['wid2'].stderr

fwhm_kms_HeII 	= convert_sigma_fwhm(wid_HeII, wid_HeII_err, wav_o_HeII, wav_o_err_HeII)

delg 			= wav_o_err_HeII 		#error on gaussian centre for all lines

# display HeII spectrum
fig = pl.figure(figsize=(10,12))
x = np.linspace( min(wav), max(wav), num=2*len(wav) )
pl.plot(x, dgauss_nocont(x, amp_blue, wid_blue, g_blue, amp_HeII, wid_HeII, wav_o_HeII),'r', label='model')
pl.plot(wav, flux, drawstyle='steps-mid', c='k')
# pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/HeII_fit_prior_0_7.png')

# -----------------------------------------------
# 	   			CONSTANTS (cgs)	
# -----------------------------------------------

e 			= 4.8e-10				#electron charge: esu
me 			= 9.1e-28				#electron mass: g
c 			= 2.9979245800e10		#cm/s

# -----------------------------------------------
# 	   	     Voigt-Hjerting Models  	
# -----------------------------------------------
# Voigt-Hjerting function
def H(a, x):
    """ The H(a, u) approximation from Tepper Garcia (2006)
    """
    P = x ** 2
    H0 = sp.e ** (-(x ** 2))
    Q = 1.5 / x ** 2
    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
    return H

# For CONVOLUTION with LSF
# From Kristian-Krogager (2018), VoigtFit package (voigt.py -> evaluate_profile function)
fwhm 	= 2.65 / 2 
sigma 	= fwhm / 2.35482

# ---------------------------------
#   Voigt models for Lya (singlet)
# ---------------------------------
def voigt_profile1(x,z1,b1,N1):	#absorber 1	
	nu 		= c / ( (x / (1.+z1) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b1
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt
	

def voigt_profile2(x,z2,b2,N2):	#absorber 2			
	nu 		= c / ( (x / (1.+z2) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b2
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt
	

def voigt_profile3(x,z3,b3,N3):	#absorber 3 			
	nu 		= c / ( (x / (1.+z3) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b3
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt
	

def voigt_profile4(x,z4,b4,N4):	#absorber 4			
	nu 		= c / ( (x / (1.+z4) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b4
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt	
	

# ------------------------------------------------------
#     Voigt models for CIV, NV and SiIV (doublets)
# ------------------------------------------------------

def voigt_profile1_1(x,z1_1,b1_1,N1_1):  	#absorber 1 at wav1		
	nu 		= c / ( (x / (1.+z1_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b1_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N1_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt

def voigt_profile1_2(x,z1_2,b1_2,N1_2): 	#absorber 1 at wav2		
	nu 		= c / ( (x / (1.+z1_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b1_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N1_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt

def voigt_profile2_1(x,z2_1,b2_1,N2_1):  	#absorber 2 at wav1		
	nu 		= c / ( (x / (1.+z2_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b2_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N2_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt	

def voigt_profile2_2(x,z2_2,b2_2,N2_2): 	#absorber 2 at wav2 			
	nu 		= c / ( (x / (1.+z2_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b2_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N2_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt

def voigt_profile3_1(x,z3_1,b3_1,N3_1):  	#absorber 2 at wav1		
	nu 		= c / ( (x / (1.+z3_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b3_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N3_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt

def voigt_profile3_2(x,z3_2,b3_2,N3_2): 	#absorber 3 at wav2 			
	nu 		= c / ( (x / (1.+z3_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b3_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N3_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	LSF 	= gaussian( len(x), sigma )
	LSF 	= LSF/LSF.sum()
	conv_voigt = fftconvolve( voigt, LSF, 'same')
	return conv_voigt
	
#radial velocity (Doppler) [cm/s]
def vel_cgs(wav_obs,wav_em,z):
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v
	
vel0 = vel_cgs(wav_o_HeII, wav_e_HeII, 0.) 	# source frame = obs frame 

# systemic redshift
z_sys 		= vel0 / c 								
z_sys_err 	= ( wav_o_err_HeII / wav_o_HeII )*z_sys

print 'Systemic Redshift: %2.6f +/- %2.6f' %(z_sys,z_sys_err)

# --------------------------------------------------------
#     Lya fit to estimate absorber redshifts/velocities
# --------------------------------------------------------
fname 			= "./out/Lya.fits"
datacube 		= mpdo.Cube(fname,ext=1)
varcube  		= mpdo.Cube(fname,ext=2)

#shift target pixel to center
center2 = (center[0]-1,center[1]-1)

cube_Lya = datacube.subcube(center2,(2*radius+1),unit_center=None,unit_size=None)
var_Lya = varcube.subcube(center2,(2*radius+1),unit_center=None,unit_size=None)

#mask cubes
cubes = [cube_Lya,var_Lya]
for cube in cubes:
	subcube_mask(cube,radius)

#obtain spatially integrated spectrum
spec_Lya   		= cube_Lya.sum(axis=(1,2))
spec_Lya_copy 	= cube_Lya.sum(axis=(1,2))
var_spec_Lya 	= var_Lya.sum(axis=(1,2))

var_flux    = var_spec_Lya.data

n 			= len(var_flux)
inv_noise 	= [ var_flux[i]**-1 for i in range(n) ] 

#mask non-continuum lines 
spec_Lya_copy.mask_region(4720.,4815.)	#Lya line
spec_Lya_copy.mask_region(4840.,4912.)	#NV line 

wav_mask 	=  spec_Lya_copy.wave.coord()  		#1.e-8 cm
flux_mask 	=  spec_Lya_copy.data				#1.e-20 erg / s / cm^2 / Ang

pars = Parameters()
pars.add_many(('grad', 0.01, True), ('cut', 50., True,))

mini = Minimizer(str_line_model, pars, nan_policy='omit')
out = mini.minimize(method='leastsq')

# report_fit(out)

grad = out.params['grad'].value
cut = out.params['cut'].value
cont = str_line(wav_mask,grad,cut)

# continuum subtract from copy of spectrum
spec_Lya_sub = spec_Lya - cont

wav 	=  spec_Lya_sub.wave.coord()  		#1.e-8 cm
flux 	=  spec_Lya_sub.data				#1.e-20 erg / s / cm^2 / Ang

f 		= 0.4162				#HI oscillator strength
lam 	= 1215.57				#rest wavelength of Lya
gamma 	= 6.265e8				#gamma of HI line
nu0		= c / (lam*1.e-8) 		#Hz 

#initial guesses (based on literature values)
pars = Parameters()

z1 = 2.9070
z2 = 2.9190
z3 = 2.9276
z4 = 2.9328

g_cen 	= lam*(1.+z_sys)
g_blue 	= lam*(1.+z_blue)
bmin = 37.4e5
bmax = 500.e5
Nmin = 1.e11

pars.add_many( ('amp1' , 8*amp_blue, True, 0. ), ('wid1', wid_blue, True, 0.), ('g_cen1', g_blue, True, g_blue-delg, g_blue+1.2*delg),\
 ('amp2' , 8*amp_HeII, True, 0. ), ('wid2', wid_HeII, True, 0.), ('g_cen2', g_cen, True, g_cen-delg, g_cen+delg),\
 ('z1', z1, True, z1-0.005, z1+0.005), ('b1', 90.e5, True, bmin, bmax ), ('N1', 1.e14, True, Nmin),
 ('z2', z2, True, z2-0.001, z2+0.001), ('b2', 60.e5, True, bmin, bmax ), ('N2', 1.e19, True, Nmin), 
 ('z3', z3, True, z3-0.004, z3+0.004), ('b3', 104.e5, False, bmin, bmax ), ('N3', 1.e14, True, Nmin), 
 ('z4', z4, True, z4-0.003, z4+0.003), ('b4', 50.e5, False, bmin, bmax ), ('N4', 1.e13, True, Nmin) )

def four_abs_model(p):
	x = wav
	mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'],p['amp2'],p['wid2'],p['g_cen2'])\
	*voigt_profile1(x,p['z1'], p['b1'], p['N1'])*voigt_profile2(x, p['z2'], p['b2'], p['N2'])\
	*voigt_profile3(x,p['z3'], p['b3'], p['N3'])*voigt_profile4(x, p['z4'], p['b4'], p['N4'])
	data = flux
	weights = inv_noise
	return weights*(mod - data)

#create minimizer
mini = Minimizer(four_abs_model, pars, nan_policy='omit')

#solve with Levenberg-Marquardt
out = mini.minimize(method='leastsq')

print "------------------------"
print "  Lya fit parameters    "
print "------------------------"
report_fit(out, min_correl=0.8)

pl.figure(figsize=(12,12))
res = mini.emcee(params=pars,steps=2000,thin=20,burn=200)

figure = corner.corner(res.flatchain, labels=out.var_names, truths=list(out.params.valuesdict().values()))
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/Lya_corner_plot_0.7.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/Lya_corner_plot_0_7.pdf')

#fit parameters to pass down
amp_blue_Lya	= out.params['amp1'].value
wid_blue_Lya 	= out.params['wid1'].value
wav_o_blue_Lya 	= out.params['g_cen1'].value

amp_blue_Lya_err	= out.params['amp1'].stderr
wid_blue_Lya_err 	= out.params['wid1'].stderr
wav_o_blue_Lya_err 	= out.params['g_cen1'].stderr

amp_Lya			= out.params['amp2'].value
amp_Lya_err		= out.params['amp2'].stderr

wid_Lya 		= out.params['wid2'].value
wid_Lya_err 	= out.params['wid2'].stderr

wav_o_Lya 		= out.params['g_cen2'].value
wav_o_Lya_err 	= out.params['g_cen2'].stderr

N1			= out.params['N1'].value
N1_err		= out.params['N1'].stderr
z1 			= out.params['z1'].value
z1_err 		= out.params['z1'].stderr
b1			= out.params['b1'].value
b1_err		= out.params['b1'].stderr

N2			= out.params['N2'].value
N2_err		= out.params['N2'].stderr
z2 			= out.params['z2'].value
z2_err 		= out.params['z2'].stderr
b2			= out.params['b2'].value
b2_err		= out.params['b2'].stderr
	
N3			= out.params['N3'].value
N3_err		= out.params['N3'].stderr
z3 			= out.params['z3'].value
z3_err 		= out.params['z3'].stderr
b3			= out.params['b3'].value
b3_err		= out.params['b3'].stderr

N4			= out.params['N4'].value
N4_err		= out.params['N4'].stderr
z4 			= out.params['z4'].value
z4_err 		= out.params['z4'].stderr
b4			= out.params['b4'].value
b4_err		= out.params['b4'].stderr

fwhm_kms_Lya 		= convert_sigma_fwhm(wid_Lya, wid_Lya_err, wav_o_Lya, wav_o_Lya_err)
fwhm_kms_blue_Lya 	= convert_sigma_fwhm(wid_blue_Lya, wid_blue_Lya_err, wav_o_blue, wav_o_blue_Lya_err)

# -------------------------
#   Resonant line Fitting 
# -------------------------
# Lya (absorption model) and HeII (emission model) fit params as initial guesses in resonant line fitting
# Atomic constants taken from Cashman et al & C. Churchill (cwc@nmsu.edu) notes (2017)

for spec_feat in spec_feat:

	if spec_feat == 'Lya':
		f 			= 0.4164				
		lam 		= 1215.67			# Ang
		gamma 		= 6.265e8	    	# /s			
		nu0			= c / (lam*1.e-8) 	# Hz		
	
	elif spec_feat == 'CIV':
		f1 			= 0.190
		f2 			= 0.0952
		gamma1 		= 2.69e8				
		gamma2  	= 2.70e8
		lam1		= 1548.187 
		lam2		= 1550.772
		nu01		= c / (lam1*1.e-8)		
		nu02		= c / (lam2*1.e-8) 
	
	elif spec_feat == 'NV':
		f1 			= 0.156
		f2 			= 0.078
		gamma1 		= 3.37e8				
		gamma2  	= 3.40e8
		lam1		= 1238.821
		lam2		= 1242.804
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

	elif spec_feat == 'HeII':
		lam 		= 1640.4	

	print '----'
	print ''+spec_feat
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

	#shift centre
	center3 = (center[0]-1,center[1]-1)

	glx = datacube.subcube(center3,(2*radius+1),unit_center=None,unit_size=None)
	var_glx = varcube.subcube(center3,(2*radius+1),unit_center=None,unit_size=None)

	cubes = [glx,var_glx]
	for cube in cubes:
		subcube_mask(cube,radius)

	spec   		= glx.sum(axis=(1,2))
	spec_copy   = glx.sum(axis=(1,2))
	var_spec    = var_glx.sum(axis=(1,2))

	var_flux    = var_spec.data

	n 			= len(var_flux)
	inv_noise 	= [ var_flux[i]**-1 for i in range(n) ] 

	c_kms = 2.9979245800e5

	if spec_feat == 'NV':
		#mask non-continuum lines 
		spec_copy.mask_region(4838., 4914.)	#NV line
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq',maxfev=4000)
		
		# report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		g_cen1 = lam1*(1.+z_sys)
	 	g_cen2 = lam2*(1.+z_sys)

	 	pars = Parameters()

	 	#initial guess Gaussian width for NV lines
		guess_wid = g_cen1 * ( wid_HeII / wav_o_HeII )

		pars.add('g_cen1',g_cen1,True, g_cen1-10.*delg, g_cen1+10.*delg)
		pars.add('g_cen2',expr='1.003228931223765*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add('amp1', 0.2*amp_HeII, True, 0.)
		pars.add('amp2',expr='0.5*amp1')

		pars.add('wid1', guess_wid, True, 0.)
		pars.add('wid2', expr='wid1')

		deltaz = 0.0006
		bmin = 37.4e5
		bmax = 200.e5
		Nmin = 1.e13
		Nmax = 1.e21
		
		# absorber 1
		pars.add('z1_1', z1, True, z1-10.*deltaz,z1+10.*deltaz)
		pars.add('z1_2',expr='z1_1')

		pars.add('N1_1', 1.e14, True, Nmin, Nmax)
		pars.add('N1_2', expr='N1_1')

		pars.add('b1_1', 180.e5, False, bmin, bmax)
		pars.add('b1_2',expr='b1_1')

		# absorber 2
		pars.add('z2_1', z2, True, z2-10.*deltaz, z2+10.*deltaz)
		pars.add('z2_2',expr='z2_1')

		pars.add('N2_1', 1.e14, True, Nmin, Nmax)
		pars.add('N2_2', expr='N2_1')

		pars.add('b2_1', 180.e5, False, bmin, bmax)
		pars.add('b2_2',expr='b2_1')

		# pars.add('b2_1', 40.e5, False, bmin, bmax)
		# pars.add('b2_2',expr='b2_1')

		# absorber 3
		pars.add('z3_1', z3, True, z3-10.*deltaz, z3+10.*deltaz)
		pars.add('z3_2',expr='z3_1')

		pars.add('N3_1', 1.e14, True, Nmin, Nmax)
		pars.add('N3_2', expr='N3_1')

		pars.add('b3_1', 60.e5, False, bmin, bmax)
		pars.add('b3_2',expr='b3_1')

		# pars.add('b3_1', b3, False, bmin, bmax)
		# pars.add('b3_2',expr='b3_1')

		def NV_model(p):
			x = wav
			mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'], p['amp2'],p['wid2'],p['g_cen2'])\
			*voigt_profile1_1(x,p['z1_1'], p['b1_1'], p['N1_1'])*voigt_profile1_2(x,p['z1_2'], p['b1_2'], p['N1_2'])\
			*voigt_profile2_1(x,p['z2_1'], p['b2_1'], p['N2_1'])*voigt_profile2_2(x,p['z2_2'], p['b2_2'], p['N2_2'])\
			*voigt_profile3_1(x,p['z3_1'], p['b3_1'], p['N3_1'])*voigt_profile3_2(x,p['z3_2'], p['b3_2'], p['N3_2'])
			data = flux
			weights = inv_noise
			return weights*(mod - data)

		#create minimizer
		mini = Minimizer(NV_model, params=pars, nan_policy='omit')

		#solve with Levenberg-Marquardt
		out = mini.minimize(method='leastsq')

		report_fit(out)

		pl.figure(figsize=(12,12))
		res = mini.emcee(params=pars,steps=1000,thin=20,burn=200)
		
		figure = corner.corner(res.flatchain, labels=out.var_names, truths=list(out.params.valuesdict().values()))

		pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_corner_plot_0.7.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_corner_plot_0_7.pdf')

	elif spec_feat == 'SiIV':
		#mask non-continuum lines 
		spec_copy.mask_region(5460.,5550.)	#NV line
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01, True), ('cut', 10., True))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		# report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		g_cen1 = lam1*(1.+z_sys)

	 	lam_OIV = 1401.20  #from Draine - physics of the ISM
	 	g_cen_OIV = lam_OIV*(1.+z_sys)

	 	pars = Parameters()

	 	#initial guess Gaussian width for NV lines
		guess_wid = g_cen1 * ( wid_HeII / wav_o_HeII ) 

		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.0064645276087705*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add('amp1', 0.2*amp_HeII, True, 0.)
		pars.add('amp2',expr='0.5*amp1')

		pars.add('wid1', guess_wid, True, 0.)
		pars.add('wid2', expr='wid1')

		deltaz = 0.0006
		bmin = 37.4e5
		bmax = 500.e5
		Nmin = 1.e13
		Nmax = 1.e21

		# absorber 1
		pars.add('z1_1', z1, True, z1-deltaz, z1+deltaz)
		pars.add('z1_2',expr='z1_1')

		pars.add('N1_1', 1.e14, True, Nmin, Nmax)
		pars.add('N1_2', expr='N1_1')

		pars.add('b1_1', b1, True, bmin, bmax)
		pars.add('b1_2',expr='b1_1')

		# absorber 2
		pars.add('z2_1', z2, True, z2-deltaz, z2+deltaz)
		pars.add('z2_2',expr='z2_1')

		pars.add('N2_1', 1.e14, True, Nmin, Nmax)
		pars.add('N2_2',expr='N2_1')

		pars.add('b2_1', b2, True ,bmin, bmax)
		pars.add('b2_2',expr='b2_1')

		# absorber 3
		pars.add('z3_1', z3, True, z3-deltaz, z3+deltaz)
		pars.add('z3_2',expr='z3_1')

		pars.add('N3_1', 1.e13, True, Nmin, Nmax)
		pars.add('N3_2', expr='N3_1')

		pars.add('b3_1', 180.e5, False, bmin, bmax)
		pars.add('b3_2',expr='b3_1')

		# OIV] params
		guess_wid = g_cen_OIV * ( wid_HeII / wav_o_HeII ) 
	 	pars.add_many( ('amp', 0.2*amp_HeII, True, 0.), ( 'wid', guess_wid, True, 0. ), 
	 		( 'g_cen', g_cen_OIV, True, g_cen_OIV-delg, g_cen_OIV+delg ) )

		def SiIV_model(p):
			x = wav
			mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'], p['amp2'],p['wid2'],p['g_cen2'])\
			*voigt_profile1_1(x,p['z1_1'], p['b1_1'], p['N1_1'])*voigt_profile1_2(x,p['z1_2'], p['b1_2'], p['N1_2'])\
			*voigt_profile2_1(x, p['z2_1'], p['b2_1'], p['N2_1'])*voigt_profile2_2(x, p['z2_2'], p['b2_2'], p['N2_2'])\
			+ gauss_nocont(x, p['amp'], p['wid'], p['g_cen'])
			data = flux
			weights = inv_noise
			return weights*(mod - data)

		#create minimizer
		mini = Minimizer(SiIV_model, params=pars, nan_policy='omit')

		#solve with Levenberg-Marquardt
		out = mini.minimize(method='leastsq')

		report_fit(out)

		pl.figure(figsize=(12,12))
		res = mini.emcee(params=pars,steps=500,burn=50)
		
		figure = corner.corner(res.flatchain, labels=out.var_names, truths=list(out.params.valuesdict().values()))
		pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_corner_plot_0.7.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_corner_plot_0_7.pdf')

	elif spec_feat == 'CIV':
		#mask non-continuum lines 
		spec_copy.mask_region(6034., 6119.)	#CIV line
		spec_copy.mask_region(6299., 6303.)	#sky line  (?)
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01, True), ('cut', 50., True))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		# report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		# initial guesses for line centres
		g_cen1 = lam1*(1.+z_sys)

		#initial guess Gaussian width for LyaV lines
		guess_wid = g_cen1 * ( wid_HeII / wav_o_HeII ) 

		pars = Parameters()

		pars.add('amp1', 1.2*amp_HeII, True, 0.)
		pars.add('amp2', expr='0.5*amp1')

		pars.add('wid1', guess_wid, True, 0., 50.)
		pars.add('wid2', expr='wid1')

		pars.add('g_cen1', g_cen1, True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2', expr='1.0016612819257436*g_cen1') #from ratio of rest-frame doublet wavelengths

		deltaz = 0.0006
		bmin   = 37.4e5
		bmax   = 200.e5
		Nmin   = 1.e13
		Nmax   = 1.e21

		# absorber 1
		pars.add('z1_1', z1,True, z1-deltaz, z1+deltaz)
		pars.add('z1_2',expr='z1_1')

		pars.add('N1_1',1.e13,True,Nmin,Nmax)
		pars.add('N1_2',expr='N1_1')

		pars.add('b1_1',b1,True,bmin,bmax)
		pars.add('b1_2',expr='b1_1')

		# absorber 2
		pars.add('z2_1',z2,True,z2-deltaz,z2+deltaz)
		pars.add('z2_2',expr='z2_1')

		pars.add('N2_1',1.e13,True,Nmin,Nmax)
		pars.add('N2_2',expr='N2_1')

		pars.add('b2_1',b2,True,bmin,bmax)
		pars.add('b2_2',expr='b2_1')

		# absorber 3
		pars.add('z3_1',z3,True,z3-deltaz,z3+deltaz)
		pars.add('z3_2',expr='z3_1')

		pars.add('N3_1',1.e13,True,Nmin,Nmax)
		pars.add('N3_2',expr='N3_1')

		pars.add('b3_1',120.e5,False)
		pars.add('b3_2',expr='b3_1')

		def CIV_model(p):
			x = wav
			mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'], p['amp2'],p['wid2'],p['g_cen2'])\
			*voigt_profile1_1(x,p['z1_1'], p['b1_1'], p['N1_1'])*voigt_profile1_2(x,p['z1_2'], p['b1_2'], p['N1_2'])\
			*voigt_profile2_1(x,p['z2_1'], p['b2_1'], p['N2_1'])*voigt_profile2_2(x,p['z2_2'], p['b2_2'], p['N2_2'])\
			*voigt_profile3_1(x,p['z3_1'], p['b3_1'], p['N3_1'])*voigt_profile3_2(x,p['z3_2'], p['b3_2'], p['N3_2'])	
			data = flux
			weights = inv_noise
			return weights*(mod - data)

		#create minimizer
		mini = Minimizer(CIV_model, params=pars, nan_policy='omit')

		#solve with Levenberg-Marquardt
		out = mini.minimize(method='leastsq')

		report_fit(out)

		pl.figure(figsize=(12,12))
		res = mini.emcee(params=pars,steps=500,thin=20,burn=20)
		
		figure = corner.corner(res.flatchain, labels=out.var_names, truths=list(out.params.valuesdict().values()))
		pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_corner_plot_0.7.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_corner_plot_0_7.pdf')
