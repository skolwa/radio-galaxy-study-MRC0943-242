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

pl.rcParams.update(params)

pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})
pl.rc('text', usetex=True)

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
# pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/HeII_fit_prior_0.7.png')

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

def voigt_profile3_1(x,z3_1,b3_1,N3_1):  	#absorber 3 at wav1		
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

pars.add_many( 
('amp1' , 8*amp_blue, True, 0. ), ('wid1', wid_blue, True, 0.), ('g_cen1', g_blue, True, g_blue-delg, g_blue+1.2*delg),
 ('amp2' , 8*amp_HeII, True, 0. ), ('wid2', wid_HeII, True, 0.), ('g_cen2', g_cen, True, g_cen-delg, g_cen+delg),
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

# # display Lya spectrum
# x = np.linspace( min(wav), max(wav), num=2*len(wav) )

# fn = dgauss(x, amp_blue, wid_blue, wav_o_blue, amp_Lya, wid_Lya, wav_o_Lya, cont)\
# *voigt_profile1(x, z1, b1, N1)*voigt_profile2(x, z2, b2, N2)\
# *voigt_profile3(x, z3, b3, N3)*voigt_profile4(x, z4, b4, N4)

# pl.plot(x, fn,'r', label='model')
# pl.plot(wav, flux, drawstyle='steps-mid', c='k')
# pl.show()

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

	# line-fitting
	if spec_feat == 'HeII':
		#mask non-continuum lines 
		spec_copy.mask_region(6298.,6302.)	#skyline (?)
		spec_copy.mask_region(6366.,6570.)	#HeII and OIII] doublet lines
		
		wav_mask 	= spec_copy.wave.coord()
		flux_mask 	= spec_copy.data
		
		pars = Parameters()
		pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
				
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		#continuum subtracted data
		wav 	=  spec_sub.wave.coord()  		#1.e-8 cm
		flux 	=  spec_sub.data				#1.e-20 erg / s / cm^2 / Ang

		fig = pl.figure(figsize=(10,12))
		ax = pl.gca()
		pl.plot(wav, gauss_nocont(wav,amp_blue,wid_blue,wav_o_blue), c='blue', ls='--', label=r'blueshifted component')
		pl.plot(wav, gauss_nocont(wav,amp_HeII,wid_HeII,wav_o_HeII), ls='--', c='orange', label=r'HeII $\lambda$'+`lam`)

		y = max(flux)	

		# HeII flux peak
		flux_peak 		= amp_HeII/(wid_HeII*np.sqrt(2.*np.pi)) 
		flux_peak_err	= flux_peak*np.sqrt( (amp_HeII_err/amp_HeII)**2 + (wid_HeII_err/wid_HeII)**2 )

		# HeII blueshifted component flux peak 
		flux_peak_blue 		= amp_blue/(wid_blue*np.sqrt(2.*np.pi)) 
		flux_peak_blue_err	= flux_peak_blue*np.sqrt( (amp_blue_err/amp_blue)**2 + (wid_blue_err/wid_blue)**2 )

	elif spec_feat == 'Lya':
		#mask non-continuum lines 
		spec_copy.mask_region(4720., 4815.)	#Lya line
		spec_copy.mask_region(4840., 4912.)	#NV line 
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01, True), ('cut', 50.,True))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		# report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value

		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub 	= spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		wav_abs1 = lam*(1.+z1)
		wav_abs2 = lam*(1.+z2)
		wav_abs3 = lam*(1.+z3)
		wav_abs4 = lam*(1.+z4)

		wav_abs1_err = abs(wav_abs1*( z1_err/z1 ))
		wav_abs2_err = abs(wav_abs2*( z2_err/z2 ))
		wav_abs3_err = abs(wav_abs3*( z3_err/z3 ))
		wav_abs4_err = abs(wav_abs4*( z4_err/z4 ))

		vel0_rest = vel_cgs(wav_o_HeII, wav_e_HeII, z_sys)

		# emission velocity
		vel_o 		= ( vel_cgs(wav_o_Lya, lam, z_sys) - vel0_rest )/1.e5
		vel_o_err 	= abs( vel_o*( np.sqrt( ( wav_o_Lya_err/wav_o_Lya )**2 + (z_sys_err/z_sys)**2 - 2*(wav_o_Lya_err/wav_o_Lya)*(z_sys_err/z_sys)\
		+ wav_o_Lya_err**2 + z_sys_err**2 ) ) )

		#absorber velocities
		vel_abs1 = (vel_cgs( wav_abs1, lam, z_sys) - vel0_rest)/1.e5
		vel_abs2 = (vel_cgs( wav_abs2, lam, z_sys) - vel0_rest)/1.e5
		vel_abs3 = (vel_cgs( wav_abs3, lam, z_sys) - vel0_rest)/1.e5
		vel_abs4 = (vel_cgs( wav_abs4, lam, z_sys) - vel0_rest)/1.e5

		vel_abs1_err = abs( vel_abs1*( np.sqrt( ( wav_abs1_err/wav_abs1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs1_err/wav_abs1)*(z_sys_err/z_sys)\
		+ wav_abs1_err**2 + z_sys_err**2 ) ) )

		vel_abs2_err = abs( vel_abs2*( np.sqrt( ( wav_abs2_err/wav_abs1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs2_err/wav_abs2)*(z_sys_err/z_sys)\
		+ wav_abs2_err**2 + z_sys_err**2 ) ) )

		vel_abs3_err = abs( vel_abs3*( np.sqrt( ( wav_abs3_err/wav_abs1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_err/wav_abs3)*(z_sys_err/z_sys)\
		+ wav_abs3_err**2 + z_sys_err**2 ) ) )

		vel_abs4_err = abs( vel_abs4*( np.sqrt( ( wav_abs4_err/wav_abs1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs4_err/wav_abs4)*(z_sys_err/z_sys)\
		+ wav_abs4_err**2 + z_sys_err**2 ) ) )

		fig = pl.figure(figsize=(10,12))
		ax = pl.gca()
		pl.plot(wav,gauss_nocont(wav, amp_blue_Lya, wid_blue_Lya, wav_o_blue_Lya), c='blue', ls='--',label=r'blueshifted component')
		pl.plot(wav,gauss_nocont(wav, amp_Lya, wid_Lya, wav_o_Lya), c='orange', ls='--', label=r'Lya $\lambda$'+`lam`)

		y = max(flux)

		# Lya peak flux
		flux_peak 		= amp_Lya/(wid_Lya*np.sqrt(2.*np.pi)) 
		flux_peak_err	= flux_peak*np.sqrt( (amp_Lya_err/amp_Lya)**2 + (wid_Lya_err/wid_Lya)**2 )

		# Lya blueshifted component peak flux
		flux_peak_blue 		= amp_blue_Lya/(wid_blue_Lya*np.sqrt(2.*np.pi)) 
		flux_peak_blue_err	= flux_peak_blue*np.sqrt( (amp_blue_Lya_err/amp_blue_Lya)**2 + (wid_blue_Lya_err/wid_blue_Lya)**2 )

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

		res 		= out.params

		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err2  = wav_o[0]*(res['g_cen1'].stderr/wav_o[1])
		wav_o_err 	= (res['g_cen1'].stderr, wav_o_err2)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err2	= wid[0]*(res['wid1']/wid[1])
		wid_err  	= (res['wid1'].stderr,wid_err2)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err2	= amp[0]*(res['amp1'].stderr/amp[1])
		amp_err 	= (res['amp1'].stderr,amp_err2)

		fwhm1_kms  = convert_sigma_fwhm(wid[0],wid_err[0],wav_o[0],wav_o_err[0])
		fwhm2_kms  = convert_sigma_fwhm(wid[1],wid_err[0],wav_o[1],wav_o_err[1])
	
		fig = pl.figure(figsize=(10,12))
		ax = pl.gca()
		pl.plot(wav,gauss_nocont(wav,amp[0],wid[0],wav_o[0]),c='blue',ls='--',label=r'CIV $\lambda$'+'%.2f'%lam1)
		pl.plot(wav,gauss_nocont(wav,amp[1],wid[1],wav_o[1]),c='orange',ls='--',label=r'CIV $\lambda$'+'%.2f'%lam2)

		wav_o1 		= res['g_cen1'].value
		wav_o_err1 	= res['g_cen1'].stderr	

		wav_o2 		= res['g_cen2'].value
		wav_o_err2 	= res['g_cen2'].stderr

		z_abs1 		= res['z1_1'].value
		z_abs2 		= res['z2_1'].value
		z_abs3 		= res['z3_1'].value

		z_abs_err1 	= res['z1_1'].stderr
		z_abs_err2 	= res['z2_1'].stderr
		z_abs_err3 	= res['z3_1'].stderr

		b_abs1 		= res['b1_1'].value
		b_abs2 		= res['b2_1'].value
		b_abs3 		= res['b3_1'].value

		b_abs_err1 	= res['b1_1'].stderr
		b_abs_err2 	= res['b2_1'].stderr
		b_abs_err3 	= res['b3_1'].stderr

		N_abs1_1 	= res['N1_1'].value
		N_abs1_2 	= res['N1_2'].value
		N_abs2_1 	= res['N2_1'].value
		N_abs2_2 	= res['N2_2'].value
		N_abs3_1 	= res['N3_1'].value
		N_abs3_2 	= res['N3_2'].value

		N_abs1_1_err 	= res['N1_1'].stderr
		N_abs1_2_err 	= res['N1_2'].stderr
		N_abs2_1_err 	= res['N2_1'].stderr
		N_abs2_2_err 	= res['N2_2'].stderr
		N_abs3_1_err 	= res['N3_1'].stderr
		N_abs3_2_err 	= res['N3_2'].stderr

		# absorber wavelengths
		wav_abs1_1 = lam1*(1.+z_abs1)
		wav_abs1_2 = lam2*(1.+z_abs1)

		wav_abs2_1 = lam1*(1.+z_abs2)
		wav_abs2_2 = lam2*(1.+z_abs2)

		wav_abs3_1 = lam1*(1.+z_abs3)
		wav_abs3_2 = lam2*(1.+z_abs3)

		wav_abs1_1_err = abs(wav_abs1_1 *( z_abs_err1/z_abs1 ))
		wav_abs1_2_err = abs(wav_abs1_2 *( z_abs_err1/z_abs1 ))

		wav_abs2_1_err = abs(wav_abs2_1 *( z_abs_err2/z_abs2 ))
		wav_abs2_2_err = abs(wav_abs2_2 *( z_abs_err2/z_abs2 ))

		wav_abs3_1_err = abs(wav_abs3_1 *( z_abs_err3/z_abs3 ))
		wav_abs3_2_err = abs(wav_abs3_2 *( z_abs_err3/z_abs3 ))

		vel0_rest = vel_cgs(wav_o_HeII, wav_e_HeII, z_sys)

		# emission velocity
		vel_o1 		= (vel_cgs(wav_o1, lam, z_sys) - vel0_rest)/1.e5
		vel_o2 		= (vel_cgs(wav_o2, lam, z_sys) - vel0_rest)/1.e5

		vel_o1_err 		= abs ( vel_o1*( ( wav_o_err1/wav_o1 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err1/wav_o1)\
			+ wav_o_err1**2 + z_sys_err**2) )
		vel_o2_err 	= abs ( vel_o2*( ( wav_o_err2/wav_o2 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err2/wav_o2)\
			+ wav_o_err2**2 + z_sys_err**2) )

		#velocity shift between doublets
		v_shift = vel_o2 - vel_o1

		# absorber velocities
		vel_abs1_1 = (vel_cgs( wav_abs1_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs1_2 = (vel_cgs( wav_abs1_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs2_1 = (vel_cgs( wav_abs2_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs2_2 = (vel_cgs( wav_abs2_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs3_1 = (vel_cgs( wav_abs3_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs3_2 = (vel_cgs( wav_abs3_2, lam2, z_sys) - vel0_rest)/1.e5


		vel_abs1_1_err = abs( vel_abs1_1*( np.sqrt( ( wav_abs1_1_err/wav_abs1_1 )**2 + ( z_sys_err/z_sys )**2 - 2*( wav_abs1_1_err/wav_abs1_1 )*( z_sys_err/z_sys )\
		+ wav_abs1_1_err**2 + z_sys_err**2 ) ) )

		vel_abs1_2_err = abs( vel_abs1_2*( np.sqrt( ( wav_abs1_2_err/wav_abs1_2 )**2 + ( z_sys_err/z_sys )**2 - 2*( wav_abs1_2_err/wav_abs1_2 )*( z_sys_err/z_sys )\
		+ wav_abs1_2_err**2 + z_sys_err**2 ) ) )

		vel_abs2_1_err = abs( vel_abs2_1*( np.sqrt( ( wav_abs2_1_err/wav_abs2_1 )**2 + ( z_sys_err/z_sys )**2 - 2*( wav_abs2_1_err/wav_abs2_1 )*( z_sys_err/z_sys )\
		+ wav_abs2_1_err**2 + z_sys_err**2 ) ) )

		vel_abs2_2_err = abs( vel_abs2_2*( np.sqrt( ( wav_abs2_2_err/wav_abs2_2 )**2 + ( z_sys_err/z_sys )**2 - 2*( wav_abs2_2_err/wav_abs2_2 )*( z_sys_err/z_sys )\
		+ wav_abs2_2_err**2 + z_sys_err**2 ) ) )

		vel_abs3_1_err = abs( vel_abs3_1*( np.sqrt( ( wav_abs3_1_err/wav_abs2_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_1_err/wav_abs3_1)*(z_sys_err/z_sys)\
		+ wav_abs3_1_err**2 + z_sys_err**2 ) ) )

		vel_abs3_2_err = abs( vel_abs3_2*( np.sqrt( ( wav_abs3_2_err/wav_abs2_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_2_err/wav_abs3_2)*(z_sys_err/z_sys)\
		+ wav_abs3_2_err**2 + z_sys_err**2 ) ) )

		y = max(flux)

		flux_peak1 		= amp[0]/(wid[0]*np.sqrt(2.*np.pi)) 
		flux_peak1_err	= flux_peak1*np.sqrt( (amp_err[0]/amp[0])**2 + (wid_err[0]/wid[0])**2 )

		flux_peak2 		= amp[1]/(wid[1]*np.sqrt(2.*np.pi)) 
		flux_peak2_err	= flux_peak2*np.sqrt( (amp_err[1]/amp[1])**2 + (wid_err[1]/wid[1])**2 )

	elif spec_feat == 'NV':
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
		Nmax = 1.e20
		
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

		res 		= out.params

		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err2  = wav_o[0]*(res['g_cen1'].stderr/wav_o[1])
		wav_o_err 	= (res['g_cen1'].stderr, wav_o_err2)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err2	= wid[0]*(res['wid1']/wid[1])
		wid_err  	= (res['wid1'].stderr,wid_err2)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err2	= amp[0]*(res['amp1'].stderr/amp[1])
		amp_err 	= (res['amp1'].stderr,amp_err2)

		fwhm1_kms  = convert_sigma_fwhm(wid[0],wid_err[0],wav_o[0],wav_o_err[0])
		fwhm2_kms  = convert_sigma_fwhm(wid[1],wid_err[0],wav_o[1],wav_o_err[1])

		fig = pl.figure(figsize=(10,12))
		ax = pl.gca()

		pl.plot(wav,gauss_nocont(wav,amp[0],wid[0],wav_o[0]),c='blue',ls='--',label=r'NV $\lambda$'+'%.2f'%lam1)
		pl.plot(wav,gauss_nocont(wav,amp[1],wid[1],wav_o[1]),c='orange',ls='--',label=r'NV $\lambda$'+'%.2f'%lam2)

		wav_o1 		= res['g_cen1'].value
		wav_o_err1 	= res['g_cen1'].stderr	

		wav_o2 		= res['g_cen2'].value
		wav_o_err2 	= res['g_cen2'].stderr

		z_abs1 		= res['z1_1'].value
		z_abs2 		= res['z2_1'].value
		z_abs3 		= res['z3_1'].value

		z_abs_err1 	= res['z1_1'].stderr
		z_abs_err2 	= res['z2_1'].stderr
		z_abs_err3 	= res['z3_1'].stderr

		b_abs1 		= res['b1_1'].value
		b_abs2 		= res['b2_1'].value
		b_abs3 		= res['b3_1'].value

		b_abs_err1 	= res['b1_1'].stderr
		b_abs_err2 	= res['b2_1'].stderr
		b_abs_err3 	= res['b3_1'].stderr

		N_abs1_1 	= res['N1_1'].value
		N_abs1_2 	= res['N1_2'].value
		N_abs2_1 	= res['N2_1'].value
		N_abs2_2 	= res['N2_2'].value
		N_abs3_1 	= res['N3_1'].value
		N_abs3_2 	= res['N3_2'].value

		N_abs1_1_err 	= res['N1_1'].stderr
		N_abs1_2_err 	= res['N1_2'].stderr
		N_abs2_1_err 	= res['N2_1'].stderr
		N_abs2_2_err 	= res['N2_2'].stderr
		N_abs3_1_err 	= res['N3_1'].stderr
		N_abs3_2_err 	= res['N3_2'].stderr

		# absorber wavelengths
		wav_abs1_1 = lam1*(1.+z_abs1)
		wav_abs1_2 = lam2*(1.+z_abs1)

		wav_abs2_1 = lam1*(1.+z_abs2)
		wav_abs2_2 = lam2*(1.+z_abs2)

		wav_abs3_1 = lam1*(1.+z_abs3)
		wav_abs3_2 = lam2*(1.+z_abs3)

		wav_abs1_1_err = abs(wav_abs1_1 *( z_abs_err1/z_abs1 ))
		wav_abs1_2_err = abs(wav_abs1_2 *( z_abs_err1/z_abs1 ))

		wav_abs2_1_err = abs(wav_abs2_1 *( z_abs_err2/z_abs2 ))
		wav_abs2_2_err = abs(wav_abs2_2 *( z_abs_err2/z_abs2 ))

		wav_abs3_1_err = abs(wav_abs3_1 *( z_abs_err3/z_abs3 ))
		wav_abs3_2_err = abs(wav_abs3_2 *( z_abs_err3/z_abs3 ))

		vel0_rest = vel_cgs(wav_o_HeII, wav_e_HeII, z_sys)

		# emission velocity
		vel_o1 		= (vel_cgs(wav_o1, lam, z_sys) - vel0_rest)/1.e5
		vel_o2 		= (vel_cgs(wav_o2, lam, z_sys) - vel0_rest)/1.e5

		vel_o1_err 		= abs ( vel_o1*( ( wav_o_err1/wav_o1 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err1/wav_o1)\
			+ wav_o_err1**2 + z_sys_err**2) )
		vel_o2_err 	= abs ( vel_o2*( ( wav_o_err2/wav_o2 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err2/wav_o2)\
			+ wav_o_err2**2 + z_sys_err**2) )

		#velocity shift between doublets
		v_shift = vel_o2 - vel_o1

		# absorber velocities
		vel_abs1_1 = (vel_cgs( wav_abs1_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs1_2 = (vel_cgs( wav_abs1_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs2_1 = (vel_cgs( wav_abs2_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs2_2 = (vel_cgs( wav_abs2_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs3_1 = (vel_cgs( wav_abs3_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs3_2 = (vel_cgs( wav_abs3_2, lam2, z_sys) - vel0_rest)/1.e5


		vel_abs1_1_err = abs( vel_abs1_1*( np.sqrt( ( wav_abs1_1_err/wav_abs1_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs1_1_err/wav_abs1_1)*(z_sys_err/z_sys)\
		+ wav_abs1_1_err**2 + z_sys_err**2 ) ) )

		vel_abs1_2_err = abs( vel_abs1_2*( np.sqrt( ( wav_abs1_2_err/wav_abs1_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs1_2_err/wav_abs1_2)*(z_sys_err/z_sys)\
		+ wav_abs1_2_err**2 + z_sys_err**2 ) ) )

		vel_abs2_1_err = abs( vel_abs2_1*( np.sqrt( ( wav_abs2_1_err/wav_abs2_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs2_1_err/wav_abs2_1)*(z_sys_err/z_sys)\
		+ wav_abs2_1_err**2 + z_sys_err**2 ) ) )

		vel_abs2_2_err = abs( vel_abs2_2*( np.sqrt( ( wav_abs2_2_err/wav_abs2_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs2_2_err/wav_abs2_2)*(z_sys_err/z_sys)\
		+ wav_abs2_2_err**2 + z_sys_err**2 ) ) )

		vel_abs3_1_err = abs( vel_abs3_1*( np.sqrt( ( wav_abs3_1_err/wav_abs2_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_1_err/wav_abs3_1)*(z_sys_err/z_sys)\
		+ wav_abs3_1_err**2 + z_sys_err**2 ) ) )

		vel_abs3_2_err = abs( vel_abs3_2*( np.sqrt( ( wav_abs3_2_err/wav_abs2_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_2_err/wav_abs3_2)*(z_sys_err/z_sys)\
		+ wav_abs3_2_err**2 + z_sys_err**2 ) ) )

		y = max(flux)

		flux_peak1 		= amp[0]/(wid[0]*np.sqrt(2.*np.pi)) 
		flux_peak1_err	= flux_peak1*np.sqrt( (amp_err[0]/amp[0])**2 + (wid_err[0]/wid[0])**2 )

		flux_peak2 		= amp[1]/(wid[1]*np.sqrt(2.*np.pi)) 
		flux_peak2_err	= flux_peak2*np.sqrt( (amp_err[1]/amp[1])**2 + (wid_err[1]/wid[1])**2 )

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
		Nmax = 1.e20

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

		res 		= out.params

		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err2  = wav_o[0]*(res['g_cen1'].stderr/wav_o[1])
		wav_o_err 	= (res['g_cen1'].stderr, wav_o_err2)

		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err2	= wid[0]*(res['wid1']/wid[1])
		wid_err  	= (res['wid1'].stderr,wid_err2)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err2	= amp[0]*(res['amp1'].stderr/amp[1])
		amp_err 	= (res['amp1'].stderr,amp_err2)

		wav_o_ox 	= res['g_cen'].value
		wid_ox 		= res['wid'].value
		amp_ox		= res['amp'].value

		wav_o_ox_err= res['g_cen'].stderr
		wid_ox_err 	= res['wid'].stderr
		amp_ox_err	= res['amp'].stderr

		fwhm1_kms  	= convert_sigma_fwhm(wid[0],wid_err[0],wav_o[0],wav_o_err[0])
		fwhm2_kms  	= convert_sigma_fwhm(wid[1],wid_err[0],wav_o[1],wav_o_err[1])
		fwhm_kms_ox = convert_sigma_fwhm(wid_ox,wid_ox_err,wav_o_ox,wav_o_ox_err)

		fig = pl.figure(figsize=(10,12))
		ax = pl.gca()

		pl.plot(wav,gauss_nocont(wav,amp_ox,wid_ox,wav_o_ox),color='purple',linestyle='--',label=r'OIV]'+`lam_OIV`)
		pl.plot(wav,gauss_nocont(wav,amp[0],wid[0],wav_o[0]),c='blue',ls='--',label=r'SiIV $\lambda$'+'%.2f'%lam1)
		pl.plot(wav,gauss_nocont(wav,amp[1],wid[1],wav_o[1]),c='orange',ls='--',label=r'SiIV $\lambda$'+'%.2f'%lam2)

		wav_o1 		= res['g_cen1'].value
		wav_o_err1 	= res['g_cen1'].stderr	

		wav_o2 		= res['g_cen2'].value
		wav_o_err2 	= res['g_cen2'].stderr

		z_abs1 		= res['z1_1'].value
		z_abs2 		= res['z2_1'].value
		z_abs3 		= res['z3_1'].value

		z_abs_err1 	= res['z1_1'].stderr
		z_abs_err2 	= res['z2_1'].stderr
		z_abs_err3 	= res['z3_1'].stderr

		b_abs1 		= res['b1_1'].value
		b_abs2 		= res['b2_1'].value
		b_abs3 		= res['b3_1'].value

		b_abs_err1 	= res['b1_1'].stderr
		b_abs_err2 	= res['b2_1'].stderr
		b_abs_err3 	= res['b3_1'].stderr

		N_abs1_1 	= res['N1_1'].value
		N_abs1_2 	= res['N1_2'].value
		N_abs2_1 	= res['N2_1'].value
		N_abs2_2 	= res['N2_2'].value
		N_abs3_1 	= res['N3_1'].value
		N_abs3_2 	= res['N3_2'].value

		N_abs1_1_err 	= res['N1_1'].stderr
		N_abs1_2_err 	= res['N1_2'].stderr
		N_abs2_1_err 	= res['N2_1'].stderr
		N_abs2_2_err 	= res['N2_2'].stderr
		N_abs3_1_err 	= res['N3_1'].stderr
		N_abs3_2_err 	= res['N3_2'].stderr

		# absorber wavelengths
		wav_abs1_1 = lam1*(1.+z_abs1)
		wav_abs1_2 = lam2*(1.+z_abs1)

		wav_abs2_1 = lam1*(1.+z_abs2)
		wav_abs2_2 = lam2*(1.+z_abs2)

		wav_abs3_1 = lam1*(1.+z_abs3)
		wav_abs3_2 = lam2*(1.+z_abs3)

		wav_abs1_1_err = abs(wav_abs1_1 *( z_abs_err1/z_abs1 ))
		wav_abs1_2_err = abs(wav_abs1_2 *( z_abs_err1/z_abs1 ))

		wav_abs2_1_err = abs(wav_abs2_1 *( z_abs_err2/z_abs2 ))
		wav_abs2_2_err = abs(wav_abs2_2 *( z_abs_err2/z_abs2 ))

		wav_abs3_1_err = abs(wav_abs3_1 *( z_abs_err3/z_abs3 ))
		wav_abs3_2_err = abs(wav_abs3_2 *( z_abs_err3/z_abs3 ))

		vel0_rest = vel_cgs(wav_o_HeII, wav_e_HeII, z_sys)

		# emission velocity
		vel_o1 		= (vel_cgs(wav_o1, lam, z_sys) - vel0_rest)/1.e5
		vel_o2 		= (vel_cgs(wav_o2, lam, z_sys) - vel0_rest)/1.e5

		vel_o1_err 		= abs ( vel_o1*( ( wav_o_err1/wav_o1 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err1/wav_o1)\
			+ wav_o_err1**2 + z_sys_err**2) )
		vel_o2_err 	= abs ( vel_o2*( ( wav_o_err2/wav_o2 )**2 + ( z_sys_err/z_sys )**2 - 2*(z_sys_err/z_sys)*(wav_o_err2/wav_o2)\
			+ wav_o_err2**2 + z_sys_err**2) )

		#velocity shift between doublets
		v_shift = vel_o2 - vel_o1

		# absorber velocities
		vel_abs1_1 = (vel_cgs( wav_abs1_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs1_2 = (vel_cgs( wav_abs1_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs2_1 = (vel_cgs( wav_abs2_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs2_2 = (vel_cgs( wav_abs2_2, lam2, z_sys) - vel0_rest)/1.e5

		vel_abs3_1 = (vel_cgs( wav_abs3_1, lam1, z_sys) - vel0_rest)/1.e5
		vel_abs3_2 = (vel_cgs( wav_abs3_2, lam2, z_sys) - vel0_rest)/1.e5


		vel_abs1_1_err = abs( vel_abs1_1*( np.sqrt( ( wav_abs1_1_err/wav_abs1_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs1_1_err/wav_abs1_1)*(z_sys_err/z_sys)\
		+ wav_abs1_1_err**2 + z_sys_err**2 ) ) )

		vel_abs1_2_err = abs( vel_abs1_2*( np.sqrt( ( wav_abs1_2_err/wav_abs1_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs1_2_err/wav_abs1_2)*(z_sys_err/z_sys)\
		+ wav_abs1_2_err**2 + z_sys_err**2 ) ) )

		vel_abs2_1_err = abs( vel_abs2_1*( np.sqrt( ( wav_abs2_1_err/wav_abs2_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs2_1_err/wav_abs2_1)*(z_sys_err/z_sys)\
		+ wav_abs2_1_err**2 + z_sys_err**2 ) ) )

		vel_abs2_2_err = abs( vel_abs2_2*( np.sqrt( ( wav_abs2_2_err/wav_abs2_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs2_2_err/wav_abs2_2)*(z_sys_err/z_sys)\
		+ wav_abs2_2_err**2 + z_sys_err**2 ) ) )

		vel_abs3_1_err = abs( vel_abs3_1*( np.sqrt( ( wav_abs3_1_err/wav_abs2_1 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_1_err/wav_abs3_1)*(z_sys_err/z_sys)\
		+ wav_abs3_1_err**2 + z_sys_err**2 ) ) )

		vel_abs3_2_err = abs( vel_abs3_2*( np.sqrt( ( wav_abs3_2_err/wav_abs2_2 )**2 + (z_sys_err/z_sys)**2 - 2*(wav_abs3_2_err/wav_abs3_2)*(z_sys_err/z_sys)\
		+ wav_abs3_2_err**2 + z_sys_err**2 ) ) )


		y = max(flux)

		flux_peak1 		= amp[0]/(wid[0]*np.sqrt(2.*np.pi)) 
		flux_peak1_err	= flux_peak1*np.sqrt( (amp_err[0]/amp[0])**2 + (wid_err[0]/wid[0])**2 )

		flux_peak2 		= amp[1]/(wid[1]*np.sqrt(2.*np.pi)) 
		flux_peak2_err	= flux_peak2*np.sqrt( (amp_err[1]/amp[1])**2 + (wid_err[1]/wid[1])**2 )

		flux_peak_ox 	= amp_ox/(wid_ox*np.sqrt(2.*np.pi)) 
		flux_peak_ox_err= flux_peak_ox*np.sqrt( (amp_ox_err/amp_ox)**2 + (wid_ox_err/wid_ox)**2 ) 

	# absorber labels
	if spec_feat == 'Lya':  
		pl.plot([wav_abs1,wav_abs1],[1.1*y,1.2*y],c='k',ls='-')
		pl.text(wav_abs1,1.22*y,'1', fontsize=16, horizontalalignment='center')

		pl.plot([wav_abs2,wav_abs2],[1.1*y,1.2*y],c='k',ls='-')
		pl.text(wav_abs2,1.22*y,'2', fontsize=16, horizontalalignment='center')

		pl.plot([wav_abs3,wav_abs3],[1.1*y,1.2*y],c='k',ls='-')
		pl.text(wav_abs3,1.22*y,'3', fontsize=16, horizontalalignment='center')

		pl.plot([wav_abs4,wav_abs4],[1.1*y,1.2*y],c='k',ls='-')
		pl.text(wav_abs4,1.22*y,'4', fontsize=16, horizontalalignment='center')

	elif spec_feat == 'CIV':
		pl.plot([wav_abs1_1,wav_abs1_1],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_1,1.22*y,'1',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs1_2,wav_abs1_2],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_2,1.06*y,'1',fontsize=16, horizontalalignment='center', color='orange')
	
		pl.plot([wav_abs2_1,wav_abs2_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_1,1.22*y,'2',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs2_2,wav_abs2_2],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_2,1.06*y,'2',fontsize=16, horizontalalignment='center', color='orange')
	
		pl.plot([wav_abs3_1,wav_abs3_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs3_1,1.22*y,'3',fontsize=14, horizontalalignment='center', color='blue')
		pl.plot([wav_abs3_2,wav_abs3_2],[1.1*y,1.2*y],c='k',ls='-',lw=1)
		pl.text(wav_abs3_2,1.06*y,'3',fontsize=14, horizontalalignment='center', color='orange')

	elif spec_feat == 'NV':
		pl.plot([wav_abs1_1,wav_abs1_1],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_1,1.22*y,'1',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs1_2,wav_abs1_2],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_2,1.06*y,'1',fontsize=16, horizontalalignment='center', color='orange')

		pl.plot([wav_abs2_1,wav_abs2_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_1,1.22*y,'2',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs2_2,wav_abs2_2],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_2,1.06*y,'2',fontsize=16, horizontalalignment='center', color='orange')
	
		pl.plot([wav_abs3_1,wav_abs3_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs3_1,1.22*y,'3',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs3_2,wav_abs3_2],[1.1*y,1.2*y],c='k',ls='-',lw=1)
		pl.text(wav_abs3_2,1.06*y,'3',fontsize=16, horizontalalignment='center', color='orange')

	elif spec_feat == 'SiIV':
		wav_abs3_1 = lam1*(1+z3)
		wav_abs3_2 = lam2*(1+z3)

		pl.plot([wav_abs1_1,wav_abs1_1],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_1,1.22*y,'1',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs1_2,wav_abs1_2],[1.1*y,1.2*y],c='k', ls='-', lw=1)
		pl.text(wav_abs1_2,1.06*y,'1',fontsize=16, horizontalalignment='center', color='orange')

		pl.plot([wav_abs2_1,wav_abs2_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_1,1.22*y,'2',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs2_2,wav_abs2_2],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs2_2,1.06*y,'2',fontsize=16, horizontalalignment='center', color='orange')

		pl.plot([wav_abs3_1,wav_abs3_1],[1.1*y,1.2*y],c='k',ls='-', lw=1)
		pl.text(wav_abs3_1,1.22*y,'3',fontsize=16, horizontalalignment='center', color='blue')
		pl.plot([wav_abs3_2,wav_abs3_2],[1.1*y,1.2*y],c='k',ls='-',lw=1)
		pl.text(wav_abs3_2,1.06*y,'3',fontsize=16, horizontalalignment='center', color='orange')

	# figure customisation
	ax.tick_params(direction='in', right=1, top=1)
	chisqr = r'$\chi^2$: %1.3f' %out.chisqr
	redchisqr = r'$\widetilde{\chi}^2$: %1.4f' %out.redchi
	pl.text(0.15, 0.96, redchisqr, ha='center', va='center', transform=ax.transAxes, fontsize=24)
	pl.fill_between(wav, flux, color='grey', interpolate=True, step='mid')
	pl.plot(wav, flux, drawstyle='steps-mid',c='k')
	
	# smoothing best-fit model
	wav_gen = np.linspace(min(wav),max(wav),num=len(wav))

	if spec_feat == 'Lya':

		voigt = \
		 voigt_profile1(wav_gen, z1, b1, N1)\
		*voigt_profile2(wav_gen, z2, b2, N2)\
		*voigt_profile3(wav_gen, z3, b3, N3)\
		*voigt_profile4(wav_gen, z4, b4, N4)\

		fn = dgauss_nocont(wav_gen, amp_blue_Lya, wid_blue_Lya,\
		 wav_o_blue_Lya, amp_Lya, wid_Lya ,wav_o_Lya)*voigt

	elif spec_feat == 'CIV':

		voigt =\
		voigt_profile1_1(wav_gen, z_abs1, b_abs1, N_abs1_1)\
		*voigt_profile1_2(wav_gen, z_abs1, b_abs1, N_abs1_2)\
		*voigt_profile2_1(wav_gen, z_abs2, b_abs2, N_abs2_1)\
		*voigt_profile2_2(wav_gen, z_abs2, b_abs2, N_abs2_2)\
		*voigt_profile3_1(wav_gen, z_abs3, b_abs3, N_abs3_1)\
		*voigt_profile3_2(wav_gen, z_abs3, b_abs3, N_abs3_2)\

		fn = dgauss_nocont(wav_gen, amp[0], wid[0], wav_o[0], amp[1], wid[1], wav_o[1])\
		*voigt

	elif spec_feat == 'NV':

		voigt = \
		voigt_profile1_1(wav_gen, z_abs1, b_abs1, N_abs1_1)\
		*voigt_profile1_2(wav_gen, z_abs1, b_abs1, N_abs1_2)\
		*voigt_profile2_1(wav_gen, z_abs2, b_abs2, N_abs2_1)\
		*voigt_profile2_2(wav_gen, z_abs2, b_abs2, N_abs2_2)\
		*voigt_profile3_1(wav_gen, z_abs3, b_abs3, N_abs3_1)\
		*voigt_profile3_2(wav_gen, z_abs3, b_abs3, N_abs3_2)

		fn = dgauss_nocont(wav_gen, amp[0], wid[0], wav_o[0], amp[1], wid[1], wav_o[1])\
		*voigt

	elif spec_feat == 'SiIV':

		voigt = \
		voigt_profile1_1(wav_gen, z_abs1, b_abs1, N_abs1_1)\
		*voigt_profile1_2(wav_gen, z_abs1, b_abs1, N_abs1_2)\
		*voigt_profile2_1(wav_gen, z_abs2, b_abs2, N_abs2_1)\
		*voigt_profile2_2(wav_gen, z_abs2, b_abs2, N_abs2_2)\
		*voigt_profile3_1(wav_gen, z_abs3, b_abs3, N_abs3_1)\
		*voigt_profile3_2(wav_gen, z_abs3, b_abs3, N_abs3_2)\

		fn = dgauss_nocont(wav_gen, amp[0], wid[0], wav_o[0], amp[1], wid[1], wav_o[1] )\
		*voigt + gauss_nocont(wav_gen, amp_ox, wid_ox, wav_o_ox )

	else:
		fn = dgauss_nocont(wav_gen,amp_blue,wid_blue,\
			wav_o_blue,amp_HeII,wid_HeII,wav_o_HeII)

	pl.plot(wav_gen, fn, c='red', label='best-fit model')
	pl.legend(loc=1,frameon=False)

	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------	
	#radial velocity (Doppler) [km/s]
	def vel(wav_obs,wav_em,z):
		v = c_kms*((wav_obs/wav_em/(1.+z_sys)) - 1.)
		return v

	vel0_rest = vel(wav_o_HeII, wav_e_HeII, 0.) 
	z_kms = vel0_rest/c_kms
	vel0 = vel(wav_o_HeII,wav_e_HeII,z_kms)

	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat == 'Lya':	
		wav_e 	= 1215.67
	
	elif spec_feat == 'CIV':
		wav_e1 	= 1548.18
		wav_e2 	= 1550.77

	elif spec_feat == 'NV':
		wav_e1	= 1238.82
		wav_e2	= 1242.80

	elif spec_feat == 'SiIV':
		wav_e1  = 1393.75
		wav_e2  = 1402.77

	elif spec_feat == 'HeII':
		wav_e 	= 1640.40

	#convert x-axis units
	if spec_feat == 'Lya':
		vel_meas = vel(wav_o_Lya,wav_e,z_kms)	#central velocity of detected line
		vel_off = vel_meas - vel0				#residual velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z_kms) - vel0 ) )

	elif spec_feat == 'HeII':
		vel_meas = vel(wav_o_HeII,wav_e,z_kms)	#central velocity of detected line
		vel_off = vel_meas - vel0				#residual velocity
	 	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e,z_kms) - vel0 ) )

	else:
		vel_meas = [ vel(wav_o1,wav_e1,z_kms), vel(wav_o2,wav_e2,z_kms) ]	
		vel_off = [ vel_meas[0] - vel0, vel_meas[1] - vel0 ]
		xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z_kms) - vel0) ) 

		ax_up = ax.twiny()
		xticks_up = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e1,z_kms) - vel0) ) 
		ax_up.xaxis.set_major_formatter(xticks_up) 

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
		wav_cent = wav_o_Lya 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy

	elif spec_feat == 'HeII':
		wav_cent = wav_o_HeII 
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   

	else:
		wav_cent = wav_o2
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   

	# ergs / s / cm^2 / Ang -> microJy
	if maxf > 0. and maxf < 1.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		major_yticks = [ flux0, flux1 ]

	elif maxf > 1. and maxf < 3.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,1.e-6)
		flux2 = flux_cgs(wav_cent,2.e-6)
		flux3 = flux_cgs(wav_cent,3.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3 ]

	elif maxf > 3. and maxf < 10.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,2.e-6)
		flux2 = flux_cgs(wav_cent,4.e-6)
		flux3 = flux_cgs(wav_cent,6.e-6)
		flux4 = flux_cgs(wav_cent,8.e-6)
		flux5 = flux_cgs(wav_cent,10.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]

	elif maxf > 10. and maxf < 20.:
		flux0 = flux_cgs(wav_cent,0.)
		flux1 = flux_cgs(wav_cent,5.e-6)
		flux2 = flux_cgs(wav_cent,10.e-6)
		flux3 = flux_cgs(wav_cent,15.e-6)
		flux4 = flux_cgs(wav_cent,20.e-6)
		major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]

	elif maxf > 20. and maxf < 60.:
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
	ax.set_yticks(major_yticks)  	#location of tickmarks	

	# get wavelengths corresponding to offset velocities
	def offset_vel_to_wav(voff):

		v1 = -voff
		v2 = voff

		if spec_feat == 'Lya' or spec_feat == 'HeII':
			wav1 = wav_e*(1.+z_sys)*(1.+(v1/c_kms))
			wav2 = wav_e*(1.+z_sys)*(1.+(v2/c_kms))
			return wav1,wav2

		else:
			wav1 = wav_e2*(1.+z_sys)*(1.+(v1/c_kms))
			wav2 = wav_e2*(1.+z_sys)*(1.+(v2/c_kms))
			return wav1,wav2

	def offset_vel_to_wav_up(voff):

		v1 = -voff
		v2 = voff
		wav1 = wav_e1*(1.+z_sys)*(1.+(v1/c_kms))
		wav2 = wav_e1*(1.+z_sys)*(1.+(v2/c_kms))
		return wav1,wav2

	wav0 	 = offset_vel_to_wav(+0.)
	wavlim1  = offset_vel_to_wav(1000.)
	wavlim2  = offset_vel_to_wav(2000.)
	wavlim3  = offset_vel_to_wav(3000.)
	wavlim35 = offset_vel_to_wav(3500.)
	wavlim4  = offset_vel_to_wav(4000.)

	major_ticks = [ wavlim4[0], wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1], wavlim3[1], wavlim4[1] ]

	ax.set_xticks(major_ticks)

	#x-axis in velocity (km/s)
	if spec_feat in ('CIV', 'NV', 'SiIV'):
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
		ax_up.tick_params(direction='in', top=1)
		ax_up.set_xlim([wavlim35[0] ,wavlim35[1] ])

		for label in ax_up.xaxis.get_majorticklabels():
			label.set_fontsize(22)
	
	#fontsize of ticks
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(22)

	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(22)

	ax.set_xlim([wavlim35[0] ,wavlim35[1] ])

	pl.plot([wavlim4[0],wavlim4[1]],[0.,0.],ls='-',color='black',zorder=-1)	#zero flux density-axis
	yticks = tk.FuncFormatter( lambda x,pos: x*1.e-3 ) 			#cgs flux units
	ax.yaxis.set_major_formatter(yticks)

	if spec_feat == 'Lya' or spec_feat == 'HeII':
		ax.set_xlabel( 'Velocity (km/s)'+r', $\lambda_{0}$:'+`lam`+r'\AA', fontsize=22 ) 

	else:
		lam1_label = '%d'%lam1
		lam2_label = '%d'%lam2
		ax.set_xlabel( 'Velocity (km/s)'+r', $\lambda_{0}$:'+lam2_label+r'\AA', fontsize=22 )
		ax_up.set_xlabel( 'Velocity (km/s)'+r', $\lambda_{0}$:'+lam1_label+r'\AA', fontsize=22 )

	ax.set_ylabel(r'Flux Density (10$^{-17}$ erg/s/cm$^2$/$\AA$)', fontsize=22)

	yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6) ) #uJy flux units
	ax.yaxis.set_major_formatter(yticks)
	ax.set_yticks(major_yticks)  	#location of tickmarks	
	
	ax.set_ylabel( r'Flux Density ($\mu$Jy)', fontsize=22 )
	if spec_feat == 'HeII':
		ax.set_ylim([-0.05*y,1.3*y])
	else:
		ax.set_ylim([-0.05*y,1.5*y])

	pl.savefig('out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_fit_cs_0.7.png')
	pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_fit.pdf')

	# -----------------------------------------------
	# 	   Write Gaussian parameters to file
	# -----------------------------------------------
	
	fname_gauss = "./out/line-fitting/4_component_Lya_plus_blue_comp/0943_gaussian_fits.txt"

	if spec_feat 	== 'Lya':
		wav_e = lam

		header = np.array( \
			["#feat wav_rest wav0(Ang)          wav0_err       flux_peak(10^{-20})     flux_peak_err	sigma(Ang)    	   sigma_err   	  fwhm(km/s)   		fwhm_err"] )

		gauss_fit_params = np.array(['%s'%spec_feat, '%.2f'%wav_e, '%.2f'%wav_o_Lya, '%.2f'%wav_o_Lya_err, '%.2f'%flux_peak, '%.2f'%flux_peak_err,\
		 '%.2f'%wid_Lya, '%.2f'%wid_Lya_err, '%.2f'%fwhm_kms_Lya[0], '%.2f'%fwhm_kms_Lya[1]])

		gauss_blue_params = np.array(['%s'%spec_feat+'_blue', '%.2f'%wav_e, '%.2f'%wav_o_blue_Lya, '%.2f'%wav_o_blue_Lya_err, '%.2f'%flux_peak_blue, '%.2f'%flux_peak_blue_err,\
		 '%.2f'%wid_blue_Lya, '%.2f'%wid_blue_Lya_err, '%.2f'%fwhm_kms_blue_Lya[0], '%.2f'%fwhm_kms_blue_Lya[1]])

		with open(fname_gauss,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params,\
				gauss_blue_params)) )
			f.write('\n')

	elif spec_feat 	== 'NV':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array(['%s'%spec_feat, '%.2f'%wav_e1, '%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%flux_peak1, '%.2f'%flux_peak1_err, '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm1_kms[0], '%.2f'%fwhm1_kms[1]])

		gauss_fit_params2 = np.array(['%s'%spec_feat, '%.2f'%wav_e2, '%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%flux_peak2, '%.2f'%flux_peak2_err, '%.2f'%wid[1],\
			'%.2f'%wid_err[0], '%.2f'%fwhm2_kms[0], '%.2f'%fwhm2_kms[1] ])

		with open(fname_gauss,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')

	elif spec_feat 	== 'SiIV':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array(['%s'%spec_feat, '%.2f'%wav_e1, '%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%flux_peak1, '%.2f'%flux_peak1_err, '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm1_kms[0], '%.2f'%fwhm1_kms[1]])

		gauss_fit_params2 = np.array(['%s'%spec_feat, '%.2f'%wav_e2, '%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%flux_peak2, '%.2f'%flux_peak2_err, '%.2f'%wid[1],\
			'%.2f'%wid_err[0], '%.2f'%fwhm2_kms[0], '%.2f'%fwhm2_kms[1] ])

		gauss_fit_OIV = np.array(['OIV]', '%.2f'%lam_OIV, '%.2f'%wav_o_ox, '%.2f'%wav_o_ox_err,\
			'%.2f'%flux_peak_ox, '%.2f'%flux_peak_ox_err, '%.2f'%wid_ox,\
			'%.2f'%wid_ox_err, '%.2f'%fwhm_kms_ox[0], '%.2f'%fwhm_kms_ox[1] ])

		with open(fname_gauss,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2, gauss_fit_OIV)) )
			f.write('\n')

	elif spec_feat 	== 'CIV':
		wav_e1 = lam1
		wav_e2 = lam2

		gauss_fit_params1 = np.array(['%s'%spec_feat, '%.2f'%wav_e1, '%.2f'%wav_o[0], '%.2f'%wav_o_err[0],\
			'%.2f'%flux_peak1, '%.2f'%flux_peak1_err, '%.2f'%wid[0],\
			'%.2f'%wid_err[0], '%.2f'%fwhm1_kms[0], '%.2f'%fwhm1_kms[1]])

		gauss_fit_params2 = np.array(['%s'%spec_feat, '%.2f'%wav_e2, '%.2f'%wav_o[1], '%.2f'%wav_o_err[1],\
			'%.2f'%flux_peak2, '%.2f'%flux_peak2_err, '%.2f'%wid[1],\
			'%.2f'%wid_err[0], '%.2f'%fwhm2_kms[0], '%.2f'%fwhm2_kms[1] ])

		with open(fname_gauss,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
				gauss_fit_params2)) )
			f.write('\n')
	
	elif spec_feat 	== 'HeII':
		wav_e = lam

		gauss_fit_params = np.array(['%s'%spec_feat, '%.2f'%wav_e, '%.2f'%wav_o_HeII, '%.2f'%wav_o_err_HeII, \
			'%.2f'%flux_peak, '%.2f'%flux_peak_err,\
		 '%.2f'%wid_HeII, '%.2f'%wid_HeII_err, '%.2f'%fwhm_kms_HeII[0], '%.2f'%fwhm_kms_HeII[1]])

		gauss_blue_params = np.array(['%s'%spec_feat+'_blue', '%.2f'%wav_e, '%.2f'%wav_o_blue, '%.2f'%wav_o_blue_err, \
			'%.2f'%flux_peak_blue, '%.2f'%flux_peak_blue_err,\
		 '%.2f'%wid_blue, '%.2f'%wid_blue_err, '%.2f'%fwhm_kms_blue_HeII[0], '%.2f'%fwhm_kms_blue_HeII[1]])

		with open(fname_gauss,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params,\
				gauss_blue_params)) )
			f.write('\n')
	
	# -----------------------------------------
	# 	  Write Voigt parameters to file
	# -----------------------------------------
	fname_voigt = "./out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fits.txt"

	if spec_feat == 'Lya':
		wav_e = lam
		header = np.array( ["#feat wav_rest   z_abs    z_abs_err   wav_abs(Ang)    b(km/s) b_err   N(cm^-2)    N_err   vel_abs(km/s)   vel_abs_err   abs_no"] )

		voigt_fit_params1 = np.array([spec_feat, '%.2f'%wav_e, '%.6f'%z1, '%.6f'%z1_err, '%.2f'%wav_abs1,\
		 '%.2f'%(b1/1.e5), '%.2f'%(b1_err/1.e5), '%.3e'%N1, '%.2e'%N1_err, '%.2f'%vel_abs1, '%.2f'%vel_abs1_err, 1])

		voigt_fit_params2 = np.array([spec_feat, '%.2f'%wav_e, '%.6f'%z2, '%.6f'%z2_err, '%.2f'%wav_abs2, \
			'%.2f'%(b2/1.e5), '%.2f'%(b2_err/1.e5), '%.3e'%N2, '%.2e'%N2_err, '%.2f'%vel_abs2, '%.2f'%vel_abs2_err, 2])

		voigt_fit_params3 = np.array([spec_feat, '%.2f'%wav_e, '%.6f'%z3, '%.6f'%z3_err, '%.2f'%wav_abs3, \
			'%.2f'%(b3/1.e5), '%.2f'%(b3_err/1.e5), '%.3e'%N3, '%.2e'%N3_err, '%.2f'%vel_abs3, '%.2f'%vel_abs3_err, 3])

		voigt_fit_params4 = np.array([spec_feat, '%.2f'%wav_e, '%.6f'%z4, '%.6f'%z4_err, '%.2f'%wav_abs4, \
			'%.2f'%(b4/1.e5), '%.2f'%(b4_err/1.e5), '%.3e'%N4, '%.2e'%N4_err, '%.2f'%vel_abs4, '%.2f'%vel_abs4_err, 4])

		with open(fname_voigt,'w') as f:
			f.write( '  '.join(map(str,header)) ) 
			f.write('\n')
			f.write( '\n'.join('  '.join(map(str,x)) for x in (voigt_fit_params1,\
				voigt_fit_params2, voigt_fit_params3, voigt_fit_params4)) )
			f.write('\n')

	elif spec_feat in ('CIV', 'NV', 'SiIV'):
		wav_e1 = lam1
		wav_e2 = lam2

		voigt_fit_params1_1 = np.array([spec_feat, '%.2f'%wav_e1, '%.6f'%z_abs1, '%.6f'%z_abs_err1, \
			'%.2f'%wav_abs1_1, '%.2f'%(b_abs1/1.e5), '%.2f'%(b_abs_err1/1.e5), '%.3e'%N_abs1_1, '%.2e'%N_abs1_1_err,\
			'%.2f'%vel_abs1_1, '%.2f'%vel_abs1_1_err, 1.1])

		voigt_fit_params1_2 = np.array([spec_feat, '%.2f'%wav_e2, '%.6f'%z_abs1, '%.6f'%z_abs_err1, \
			'%.2f'%wav_abs1_2, '%.2f'%(b_abs1/1.e5), '%.2f'%(b_abs_err1/1.e5), '%.3e'%N_abs1_2, '%.2e'%N_abs1_2_err,\
			'%.2f'%vel_abs1_2, '%.2f'%vel_abs1_2_err, 1.2])

		voigt_fit_params2_1 = np.array([spec_feat, '%.2f'%wav_e1, '%.6f'%z_abs2, '%.6f'%z_abs_err2,\
			'%.2f'%wav_abs2_1, '%.2f'%(b_abs2/1.e5), '%.2f'%(b_abs_err2/1.e5), '%.3e'%N_abs2_1, '%.2e'%N_abs2_1_err,\
			'%.2f'%vel_abs2_1, '%.2f'%vel_abs2_1_err, 2.1])

		voigt_fit_params2_2 = np.array([spec_feat, '%.2f'%wav_e2, '%.6f'%z_abs2, '%.6f'%z_abs_err2,\
			'%.2f'%wav_abs2_2, '%.2f'%(b_abs2/1.e5), '%.2f'%(b_abs_err2/1.e5), '%.3e'%N_abs2_2, '%.2e'%N_abs2_2_err,\
			'%.2f'%vel_abs2_2, '%.2f'%vel_abs2_2_err, 2.2])

		voigt_fit_params3_1 = np.array([spec_feat, '%.2f'%wav_e1, '%.6f'%z_abs3, '%.6f'%z_abs_err3, \
			'%.2f'%wav_abs3_1, '%.2f'%(b_abs3/1.e5), '%.2f'%(b_abs_err3/1.e5), '%.3e'%N_abs3_1, '%.2e'%N_abs3_1_err,\
			'%.2f'%vel_abs3_1, '%.2f'%vel_abs3_1_err, 3.1])

		voigt_fit_params3_2 = np.array([spec_feat, '%.2f'%wav_e2, '%.6f'%z_abs3, '%.6f'%z_abs_err3,\
			'%.2f'%wav_abs3_2, '%.2f'%(b_abs3/1.e5), '%.2f'%(b_abs_err3/1.e5), '%.3e'%N_abs3_2, '%.2e'%N_abs3_2_err,\
			'%.2f'%vel_abs3_2, '%.2f'%vel_abs3_2_err, 3.2])

		with open(fname_voigt,'a') as f:
			f.write( '\n'.join('  '.join(map(str,x)) for x in ( voigt_fit_params1_1, voigt_fit_params1_2,\
				 voigt_fit_params2_1, voigt_fit_params2_2, voigt_fit_params3_1, voigt_fit_params3_2 )) )
			f.write('\n')

#align columns in text file for better readability
res = "./out/line-fitting/4_component_Lya_plus_blue_comp/0943_gaussian_fits.txt"
fit = open("./out/line-fitting/4_component_Lya_plus_blue_comp/0943_gaussian_fit_0.7.txt", "w")

with open(res, 'r') as f:
    for line in f:
        data = line.split()    # Splits on whitespace
        x = '{0[0]:<12}{0[1]:<10}{0[2]:<10}{0[3]:<10}{0[4]:<20}{0[5]:<14}{0[6]:<12}{0[7]:<12}{0[8]:<12}{0[9]:<12}'.format(data)
        fit.write(x[:]+'\n')	

#delete original file
os.system('rm ./out/line-fitting/4_component_Lya_plus_blue_comp/0943_gaussian_fits.txt')

res = "./out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fits.txt"
fit = open("./out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fit_0.7.txt", "w")

with open(res, 'r') as f:
    for line in f:
        data = line.split()    # Splits on whitespace
        x = '{0[0]:<6}{0[1]:<10}{0[2]:<10}{0[3]:<10}{0[4]:<12}{0[5]:<12}{0[6]:<12}{0[7]:<12}{0[8]:<12}{0[9]:<15}{0[10]:<15}{0[11]:<5}'.format(data)
        fit.write(x[:]+'\n')	

#delete original file
os.system('rm ./out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fits.txt')

t_end = (time() - t_start)/60.
print 'Run-time: %f mins'%t_end