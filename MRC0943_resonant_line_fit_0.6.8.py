#S.N. Kolwa (2018)
#MRC0943_resonant_line_fit_0.6.8.py
# Purpose:  
# - Resonant line profile fit i.e. Gaussian and Voigt
# - Flux in uJy 
# - Wavelength axis in velocity offset w.r.t. HeII (systemic velocity)

# - fits 4 Lya absorption profiles
# - fits 3 CIV, NV absorbers
# - double gaussian to account for blueshifted source
# - absorbers are in 2sigma agreement with Lya absorber along L.O.S. 
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

t_start = time()

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat = [ 'Lya', 'CIV', 'NV' ]

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

report_fit(out)

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
	('g_cen1', g_blue, True, g_blue-1., g_blue+1.),\
	('amp1', 2.e3, True, 0.),\
	('wid1', 12., True, 0.),\
	('g_cen2', g_cen, True, g_cen-1., g_cen+1.),\
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
res = mini.emcee(params=pars,steps=2000,thin=10,burn=200)

figure = corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/HeII_corner_plot_0.6.8.png')

amp_blue		= out.params['amp1'].value
wid_blue		= out.params['wid1'].value
g_blue 			= out.params['g_cen1'].value

z_blue 			= g_blue/wav_e_HeII - 1.   #redshift of blue absorber

amp_HeII		= out.params['amp2'].value
wav_o_HeII 		= out.params['g_cen2'].value
wav_o_err_HeII 	= out.params['g_cen2'].stderr
amp 			= out.params['amp2'].value
amp_err			= out.params['amp2'].stderr
wid_HeII 		= out.params['wid2'].value
wid_err			= out.params['wid2'].stderr

delg 			= wav_o_err_HeII 		#error on gaussian centre for all lines

# display HeII spectrum
x = np.linspace( min(wav), max(wav), num=2*len(wav) )
pl.plot(x, dgauss_nocont(x, amp_blue, wid_blue, g_blue, amp_HeII, wid_HeII, wav_o_HeII),'r', label='model')
pl.plot(wav, flux, drawstyle='steps-mid', color='k')
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/HeII_fit_prior_0.6.8.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/HeII_fit_prior_0.6.8.pdf')

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
    """ The H(a, u) function of Tepper Garcia 2006
    """
    P = x ** 2
    H0 = sp.e ** (-(x ** 2))
    Q = 1.5 / x ** 2
    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
    return H

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
	return voigt	

def voigt_profile2(x,z2,b2,N2):	#absorber 2			
	nu 		= c / ( (x / (1.+z2) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b2
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile3(x,z3,b3,N3):	#absorber 3 			
	nu 		= c / ( (x / (1.+z3) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b3
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N3 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile4(x,z4,b4,N4):	#absorber 4			
	nu 		= c / ( (x / (1.+z4) ) * 1.e-8 )	
	nud     = ( nu0 / c ) * b4
	a 		= gamma / ( 4. * np.pi * nud )
	u 		= ( nu - nu0 ) / nud
	tau 	= N4 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

# --------------------------------------------
#     Voigt models for CIV, NV (doublets)
# --------------------------------------------

def voigt_profile1_1(x,z1_1,b1_1,N1_1):  	#absorber 1 at wav1		
	nu 		= c / ( (x / (1.+z1_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b1_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N1_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile1_2(x,z1_2,b1_2,N1_2): 	#absorber 1 at wav2		
	nu 		= c / ( (x / (1.+z1_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b1_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N1_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile2_1(x,z2_1,b2_1,N2_1):  	#absorber 2 at wav1		
	nu 		= c / ( (x / (1.+z2_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b2_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N2_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile2_2(x,z2_2,b2_2,N2_2): 	#absorber 2 at wav2 			
	nu 		= c / ( (x / (1.+z2_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b2_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N2_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	

def voigt_profile3_1(x,z3_1,b3_1,N3_1):  	#absorber 2 at wav1		
	nu 		= c / ( (x / (1.+z3_1)) * 1.e-8 )	
	nud     = ( nu01 / c ) * b3_1
	a 		= gamma1 / ( 4. * np.pi * nud )
	u 		= ( nu - nu01 ) / nud
	tau 	= N3_1 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f1 / ( nud * me * c )
	return np.exp( -tau )

def voigt_profile3_2(x,z3_2,b3_2,N3_2): 	#absorber 3 at wav2 			
	nu 		= c / ( (x / (1.+z3_2)) * 1.e-8 )	
	nud     = ( nu02 / c ) * b3_2
	a 		= gamma2 / ( 4. * np.pi * nud )
	u 		= ( nu - nu02 ) / nud
	tau 	= N3_2 * np.sqrt(np.pi) * e**2 * H(a,abs(u)) * f2 / ( nud * me * c )
	voigt   = np.exp( -tau )
	return voigt	
	
#radial velocity (Doppler) [cm/s]
def vel(wav_obs,wav_em,z):
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v
	
vel0 = vel(wav_o_HeII,wav_e_HeII,0.) 	# source frame = obs frame 
z = vel0/c 								# systemic redshift
z_err 	= ( wav_o_err_HeII/wav_o_HeII )*z

print 'Systemic Redshift: %2.6f +/- %2.6f' %(z,z_err)

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
pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))

mini = Minimizer(str_line_model, pars, nan_policy='omit')
out = mini.minimize(method='leastsq')

report_fit(out)

grad = out.params['grad'].value
cut = out.params['cut'].value
cont = str_line(wav_mask,grad,cut)

# continuum subtract from copy of spectrum
spec_Lya_sub = spec_Lya - cont

wav 	=  spec_Lya_sub.wave.coord()  		#1.e-8 cm
flux 	=  spec_Lya_sub.data				#1.e-20 erg / s / cm^2 / Ang

f 			= 0.4162				#HI oscillator strength
lam 		= 1215.57				#rest wavelength of Lya
gamma 		= 6.265e8				#gamma of HI line
nu0			= c / (lam*1.e-8) 		#Hz 

#initial guesses (based on literature values)
pars = Parameters()

z1 = 2.9070
z2 = 2.9190
z3 = 2.9276
z4 = 2.9328

g_cen 	= lam*(1.+z)
g_blue 	= lam*(1.+z_blue)
bmin = 36.7e5
bmax = 2000.e5
Nmin = 1.e11

pars.add_many( ('amp1' , 8*amp_blue, True, 0. ), ('wid1', wid_blue, True, 0.), ('g_cen1', g_blue, True, g_blue-delg, g_blue+delg),\
	('amp2' , 8*amp_HeII, True, 0. ), ('wid2', wid_HeII, True, 0.), ('g_cen2', g_cen, True, g_cen-delg, g_cen+delg),\
 ('z1', z1, True, z1-0.001, z1+0.001), ('b1', 90.e5, True, bmin, bmax ), ('N1', 1.e14, True, Nmin),
 ('z2', z2, True, z2-0.001, z2+0.001), ('b2', 60.e5, True, bmin, bmax ), ('N2', 1.e19, True, Nmin), 
 ('z3', z3, True, z3-0.001, z3+0.001), ('b3', 110.e5, True, bmin, bmax ), ('N3', 1.e14, True, Nmin), 
 ('z4', z4, True, z4-0.001, z4+0.001), ('b4', 50.e5, True, bmin, bmax ), ('N4', 1.e13, True, Nmin) )

def four_abs_model(p):
	x = wav
	mod = dgauss_nocont(x,p['amp1'],p['wid1'],p['g_cen1'],p['amp2'],p['wid2'],p['g_cen2'])\
	*voigt_profile1(x,p['z1'], p['b1'], p['N1'])*voigt_profile2(x,p['z2'], p['b2'], p['N2'])\
	*voigt_profile3(x,p['z3'], p['b3'], p['N3'])*voigt_profile4(x,p['z4'], p['b4'], p['N4'])
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

report_fit(out)

pl.figure(figsize=(12,12))
res = mini.emcee(params=pars,steps=2000,thin=10,burn=200)

figure = corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/Lya_corner_plot_0.6.8.png')

#fit parameters to pass down
amp_blue	= out.params['amp1'].value
wid_blue 	= out.params['wid1'].value
wav_o_blue 	= out.params['g_cen1'].value

amp_Lya		= out.params['amp2'].value
wid_Lya 	= out.params['wid2'].value
wav_o_Lya 	= out.params['g_cen2'].value

N1			= out.params['N1'].value
z1 			= out.params['z1'].value
b1			= out.params['b1'].value

N2			= out.params['N2'].value
z2 			= out.params['z2'].value
b2			= out.params['b2'].value
	
N3			= out.params['N3'].value
z3 			= out.params['z3'].value
b3			= out.params['b3'].value
	
N4			= out.params['N4'].value
z4 			= out.params['z4'].value
b4			= out.params['b4'].value

# # display Lya spectrum
# x = np.linspace( min(wav), max(wav), num=2*len(wav) )

# fn = dgauss(x, amp_blue, wid_blue, wav_o_blue, amp_Lya, wid_Lya, wav_o_Lya, cont)\
# *voigt_profile1(x, z1, b1, N1)*voigt_profile2(x, z2, b2, N2)\
# *voigt_profile3(x, z3, b3, N3)*voigt_profile4(x, z4, b4, N4)

# pl.plot(x, fn,'r', label='model')
# pl.plot(wav, flux, drawstyle='steps-mid', color='k')
# pl.show()

# -------------------------
#   Resonant line Fitting 
# -------------------------
# Lya (absorption model) and HeII (emission model) fit params as initial guesses in resonant line fitting
# Atomic constants taken from Cashman et al (2017)

for spec_feat in spec_feat:

	if spec_feat == 'Lya':
		f 			= 0.4162				
		lam 		= 1215.57
		gamma 		= 6.265e8				
		nu0			= c / (lam*1.e-8) 		
	
	elif spec_feat == 'CIV':
		f1 			= 0.0190
		f2 			= 0.0948
		gamma1 		= 2.69e8				
		gamma2  	= 2.70e8
		lam1		= 1548.20 
		lam2		= 1550.77
		nu01		= c / (lam1*1.e-8)		
		nu02		= c / (lam2*1.e-8) 
	
	elif spec_feat == 'NV':
		f1 			= 0.0777
		f2 			= 0.0156
		gamma1 		= 3.37e8				
		gamma2  	= 3.40e8
		lam1		= 1238.88
		lam2		= 1242.80
		nu01		= c / (lam1*1.e-8)
		nu02		= c / (lam2*1.e-8) 
	
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

	if spec_feat == 'Lya':
		#mask non-continuum lines 
		spec_copy.mask_region(4720.,4815.)	#Lya line
		spec_copy.mask_region(4840.,4912.)	#NV line 
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		wav_abs1 = lam*(1.+z1)
		wav_abs2 = lam*(1.+z2)
		wav_abs3 = lam*(1.+z3)
		wav_abs4 = lam*(1.+z4)

		fig = pl.figure(figsize=(10,10))
		ax = pl.gca()
		pl.plot(wav,gauss_nocont(wav,amp_blue, wid_blue, wav_o_blue),color='blue',linestyle='--',label=r'blueshifted component')
		pl.plot(wav,gauss_nocont(wav,amp_Lya, wid_Lya, wav_o_Lya),color='orange',linestyle='--',label=r'Lya $\lambda$'+`lam`+r'$\AA$')
		# pl.plot(wav,str_line(wav,grad,cut),label='continuum_fit',c='purple')
		y = max(flux)

	elif spec_feat == 'CIV':
		#mask non-continuum lines 
		spec_copy.mask_region(6034.,6119.)	#CIV line
		spec_copy.mask_region(6299.,6303.)	#sky line  (?)
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		# initial guesses for line centres
		g_cen1 = lam1*(1.+z)
		g_cen2 = lam2*(1.+z)

		#HeII gaussian width (km/s)
		v = c_kms * ( wid_HeII / wav_o_HeII ) 

		#initial guess Gaussian width for LyaV lines
		guess_wid1 = g_cen1 * ( v / c_kms )
		guess_wid2 = g_cen2 * ( v / c_kms )

		pars = Parameters()

		pars.add('amp1', 1.2*amp_HeII, True, 0.)
		pars.add('amp2', expr='0.5*amp1')

		pars.add('wid1', guess_wid1, True, 0.)
		pars.add('wid2', expr='wid1')

		pars.add('g_cen1', g_cen1, True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2', expr='1.0016612819257436*g_cen1') #from ratio of rest-frame doublet wavelengths

		deltaz = 0.0006
		bmin   = 36.7e5
		bmax   = 2000.e5
		Nmin   = 1.e11

		# absorber 1
		pars.add('z1_1', z1,True, z1-deltaz, z1+deltaz)
		pars.add('z1_2',expr='z1_1')

		pars.add('N1_1',1.e12,True,Nmin)
		pars.add('N1_2',1.e14,True,Nmin)

		pars.add('b1_1',b1,True,bmin,bmax)
		pars.add('b1_2',expr='b1_1')

		# absorber 2
		pars.add('z2_1',z2,True,z2-deltaz,z2+deltaz)
		pars.add('z2_2',expr='z2_1')

		pars.add('N2_1',1.e14,True,Nmin)
		pars.add('N2_2',1.e14,True,Nmin)

		pars.add('b2_1',b2,True,bmin,bmax)
		pars.add('b2_2',expr='b2_1')

		# absorber 3
		pars.add('z3_1',z3,True,z3-deltaz,z3+deltaz)
		pars.add('z3_2',expr='z3_1')

		pars.add('N3_1',1.e12,True,Nmin)
		pars.add('N3_2',1.e13,True,Nmin)

		pars.add('b3_1',b2,True,bmin,bmax)
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
		res = mini.emcee(params=pars,steps=2000,thin=10,burn=200)
		
		figure = corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
		pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/CIV_corner_plot_0.6.8.png')
		
		# mod 	= Model(dgauss)*Model(voigt_profile1_1)*Model(voigt_profile1_2)*\
		# Model(voigt_profile2_1)*Model(voigt_profile2_2)

		# fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		# print fit.fit_report()

		res 		= out.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)
	
		fig = pl.figure(figsize=(10,10))
		ax = pl.gca()
		pl.plot(wav,gauss_nocont(wav,amp[0],wid[0],wav_o[0]),color='blue',linestyle='--',label=r'CIV $\lambda$'+`lam1`+r'$\AA$')
		pl.plot(wav,gauss_nocont(wav,amp[1],wid[1],wav_o[1]),color='orange',linestyle='--',label=r'CIV $\lambda$'+`lam2`+r'$\AA$')

		wav_o1 		= res['g_cen1'].value
		wav_o_err1 	= res['g_cen1'].stderr	

		wav_o2 		= res['g_cen2'].value
		wav_o_err2 	= res['g_cen2'].stderr

		z_abs1 		= out.params['z1_1'].value
		z_abs2 		= out.params['z2_1'].value
		z_abs3 		= out.params['z3_1'].value


		b_abs1 		= out.params['b1_1'].value
		b_abs2 		= out.params['b2_1'].value
		b_abs3 		= out.params['b3_1'].value


		N_abs1_1 		= out.params['N1_1'].value
		N_abs1_2 		= out.params['N1_2'].value
		N_abs2_1 		= out.params['N2_1'].value
		N_abs2_2 		= out.params['N2_2'].value
		N_abs3_1 		= out.params['N3_1'].value
		N_abs3_2 		= out.params['N3_2'].value


		wav_abs1_1 = lam1*(1.+z_abs1)
		wav_abs1_2 = lam2*(1.+z_abs1)

		wav_abs2_1 = lam1*(1.+z_abs2)
		wav_abs2_2 = lam2*(1.+z_abs2)

		wav_abs3_1 = lam1*(1.+z_abs3)
		wav_abs3_2 = lam2*(1.+z_abs3)

		y = max(flux)

	elif spec_feat == 'NV':
		#mask non-continuum lines 
		spec_copy.mask_region(4838.,4914.)	#NV line
		
		wav_mask 	=  spec_copy.wave.coord()  		#1.e-8 cm
		flux_mask 	=  spec_copy.data				#1.e-20 erg / s / cm^2 / Ang
		
		pars = Parameters()
		pars.add_many(('grad', 0.01,True), ('cut', 50.,True,))
		
		mini = Minimizer(str_line_model, pars, nan_policy='omit')
		out = mini.minimize(method='leastsq')
		
		report_fit(out)
		
		grad = out.params['grad'].value
		cut = out.params['cut'].value
		cont = str_line(wav_mask,grad,cut)
		
		# continuum subtract from copy of spectrum
		spec_sub = spec - cont
		
		wav 		= spec_sub.wave.coord()		
		flux  		= spec_sub.data

		g_cen1 = lam1*(1.+z)
	 	g_cen2 = lam2*(1.+z)

	 	pars = Parameters()

	 	#HeII gaussian width (km/s)
	 	v = c_kms * ( wid_HeII / wav_o_HeII ) 

	 	#initial guess Gaussian width for NV lines
		guess_wid1 = g_cen1 * ( v / c_kms )
		guess_wid2 = g_cen2 * ( v / c_kms )

		pars.add('g_cen1',g_cen1,True, g_cen1-delg, g_cen1+delg)
		pars.add('g_cen2',expr='1.003228931223765*g_cen1') #from ratio of rest-frame doublet wavelengths

		pars.add('amp1', 0.2*amp_HeII, True, 0.)
		pars.add('amp2',expr='0.5*amp1')

		pars.add('wid1', guess_wid1, True, 0.)
		pars.add('wid2', guess_wid2, expr='wid1')

		deltaz = 0.0006
		bmax = 2000.e5
		Nmin = 1.e11

		# absorber 1
		pars.add('z1_1', z1,True, z1-deltaz, z1+deltaz)
		pars.add('z1_2',expr='z1_1')

		pars.add('N1_1', 1.e14, True, Nmin)
		pars.add('N1_2', 1.e14, True, Nmin)

		pars.add('b1_1', b1,True,bmin,bmax)
		pars.add('b1_2',expr='b1_1')

		# absorber 2
		pars.add('z2_1',z2,True,z2-deltaz,z2+deltaz)
		pars.add('z2_2',expr='z2_1')

		pars.add('N2_1',1.e15,True,Nmin)
		pars.add('N2_2',1.e15,True,Nmin)

		pars.add('b2_1',b2,True,bmin,bmax)
		pars.add('b2_2',expr='b2_1')

		# absorber 3
		pars.add('z3_1',z3,True,z3-deltaz,z3+deltaz)
		pars.add('z3_2',expr='z3_1')

		pars.add('N3_1',1.e14,True,Nmin)
		pars.add('N3_2',1.e14,True,Nmin)

		pars.add('b3_1',b3,True,bmin,bmax)
		pars.add('b3_2',expr='b3_1')

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
		res = mini.emcee(params=pars,steps=2000,thin=10,burn=400)
		
		figure = corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
		pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/NV_corner_plot_0.6.8.png')

		# mod 	= Model(dgauss)*Model(voigt_profile1_1)*Model(voigt_profile1_2)\
		# *Model(voigt_profile2_1)*Model(voigt_profile2_2)\

		# fit 	= mod.fit(flux,pars,x=wav,weights=inv_noise,fit_kws={'nan_policy':'omit'})

		# print fit.fit_report()

		res 		= out.params
		wav_o 		= (res['g_cen1'].value,res['g_cen2'].value)
		wav_o_err 	= (res['g_cen1'].stderr,res['g_cen2'].stderr)
		
		wid 		= (res['wid1'].value,res['wid2'].value)
		wid_err  	= (res['wid1'].stderr,res['wid2'].stderr)
		
		amp 		= (res['amp1'].value,res['amp2'].value)
		amp_err 	= (res['amp1'].stderr,res['amp2'].stderr)

		fig = pl.figure(figsize=(10,10))
		ax = pl.gca()
		pl.plot(wav,gauss_nocont(wav,amp[0],wid[0],wav_o[0]),color='blue',linestyle='--',label=r'NV $\lambda$'+`lam1`+r'$\AA$')
		pl.plot(wav,gauss_nocont(wav,amp[1],wid[1],wav_o[1]),color='orange',linestyle='--',label=r'NV $\lambda$'+`lam2`+r'$\AA$')

		wav_o1 		= out.params['g_cen1'].value
		wav_o_err1 	= out.params['g_cen1'].stderr	

		wav_o2 		= out.params['g_cen2'].value
		wav_o_err2 	= out.params['g_cen2'].stderr

		z_abs1 		= out.params['z1_1'].value
		z_abs2 		= out.params['z2_1'].value
		z_abs3 		= out.params['z3_1'].value


		b_abs1 		= out.params['b1_1'].value
		b_abs2 		= out.params['b2_1'].value
		b_abs3 		= out.params['b3_1'].value


		N_abs1_1 		= out.params['N1_1'].value
		N_abs1_2 		= out.params['N1_2'].value
		N_abs2_1 		= out.params['N2_1'].value
		N_abs2_2 		= out.params['N2_2'].value
		N_abs3_1 		= out.params['N3_1'].value
		N_abs3_2 		= out.params['N3_2'].value


		wav_abs1_1 = lam1*(1.+z_abs1)
		wav_abs1_2 = lam2*(1.+z_abs1)

		wav_abs2_1 = lam1*(1.+z_abs2)
		wav_abs2_2 = lam2*(1.+z_abs2)

		wav_abs3_1 = lam1*(1.+z_abs3)
		wav_abs3_2 = lam2*(1.+z_abs3)

		y = max(flux)

	# absorber labels
	if spec_feat in ('NV', 'CIV'):
		pl.plot([wav_abs1_1,wav_abs1_1],[1.1*y,1.2*y],color='k', ls='-', lw=0.8)
		pl.text(wav_abs1_1,1.22*y,'1',fontsize=10, horizontalalignment='center', color='blue')
		pl.plot([wav_abs1_2,wav_abs1_2],[1.1*y,1.2*y],color='k', ls='-', lw=0.8)
		pl.text(wav_abs1_2,1.06*y,'1',fontsize=10, horizontalalignment='center', color='orange')
	
		pl.plot([wav_abs2_1,wav_abs2_1],[1.1*y,1.2*y],color='k',ls='-', lw=0.8)
		pl.text(wav_abs2_1,1.22*y,'2',fontsize=10, horizontalalignment='center', color='blue')
		pl.plot([wav_abs2_2,wav_abs2_2],[1.1*y,1.2*y],color='k',ls='-', lw=0.8)
		pl.text(wav_abs2_2,1.06*y,'2',fontsize=10, horizontalalignment='center', color='orange')
	
		pl.plot([wav_abs3_1,wav_abs3_1],[1.1*y,1.2*y],color='k',ls='-', lw=0.8)
		pl.text(wav_abs3_1,1.22*y,'3',fontsize=10, horizontalalignment='center', color='blue')
		pl.plot([wav_abs3_2,wav_abs3_2],[1.1*y,1.2*y],color='k',ls='-',lw=0.8)
		pl.text(wav_abs3_2,1.06*y,'3',fontsize=10, horizontalalignment='center', color='orange')
			
	else: #Lya 
		pl.plot([wav_abs1,wav_abs1],[1.1*y,1.2*y],color='k',ls='-')
		pl.text(wav_abs1,1.22*y,'1', fontsize=12, horizontalalignment='center')

		pl.plot([wav_abs2,wav_abs2],[1.1*y,1.2*y],color='k',ls='-')
		pl.text(wav_abs2,1.22*y,'2',fontsize=12, horizontalalignment='center')

		pl.plot([wav_abs3,wav_abs3],[1.1*y,1.2*y],color='k',ls='-')
		pl.text(wav_abs3,1.22*y,'3',fontsize=12, horizontalalignment='center')

		pl.plot([wav_abs4,wav_abs4],[1.1*y,1.2*y],color='k',ls='-')
		pl.text(wav_abs4,1.22*y,'4',fontsize=12, horizontalalignment='center')
	
	chisqr = r'$\chi^2$: %1.3f' %out.chisqr
	redchisqr = r'$\widetilde{\chi}^2$: %1.4f' %out.redchi
	pl.text(0.08, 0.98, redchisqr, ha='center', va='center', transform=ax.transAxes, fontsize=14)
	pl.fill_between(wav,flux,color='grey',interpolate=True,step='mid')
	pl.plot(wav,flux,drawstyle='steps-mid',color='k')

	# From Kristian-Krogager, VoigtFit package (voigt.py -> evaluate_profile function)
	#convolve best-fit voigt with LSF after fitting
	x = np.linspace( min(wav), max(wav), num=2*len(wav) )

	# draw best fit model - twice as smooth as original grid 
	wav_gen = np.linspace(min(wav),max(wav),num=2*len(wav))

	fwhm 	= 2.65
	sigma 	= fwhm/2*np.sqrt(2*np.log(2))
	LSF 	= gaussian( len(wav), sigma )

	LSF 	= LSF/LSF.sum()

	if spec_feat == 'Lya':

		voigt = \
		voigt_profile1(wav_gen, z1, b1, N1)\
		*voigt_profile2(wav_gen, z2, b2, N2)\
		*voigt_profile3(wav_gen, z3, b3, N3)\
		*voigt_profile4(wav_gen, z4, b4, N4)\

		convolved_voigt = fftconvolve(voigt, LSF, 'same')

		fn = dgauss_nocont(wav_gen,amp_blue,wid_blue,wav_o_blue,amp_Lya,wid_Lya,wav_o_Lya)*convolved_voigt

	elif spec_feat in ('NV', 'CIV'):

		voigt = \
		voigt_profile1_1(wav_gen, z_abs1, b_abs1, N_abs1_1)\
		*voigt_profile1_2(wav_gen, z_abs1, b_abs1, N_abs1_2)\
		*voigt_profile2_1(wav_gen, z_abs2, b_abs2, N_abs2_1)\
		*voigt_profile2_2(wav_gen, z_abs2, b_abs2, N_abs2_2)\
		*voigt_profile3_1(wav_gen, z_abs3, b_abs3, N_abs3_1)\
		*voigt_profile3_2(wav_gen, z_abs3, b_abs3, N_abs3_2)

		fn = dgauss_nocont(wav_gen, amp[0], wid[0], wav_o[0], amp[1], wid[1], wav_o[1])\
		*voigt

	pl.plot(wav_gen,fn,c='red', label='best-fit model')
	pl.legend(loc=1)

	# -----------------------------------------------
	#    PLOT Velocity-Integrated Line Profiles
	# -----------------------------------------------	
	#radial velocity (Doppler) [km/s]
	def vel(wav_obs,wav_em,z):
		v = c_kms*((wav_obs/wav_em/(1.+z)) - 1.)
		return v

	vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	

	z_kms = vel0_rest/c_kms

	vel0 = vel(wav_o_HeII,wav_e_HeII,z_kms)

	#residual velocity and velocity offset scale (Ang -> km/s)
	#singlet state
	if spec_feat == 'Lya':	
		wav_e = 1215.57
	
	elif spec_feat == 'CIV':
		wav_e1 	= 1548.202
		wav_e2 	= 1550.774

	elif spec_feat == 'NV':
		wav_e1	= 1238.8
		wav_e2	= 1242.8

	#convert x-axis units
	if spec_feat == 'Lya':
		vel_meas = vel(wav_o_Lya,wav_e,z_kms)	#central velocity of detected line
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

	else:
		wav_cent = wav_o2
		maxf = flux_Jy(wav_cent,max(flux))*1.e6   #max flux in microJy	

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

	elif  maxf > 3. and maxf < 10.:
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

	#get wavelengths corresponding to offset velocities
	def offset_vel_to_wav(voff):
		v1 = -voff
		v2 = voff

		if spec_feat == 'Lya':
			wav1 = wav_e*(1.+z)*(1.+(v1/c_kms))
			wav2 = wav_e*(1.+z)*(1.+(v2/c_kms))
			return wav1,wav2

		else:
			wav1 = wav_e2*(1.+z)*(1.+(v1/c_kms))
			wav2 = wav_e2*(1.+z)*(1.+(v2/c_kms))
			return wav1,wav2

	def offset_vel_to_wav_up(voff):
		v1 = -voff
		v2 = voff
		wav1 = wav_e1*(1.+z)*(1.+(v1/c_kms))
		wav2 = wav_e1*(1.+z)*(1.+(v2/c_kms))
		return wav1,wav2

	# bottom x-axis in velocity (km/s)
	wav0 	= offset_vel_to_wav(+0.)
	wavlim1 = offset_vel_to_wav(1000.)
	wavlim2 = offset_vel_to_wav(2000.)
	wavlim3 = offset_vel_to_wav(3000.)
	wavlim4 = offset_vel_to_wav(4000.)

	major_ticks = [ wavlim4[0], wavlim3[0], wavlim2[0], wavlim1[0], wav0[1],\
	wavlim1[1],  wavlim2[1], wavlim3[1], wavlim4[1] ]

	ax.set_xticks(major_ticks)

	xmin = wavlim4[0] 
	xmax = wavlim4[1]

	#fontsize of ticks
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)

	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(18)

	# top x-axis in velocity (km/s)
	if spec_feat in ('CIV', 'NV'):
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
			label.set_fontsize(18)
		
	ax.set_xlabel( 'Velocity Offset (km/s)', fontsize=18 )
	ax.set_ylabel( r'Flux Density ($\mu$Jy)', fontsize=18 )
	# ax.set_ylabel(r'Flux Density (10$^{-20}$ erg/s/cm$^2$/$\AA$)')
	pl.plot([xmin,xmax],[0.,0.],ls='--',color='black')	#zero flux density-axis
	ax.set_xlim([xmin,xmax])
	pl.savefig('out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_components_cs.png')
	
	# draw plot
	ax.set_ylim([-0.1*y,1.3*y])
	pl.savefig('out/line-fitting/4_component_Lya_plus_blue_comp/'+spec_feat+'_fit_cs.png')
	pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/'+spec_feat+'_fit.pdf')

t_end = (time() - t_start)/60.
print 'Run-time: %f mins'%t_end