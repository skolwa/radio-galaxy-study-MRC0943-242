#S.N. Kolwa (2017)
#0943_line_fit.py
# Purpose:  
# - Non-resonant line profile fit i.e. Gaussian and Lorentzian only
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

import sys

#ignore those pesky warnings
warnings.filterwarnings('ignore' , 	category=UserWarning, append=True)
warnings.simplefilter  ('ignore' , 	category=AstropyWarning          )

spec_feat 	= sys.argv[1]
lam1 		= float(sys.argv[2])
lam2 		= float(sys.argv[3])

print spec_feat
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
spec_HeII.plot( color='black' )

ax 		= pl.gca()				
data 	= ax.lines[0]
wav 	= data.get_xdata()			
flux    = data.get_ydata()	

lorenz 		= lm.LorentzianModel(prefix='lorenz_')
pars 		= lorenz.make_params()
pars['lorenz_center'].set(6433.)
pars['lorenz_sigma'].set(10.)

gauss 		= lm.GaussianModel(prefix='gauss_')
pars.update(gauss.make_params())

pars['gauss_center'].set(6420.)
pars['gauss_sigma'].set(15.)

#composite model
mod 	= lorenz + gauss

#model fit
out 	= mod.fit(flux,pars,x=wav)

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

#--------------------------
# LOAD spectral data cubes
#--------------------------
if spec_feat != 'HeII':
	fname 				= "./out/"+spec_feat+".fits"
	cube 				= sc.SpectralCube.read(fname,hdu=1,formt='fits')
	cube.write(fname, overwrite=True)
	cube 				= mpdo.Cube(fname)

# -----------------------------------------------
# 		EXTRACT SPECTRUM for LINE FIT
# -----------------------------------------------
# we leave out CIV and Lya because they have absorption that requires Voigt-fitting
# this will be extended to pixel table spectral extraction
if spec_feat == 'NIV]':
	glx 	= cube.subcube_circle_aperture(center=(52,46),radius=2,\
		unit_center=None	,unit_radius=None)

elsif spec_feat in ('SiIV','CII'):
	glx 	= cube.subcube_circle_aperture(center=(52,46),radius=3,\
		unit_center=None	,unit_radius=None)

elif spec_feat == 'HeII':
	glx		= cube.subcube_circle_aperture(center=(52,46),radius=16,\
		unit_center	=None,unit_radius=None)

elif spec_feat == 'CII]':
	glx 	= cube.subcube_circle_aperture(center=(52,46),radius=7,\
		unit_center=None,unit_radius=None)

elif spec_feat == 'CIII]':
	glx 	= cube.subcube_circle_aperture(center=(52,46),radius=11,\
		unit_center=None,unit_radius=None)

elif spec_feat in ('NV','OIII]'):
	glx 	= cube.subcube_circle_aperture(center=(52,46),radius=6,\
		unit_center=None,unit_radius=None)

pl.figure()
img 			= glx.sum(axis=0)
ax 				= img.plot( scale='arcsinh' )
pl.colorbar(ax,orientation = 'vertical')
pl.savefig('./out/line-fitting/'+spec_feat+'_img.png')

pl.figure()
spec 			= glx.sum(axis=(1,2))

# -----------------------------------
# 	  MODEL FIT to EMISSION LINE
# -----------------------------------
#SINGLET states
spec.plot( color='black' )

ax 			= pl.gca()				
data 		= ax.lines[0]
wav_ax 		= data.get_xdata()			
flux_ax    	= data.get_ydata()	

#Gaussian
if spec_feat in ('CII', 'NIV]','SiIV'): 

	line = spec.gauss_fit(lmin=lam1,lmax=lam2,unit=u.Angstrom,plot=True)

	wav_o 			= line.lpeak
	wav_o_err 		= line.err_lpeak 

	fwhm 			= line.fwhm 
	fwhm_err 		= line.err_fwhm

	height 			= line.peak 
	height_err 		= line.err_peak

#Gaussian plus Lorentzian fit
elif spec_feat == 'HeII': 
	lorenz 		= lm.LorentzianModel(prefix='lorenz_')
	pars['lorenz_center'].set(6433.)
	pars['lorenz_sigma'].set(10.)
	
	gauss 		= lm.GaussianModel(prefix='gauss_')
	pars.update(gauss.make_params())
	
	pars['gauss_center'].set(6420.)
	pars['gauss_sigma'].set(15.)
	
	mod 	= lorenz + gauss
	out 	= mod.fit(flux_ax,pars,x=wav_ax)
	report 	= out.fit_report(min_correl=0.1)

	res = out.params
		
	comps 	= out.eval_components(x=wav)

	pl.plot(wav,comps['lorenz_'],'b--')
	pl.plot(wav,comps['gauss_'],'b--')

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

	height 		= res['height'].value
	height_err	= res['height'].stderr
	
	pl.plot(wav_ax,out.best_fit,'r--')

#DOUBLET states
#double Gaussian
elif spec_feat == 'NV':													
	line = spec.gauss_dfit(lmin=lam1,lmax=lam2,unit=u.Angstrom,plot=\
		True,lpeak_1=4859.8,wratio=1.003230586)

elif spec_feat == 'OIII]':
	line = spec.gauss_dfit(lmin=lam1,lmax=lam2,unit=u.Angstrom,plot=\
		True,lpeak_1=6516.3,wratio=1.003192485)

elif spec_feat == 'CIII]':
	line = spec.gauss_dfit(lmin=lam1,lmax=lam2,unit=u.Angstrom,plot=\
		True,lpeak_1=7480.0,wratio=1.001048933)

pl.fill_between(wav_ax,flux_ax,color='grey',interpolate=True,step='mid')
# pl.savefig('./out/line-fitting/'+spec_feat+' original Fit.eps')

# -----------------------------------------------
# 	   MPDAF GAUSSIAN FIT PARAMETERS
# -----------------------------------------------
# line.flux 	=> integrated flux
# line.fwhm 	=> FHWM
# line.lpeak 	=> central wavelength
# line.cont 	=> continuum flux
# line.peak 	=> peak flux

filename = "./out/line-fitting/0943 fit.txt"

#rest (lab-frame) wavelengths of emission lines
if spec_feat 	== 'HeII':
	wav_e = 1640.4
	wav_o = wav_o_HeII
	gauss_fit_params = np.array([spec_feat, wav_o, wav_o_err_HeII, height_HeII,\
	 height_err_HeII, fwhm_HeII, fwhm_err_HeII, wav_e])
	with open(filename,'a') as f:
		f.write( '  '.join(map(str,gauss_fit_params)) ) 
		f.write('\n')

elif spec_feat 	== 'CII]':
	wav_e = 2326.0
	gauss_fit_params = np.array([spec_feat, wav_o, wav_o_err, height, height_err,\
	 fwhm, fwhm_err, wav_e])
	with open(filename,'a') as f:
		f.write( '  '.join(map(str,gauss_fit_params)) ) 
		f.write('\n')

elif spec_feat 	== 'NIV]':
	wav_e = 1486.5
	gauss_fit_params = np.array([spec_feat, wav_o, wav_o_err, height, height_err,\
	 fwhm, fwhm_err, wav_e])
	with open(filename,'a') as f:
		f.write( '  '.join(map(str,gauss_fit_params)) ) 
		f.write('\n')

elif spec_feat 	== 'SiIV':
	wav_e = 1402.8
	gauss_fit_params = np.array([spec_feat, wav_o, wav_o_err, height, height_err,\
	 fwhm, fwhm_err, wav_e])
	with open(filename,'a') as f:
		f.write( '  '.join(map(str,gauss_fit_params)) ) 
		f.write('\n')

elif spec_feat 	== 'CII':
	wav_e = 1338.0
	gauss_fit_params = np.array([spec_feat, wav_o, wav_o_err, height, height_err,\
	 fwhm, fwhm_err, wav_e])
	with open(filename,'a') as f:
		f.write( '  '.join(map(str,gauss_fit_params)) ) 
		f.write('\n')

elif spec_feat 	== 'NV':
	wav_e1 = 1238.8
	wav_e2 = 1242.8
	header = np.array( \
		["# spec_feat    wav0        err_wav0       flux_peak        err_flux_peak        FWHM    	err_FWHM       wav_rest"] )
	gauss_fit_params1 = np.array([spec_feat, line[0].lpeak, line[0].err_lpeak,\
		line[0].peak, line[0].err_peak, line[0].fwhm,\
		line[0].err_fwhm, wav_e1])
	gauss_fit_params2 = np.array([spec_feat, line[1].lpeak, line[1].err_lpeak,\
		line[1].peak, line[1].err_peak, line[1].fwhm,\
		line[1].err_fwhm, wav_e2])
	with open(filename,'w') as f:
		f.write( '  '.join(map(str,header)) ) 
		f.write('\n')
		f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
			gauss_fit_params2)) )
		f.write('\n')

elif spec_feat 	== 'OIII]':
	wav_e1 = 1660.8
	wav_e2 = 1666.1
	gauss_fit_params1 = np.array([spec_feat, line[0].lpeak, line[0].err_lpeak,\
		line[0].peak, line[0].err_peak, line[0].fwhm,\
		line[0].err_fwhm, wav_e1])
	gauss_fit_params2 = np.array([spec_feat, line[1].lpeak, line[1].err_lpeak,\
		line[1].peak, line[1].err_peak, line[1].fwhm,\
		line[1].err_fwhm, wav_e2])
	with open(filename,'a') as f:
		f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
			gauss_fit_params2)) )
		f.write('\n')

elif spec_feat 	== 'CIII]':
	wav_e1 = 1906.7
	wav_e2 = 1908.7
	gauss_fit_params1 = np.array([spec_feat, line[0].lpeak, line[0].err_lpeak,\
		line[0].peak, line[0].err_peak, line[0].fwhm,\
		line[0].err_fwhm, wav_e1])
	gauss_fit_params2 = np.array([spec_feat, line[1].lpeak, line[1].err_lpeak,\
		line[1].peak, line[1].err_peak, line[1].fwhm,\
		line[1].err_fwhm, wav_e2])
	with open(filename,'a') as f:
		f.write( '\n'.join('  '.join(map(str,x)) for x in (gauss_fit_params1,\
			gauss_fit_params2)) )
		f.write('\n')

# -----------------------------------------------
#    PLOT Velocity-Integrated Line Profiles
# -----------------------------------------------

#precise speed of light in km/s
c   = 299792.458											

#radial velocity (Doppler)
def vel(wav_obs,wav_em,z):
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v

#systemic velocity and redshift based on HeII line
wav_e_HeII = 1640.4
vel0_rest = vel(wav_o_HeII,wav_e_HeII,0.) 	#source frame = observer frame (z=0)
z = vel0_rest/c

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
	vel_meas = [vel(line[0].lpeak,wav_e1,z),vel(line[1].lpeak,wav_e2,z)]	#central velocity of first detected line
	vel_off = [vel_meas[0] - vel0, vel_meas[1] - vel0 ]
	xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( vel(x,wav_e2,z) - vel0)	) 
	# print vel_meas
	# print vel_off[0]														#residual velocity
	# print vel_off[1]

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
else:
	wav_cent = line[1].lpeak
	maxf = flux_Jy(wav_cent,max(flux_ax))*1.e6   #max flux in microJy	

if maxf > 4. and maxf < 5.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,1.e-6)
	flux2 = flux(wav_cent,2.e-6)
	flux3 = flux(wav_cent,3.e-6)
	flux4 = flux(wav_cent,4.e-6)
	flux5 = flux(wav_cent,5.e-6)
	major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]

elif maxf > 1. and maxf < 2.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,1.e-6)
	flux2 = flux(wav_cent,2.e-6)
	major_yticks = [ flux0, flux1, flux2 ]

elif maxf > 20. and maxf < 30.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,10.e-6)
	flux2 = flux(wav_cent,20.e-6)
	flux3 = flux(wav_cent,30.e-6)
	major_yticks = [ flux0, flux1, flux2, flux3, ]

elif maxf > 4. and maxf < 5.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,1.e-6)
	flux2 = flux(wav_cent,2.e-6)
	flux3 = flux(wav_cent,3.e-6)
	flux4 = flux(wav_cent,4.e-6)
	flux5 = flux(wav_cent,5.e-6)
	major_yticks = [ flux0, flux1, flux2, flux3, flux4, flux5 ]

elif maxf > 10. and maxf < 15.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,5.e-6)
	flux2 = flux(wav_cent,10.e-6)
	flux3 = flux(wav_cent,10.e-6)
	flux4 = flux(wav_cent,15.e-6)
	major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]

elif maxf > 30.:
	flux0 = flux(wav_cent,0.)
	flux1 = flux(wav_cent,10.e-6)
	flux2 = flux(wav_cent,20.e-6)
	flux3 = flux(wav_cent,30.e-6)
	flux4 = flux(wav_cent,30.e-6)	
	major_yticks = [ flux0, flux1, flux2, flux3, flux4 ]

ax.set_yticks(major_yticks)

#define y-axis as flux in Jy
if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
	yticks = tk.FuncFormatter( lambda x,pos: '%.0f'%( flux_Jy(wav_cent,x)*1.e6 ) 	)
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
if spec_feat == 'CII':
	ymax = 1.4*max(flux_ax)
	ymin = -10.
elif spec_feat == 'CII]' or spec_feat == 'NIV]' or spec_feat == 'SiIV':
	ymax = 1.3*max(flux_ax)
	ymin = -10.
elif spec_feat in ('NV','OIII]'):
	ymax = 1.2*max(flux_ax)
	ymin = -10.
elif spec_feat == 'HeII' or spec_feat == 'CIII]':
	ymax = 1.2*max(flux_ax)
	ymin = -50.

#draw line representing central velocity of spectral feature
if spec_feat in ('HeII','NIV]','SiIV','CII]','CII'):
	pl.plot([wav_o,wav_o],[ymin,ymax],color='green',ls='--')
else:
	pl.plot([line[0].lpeak,line[0].lpeak],[ymin,ymax],color='green',ls='--')
	pl.plot([line[1].lpeak,line[1].lpeak],[ymin,ymax],color='green',ls='--')

#draw plot
pl.title(spec_feat+' Fit')
ax.set_xlabel( 'Velocity Offset (km/s)' )
ax.set_ylabel( r'Flux Density ($\mu$Jy)' )

# ax.set_ylabel(r'Flux Density (10$^{-20}$ erg / s / cm$^{2}$ / $\AA$)')
ax.set_ylim([ymin,ymax])
ax.set_xlim([xmin,xmax])
pl.plot([xmin,xmax],[0.,0.],ls='--',color='grey')	#zero flux density-axis

pl.savefig('./out/line-fitting/'+spec_feat+' Fit.eps')
# pl.show()
print '----------------------------------------------------------'		