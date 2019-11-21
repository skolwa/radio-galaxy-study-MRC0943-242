# S.N. Kolwa 
# ESO (2017)
# MRC0943_NB_imgs.py

# Purpose: 
# - Construct narrow-band images 
# - Rest-frame UV lines detected in  
# 	MUSE in WFM (4800 - 9300 Ang) for z=2.92 source

import matplotlib.pyplot as pl
import numpy as np 

from astropy.io import fits
from astropy.wcs import WCS

import mpdaf.obj as mpdo
import scipy.ndimage as ndimage
import matplotlib.ticker as tk

import warnings
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
import matplotlib.ticker as tk

from itertools import chain 

import sys
from time import time

import astropy.cosmology as ac

from functions import*

pl.rc('text', usetex=True)
pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})

t = time()

img_scale = 'linear'

#ignore those pesky warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ( 'ignore', category=AstropyWarning         )

home = sys.argv[1]

#custom-selected min,max wavelengths
spec_feat 	= ['HeII','HeII', 'HeII', 'CIV','CIII]','CII]','CII','SiIV','OIII]','NIV]','NV']+['Lya']*4 
lam1 		= [6425., 6400., 6445., 6050., 7455., 9096., 5232., 5472., 6532., 5820., 4848., 4732.5, 4754.3, 4768.8, 4778.8]
lam2 		= [6430., 6412., 6450., 6090., 7485., 9146., 5258., 5533., 6536., 5841., 4892., 4747.5, 4758.8, 4776.2, 4793.8]

#radial velocity (Doppler)
def vel(wav_obs,wav_em,z):
	c = 2.9979245800e5					#km/s
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v

#display spectral range on the spectrum plot
#for Lya and HeII, different colour regions on same plot hence arrays below
Lya1 = [lam1[11], lam2[11]]
Lya2 = [lam1[12], lam2[12]]
Lya3 = [lam1[13], lam2[13]]
Lya4 = [lam1[14], lam2[14]]

# HeII
z = 2.923
wav_rest = 1640.4

HeII1 = [vel(lam1[0],wav_rest,z), vel(lam2[0],wav_rest,z)]
HeII2 = [vel(lam1[1],wav_rest,z), vel(lam2[1],wav_rest,z)]
HeII3 = [vel(lam1[2],wav_rest,z), vel(lam2[2],wav_rest,z)]

# define region for extraction
center = (48,46)
radius = 3

for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):

	print spec_feat
	#--------------------------
	# LOAD spectral data cubes
	#--------------------------
	fname = "./out/"+spec_feat+"_cs.fits"
	cube 		= mpdo.Cube(fname,ext=1)

	#---------------
	# LINE EMISSION
	#---------------
	#get WCS data from header
	hdu 			= fits.open(fname)[1]
	wcs 			= WCS(hdu.header).celestial

	pl.figure(figsize=(8,8))
	pl.subplots(1,1)

	# #shift centre
	center1 = (center[0]-1,center[1]-1)
	
	cont_cube = cube.subcube(center1, (2*radius+1), unit_center=None, unit_size=None)
	
	subcube_mask(cont_cube,radius)

	spec = cont_cube.sum(axis=(1,2))

	wav 	=  spec.wave.coord()  		# Ang
	flux 	=  spec.data				#1.e-20 erg / s / cm^2 / Ang
	# pl.plot(wav,flux,c='k',drawstyle='steps-mid')

	p1,p2 			= spec.wave.pixel([lam1,lam2], nearest=True)

	line_em 		= cube[p1:p2+1,10:70,20:90]		#spectral and spatial extraction for narrow-band image

	new_fname 		= "./out/"+spec_feat+"_trunc.fits"
	line_em.write(new_fname)
	hdu 			= fits.open(new_fname)[1]
	wcs 			= WCS(hdu.header).celestial

	line_em_img 	= line_em.sum(axis=0) 		#collapse spectral axis to get image

	ax = pl.gca()

	pl.xlabel(r'Velocity (km s$^{-1}$)', fontsize=15)	
	pl.ylabel(r'Flux Density (10$^{-20}$ erg s$^{-1}$ cm$^{-2}$  $\AA^{-1}$)', fontsize=15)

	for label in ax.xaxis.get_majorticklabels():
		label.set_fontsize(15)

	for label in ax.yaxis.get_majorticklabels():
		label.set_fontsize(15)

	if spec_feat == 'Lya':
		spec.plot(c='k')

		pl.axvspan(Lya1[0],Lya1[1],ymax=0.98,color='blue',alpha=0.2)
		pl.axvspan(Lya2[0],Lya2[1],ymax=0.98,color='green',alpha=0.2)
		pl.axvspan(Lya3[0],Lya3[1],ymax=0.98,color='orange',alpha=0.2)
		pl.axvspan(Lya4[0],Lya4[1],ymax=0.98,color='red',alpha=0.2)

	elif spec_feat == 'HeII':
		N = len(wav)

		wav_arr = [ wav[i] for i in xrange(N) ]
		vel_arr = [ vel(wav_arr[i], wav_rest, z) for i in xrange(N) ]

		pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
		pl.axvspan(HeII1[0],HeII1[1],ymax=0.98,color='green',alpha=0.2)
		pl.axvspan(HeII2[0],HeII2[1],ymax=0.98,color='blue',alpha=0.2)
		pl.axvspan(HeII3[0],HeII3[1],ymax=0.98,color='red',alpha=0.2)
		pl.xlim(-2500.,2500.)
		pl.subplots_adjust(left=0.2,top=0.98)

	else:
		spec.plot(c='k')

		pl.axvspan(lam1,lam2,ymax=0.98,color='magenta',alpha=0.2) 
	
	#------------------
	# get VLA CONTOURS
	#------------------
	vla = fits.open(home+'/DATA/VLA_DATA/0943C.ICLN')[0]
	wcs_vla = WCS(vla.header).celestial
	
	#define contour parameters
	radio_rms 		= 8.e-5
	
	number_contours =	4
	start_sigma		= 	3
	odd_number 		=   1
	
	start_level 	= radio_rms*start_sigma
	contours 		= np.zeros(number_contours)
	contours[0] 	= start_level
	
	for i in range(1,number_contours):
		contours[i] = start_level*(np.sqrt(2))*odd_number
		odd_number	+= 2
å
	vla_arr = vla.data[0,0,:,:]
	
	#save VLA .ICLN file in .fits format
	fits.writeto(home+'/DATA/VLA_DATA/0943C.fits',vla_arr,clobber=True)

	#5arcsec -> kpc conversion
	dl 	  = ac.Planck15.luminosity_distance(2.923)	#Mpc
	z     = 2.923
	D     = 40.		#kpc

	theta = D*(1.+z)**2/(dl.value*1000.)  #radians
	theta_arc = theta*206265.		#arcsec
å
	#----------------------------------------------------------------
	#  OVERLAY VLA contours on Gaussian-smoothed narrow band images
	#----------------------------------------------------------------
	fig = pl.figure(figsize=(12,9))

	line_em_arr = line_em_img.data[:,:]

	ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
	ax.imshow(line_em_arr,origin='lower',interpolation='nearest')

	#transform truncated image to pixel original dimensions
	samp = 0.2  	#arcsec / pixel
	ax.xaxis.set_major_formatter( tk.FuncFormatter( lambda x,pos: '%1.0f'%((x-center[1])*samp) ) ) 
	ax.yaxis.set_major_formatter( tk.FuncFormatter( lambda y,pos: '%1.0f'%((y-center[0])*samp) ) )

	#center nucleus and convert pix to arcsec
	x1,x2 = center[1]-30,center[1]+40
	dx = 2./samp	#tick spacing
	ax.xaxis.set_ticks(np.arange( x1, x2, dx ))  #in pixels

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize('24')
	
	y1,y2 = center[0]-40,center[0]+20
	dy = 2./samp	
	ax.yaxis.set_ticks( np.arange( y1, y2, dy ))  #in pixels

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize('24')

	ax1 = fig.add_axes([0.15, 0.15, 0.7, 0.7], projection=wcs)
	# (x,y) are dimensions of image array
	x = len(line_em_arr)	#dimension 1
	y = len(line_em_arr[0])	#dimension 2

	#convert from erg/s/cm^2/Ang/pix to erg/s/cm^2/Ang/arcsec^2: 1pix = 0.04 arcsec^2
	line_em_arr = [ [line_em_arr[i][j]/0.04e3 for j in range(y)] for i in range(x) ]
	ax1.imshow(line_em_arr,origin='lower',interpolation='nearest')

	if lam1 == 6400.:
		HeII_offset_img = fits.open('./out/extra_fits_files/HeII_offset_img.fits')[0]

		wcs_HeII 		= WCS(HeII_offset_img.header).celestial
		HeII_offset_arr = HeII_offset_img.data[:,:] 

		contours_HeII 		= [ 1.5, 3.1682, 4.83641 ]  #1.e-20 erg/s/cm^2/pix
		contours_HeII_conv 	= [ contours_HeII[i]/0.04e3 for i in xrange(3) ]

		print contours_HeII_conv

		ax1.contour(HeII_offset_arr, levels=contours_HeII, colors='yellow', \
		transform=ax1.get_transform(wcs_HeII), zorder=1)

	#vmin,vmax auto-scale
	pix = list(chain(*line_em_arr))
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = pix_med + pix_rms
	vmin = pix_med - pix_rms
	# print vmin,vmax
	
	ax1.contour(vla_arr,levels=contours, colors='red', transform=ax1.get_transform(wcs_vla),linewidths=0.8)
	pl.annotate(s='', xy=(1.4,1.4), xytext=(15.5,1.4), arrowprops=dict(arrowstyle='->',ec='red'))		#horizontal
	pl.annotate(s='', xy=(15.,15.), xytext=(15.,0.8), arrowprops=dict(arrowstyle='->',ec='red'))		#vertical

	pl.text(14.3,15,'N',color='red', fontsize=26)
	pl.text(-0.3,0.5,'E',color='red', fontsize=26)

	ax.set_xlabel('RA (arcsec)', fontsize=24)
	ax.set_ylabel('DEC (arcsec)', fontsize=24)

	deltax = 20
	deltay = 10
	pl.scatter(46-deltax, 48-deltay, s=300, marker='P', edgecolors='k', c='green' )
	pl.scatter(50-deltax, 46-deltay, s=300, marker='P', edgecolors='k', c='cyan' )

	ax.set_xlim([20,89])
	ax.set_ylim([10,69])

	lon = ax1.coords[0]
	lat = ax1.coords[1]

	lon.set_ticklabel_visible(0)
	lat.set_ticklabel_visible(0)

	lon.set_ticks_visible(0)
	lat.set_ticks_visible(0)
	
	gs_img = ndimage.gaussian_filter(line_em_arr, sigma=(1, 1), order=0)
	figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
		transform=ax1.get_transform(wcs), vmin=vmin, vmax=vmax)

	left, bottom, width, height = ax.get_position().bounds
	cax = fig.add_axes([ 5.5*left, 0.15, width*0.05, height ])
	cb = pl.colorbar(figure, orientation = 'vertical', cax=cax)
	cb.set_label(r'SB (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$)',rotation=90, fontsize=28)
	cb.ax.tick_params(labelsize=22)

	if lam1 == 6425.:  
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
		transform=ax1.get_transform(wcs), vmin=0.4*vmin, vmax=vmax)
		pl.savefig("./out/narrow-band/png/"+spec_feat+"_green_VLA_arcsec.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_green_VLA_arcsec.pdf")
		pl.savefig(home+"/PUBLICATIONS/0943_absorption/plots/"+spec_feat+"_green_VLA_arcsec.pdf")

	elif lam1 == 6400.: 
		pl.savefig("./out/narrow-band/png/"+spec_feat+"_blue_VLA_arcsec.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_blue_VLA_arcsec.pdf")
		pl.savefig(home+"/PUBLICATIONS/0943_absorption/plots/"+spec_feat+"_blue_VLA_arcsec.pdf")

	elif lam1 == 6445.:
		pl.savefig("./out/narrow-band/png/"+spec_feat+"_red_VLA_arcsec.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_red_VLA_arcsec.pdf")
		pl.savefig(home+"/PUBLICATIONS/0943_absorption/plots/"+spec_feat+"_red_VLA_arcsec.pdf")

	elif lam1 == 4732.5:
		pl.savefig("./out/narrow-band/png/"+spec_feat+"1_grey_VLA_arcsec.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"1_VLA_arcsec.pdf")

	elif lam1 == 4754.3:
		pl.savefig("./out/narrow-band/png/"+spec_feat+"2_grey_VLA_arcsec.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"2_VLA_arcsec.pdf")

	elif lam1 == 4768.8:
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs), vmin=vmin, vmax=0.2*vmax)
		pl.savefig("./out/narrow-band/png/"+spec_feat+"3_grey_VLA_arcsec.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"3_VLA_arcsec.pdf")

	elif lam1 == 4778.8:
		pl.savefig("./out/narrow-band/png/"+spec_feat+"4_grey_VLA_arcsec.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"4_VLA_arcsec.pdf")

	else:
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs), vmin=0.2*vmin, vmax=vmax)
		pl.savefig("./out/narrow-band/"+spec_feat+"_grey_VLA_arcsec.pdf")
		pl.savefig("./out/narrow-band/png/"+spec_feat+" grey_VLA_arcsec.png")	

time_elapsed = time() - t
print "build time: %4.2f s" %time_elapsed	
