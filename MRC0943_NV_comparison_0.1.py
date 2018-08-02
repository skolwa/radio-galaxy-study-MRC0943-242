#S.N. Kolwa (2018)
#MRC0943_NV_comparison.py

# Purpose:  
# - Compare NV QSO systems to our results

import matplotlib.pyplot as pl
import numpy as np
from operator import itemgetter
from itertools import groupby

# data for intervening and associated QSO absorption ( Table B.1 and B.2 in Fechner et al (2009))
direct = '/Users/skolwa/PHD_WORK/catalogues/'
filenames = [ direct+'Fechner_2009_interven_abs_1.txt',
direct+'Fechner_2009_assoc_abs_1.txt' ] 

params = {'legend.fontsize': 14,
          'legend.handlelength': 2}

pl.rcParams.update(params)
pl.rc('text', usetex=True)
pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})

for file in filenames:
	# N(NV) vs N(CIV)
	data_0943 = np.genfromtxt('out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fit_0.7.txt',\
		dtype=[('spec_feat', 'S5'), ('wav0', np.float64), ('z_abs', np.float64), ('z_abs_err', np.float64),\
		 ('wav_abs', np.float64), ('b', np.float64), ('b_err', np.float64), ('N', np.float64)\
		 , ('N_err', np.float64), ('vel_abs', np.float64), ('vel_abs_err', np.float64), ('abs_no', np.float64)])
	
	n = len(data_0943)
	
	N_CIV 			= [ data_0943[i][7] for i in range(n) if data_0943[i][0] == 'CIV']
	N_CIV_err 		= [ data_0943[i][8] for i in range(n) if data_0943[i][0] == 'CIV']
	N_CIV_no 		= [ data_0943[i][11] for i in range(n) if data_0943[i][0] == 'CIV']
	z_abs_CIV		= [ data_0943[i][2] for i in range(n) if data_0943[i][0] == 'CIV']
	z_abs_err_CIV 	= [ data_0943[i][3] for i in range(n) if data_0943[i][0] == 'CIV']

	logN_CIV 		= [ np.log10(N_CIV[i]) for i in range(len(N_CIV)) if  N_CIV_err[i]/N_CIV[i] < 5.]
	logN_CIV_err 	= [ N_CIV_err[i]/N_CIV[i] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] < 5.]
	N_CIV_no 		= [ N_CIV_no[i] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] < 5.]
	z_abs_CIV 		= [ [ z_abs_CIV[i], z_abs_err_CIV[i] ] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] < 5. ]
	
	N_NV 			= [ data_0943[i][7] for i in range(n) if data_0943[i][0] == 'NV']
	N_NV_err 		= [ data_0943[i][8] for i in range(n) if data_0943[i][0] == 'NV']
	N_NV_no 		= [ data_0943[i][11] for i in range(n) if data_0943[i][0] == 'NV']
	z_abs_NV		= [ data_0943[i][2] for i in range(n) if data_0943[i][0] == 'NV']
	z_abs_err_NV 	= [ data_0943[i][3] for i in range(n) if data_0943[i][0] == 'NV']
	
	logN_NV 		= [ np.log10(N_NV[i]) for i in range(len(N_NV)) if N_NV_err[i]/N_NV[i] < 5.]
	logN_NV_err 	= [ N_NV_err[i]/N_NV[i] for i in range(len(N_NV)) if N_NV_err[i]/N_NV[i] < 5. ]
	N_NV_no 		= [ N_NV_no[i] for i in range(len(N_NV)) if N_NV_err[i]/N_NV[i] < 5.]
	z_abs_NV 		= [  [z_abs_NV[i], z_abs_err_NV[i]] for i in range(len(N_NV)) if N_NV_err[i]/N_NV[i] < 5. ]

	pl.figure(figsize=(10,8))
	ax = pl.gca()

	n = len(N_CIV_no)
	m = len(N_NV_no)

	# column densities between doublets are equal 
	# hence we pick the blue wavelength results of each doublet to plot
	for i in range(0,n,2):
		for j in range(0,m,2):
			if N_CIV_no[i] == N_NV_no[j]:
				pl.scatter(logN_CIV[i], logN_NV[j], c='red', label='MRC 0943-242 (MUSE)' if i==2 else "", s=60 )
				pl.text( 0.9985*logN_CIV[i], 1.005*logN_NV[j], `int(N_CIV_no[i])`, fontsize=16 )
				pl.errorbar(logN_CIV[i], logN_NV[j], xerr=logN_CIV_err[i], yerr=logN_NV_err[j], ecolor='red', capsize=5, uplims=True)
				pl.xlabel(r'$\log{\rm{N}_{CIV}}$', fontsize=16)
				pl.ylabel(r'$\log{\rm{N}_{NV}}$', fontsize=16)
	
	ax.tick_params(direction='in', right=1, top=1)
	
	for tick in ax.xaxis.get_major_ticks():
		   tick.label.set_fontsize(18)
		   tick.label.set_family('serif')
	
	for tick in ax.yaxis.get_major_ticks():
		   tick.label.set_fontsize(18)
		   tick.label.set_family('serif')

	abs_data = np.genfromtxt( file,\
	dtype=[ ('QSO', '|S25'), ('z_sys', '<f8'), ('no.', '<i2'), ('ion', '|S8'),\
	 ('v','<f8'), ('verr', '<f8'), ('logN', '<f8'), ('logNerr', '<f8'), ('b', '<f8'), ('berr', '<f8') ] )

	#sort by source
	n = len(abs_data)
	abs_data = sorted(abs_data, key=itemgetter(0))
	
	#group by source (QSO)
	sources = []
	for key,g in groupby( abs_data, lambda x: x[0] ):
		sources.append( list(g))
	
	#group by systemic redshift (z_sys)
	sources_z_grp = []
	for i in range(len(sources)):
		for key,g in groupby( sources[i], lambda x: x[1] ):
			sources_z_grp.append( list(g) )
	
	#groupby absorber (#)
	sources_abs_grp = []
	for i in range(len(sources_z_grp)):
		for key,g in groupby( sources_z_grp[i], lambda x: x[2]):
			sources_abs_grp.append( list(g) )
	
	N =  len(sources_abs_grp)
	
	n = [ [] for _ in range(N) ]
	c = [ [] for _ in range(N) ]
	
	for k in range(N):
	
		for l in range(len(sources_abs_grp[k])):
	
				QSO = sources_abs_grp[k][l][0]
				z_sys = sources_abs_grp[k][l][1]
				abs_no = sources_abs_grp[k][l][2]
				ion = sources_abs_grp[k][l][3]
				logN = sources_abs_grp[k][l][6]
				logNerr = sources_abs_grp[k][l][7]
	
				if ion == 'NV':
					n[k].append( (QSO, z_sys, abs_no,
						ion, logN, logNerr) ) 
				
				elif ion == 'CIV':
					c[k].append( (QSO, z_sys, abs_no,
						ion, logN, logNerr) )
	c_index = []
	n_index = []
	
	for i in range(N):
		if c[i] == []:
			c_index.append(i)
	
		elif n[i] == []:
			n_index.append(i)
	
	c = np.delete(c, c_index)
	n = np.delete(n, n_index)
	
	# print c
	# print '----'
	# print n
	
	for i in range(len(n)):
		for j in range(len(c)):

			#second CIV detections
			if ( len(c[j]) > 1. and (n[i][0][0] == c[j][1][0] and n[i][0][1] == c[j][1][1] and n[i][0][2] == c[j][1][2])):
				pl.scatter( c[j][1][4], n[i][0][4], c='blue')
				pl.errorbar( c[j][1][4], n[i][0][4], xerr=c[j][1][5], yerr=n[i][0][5], capsize=6, c='blue', zorder=0)

			#all primary detections
			elif ( (n[i][0][0] == c[j][0][0] and n[i][0][1] == c[j][0][1] and n[i][0][2] == c[j][0][2]) ):
				pl.scatter( c[j][0][4], n[i][0][4], c='grey', label='Fechner et al (2009)' if i==1 else "")
				pl.errorbar( c[j][0][4], n[i][0][4], xerr=c[j][0][5], yerr=n[i][0][5], capsize=6, c='grey', zorder=10)


	if file == direct+'Fechner_2009_interven_abs_1.txt':	
		pl.text( 0.04, 0.92, 'intervening', transform=ax.transAxes, fontsize=16)
		pl.legend(loc=4)

		pl.savefig('out/QSO_vs_HzRGs/intervening_QSO_HzRGs.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/intervening_QSO_HzRGs.pdf')


	elif file == direct+'Fechner_2009_assoc_abs_1.txt':
		pl.text( 0.04, 0.92, 'associated', transform=ax.transAxes, fontsize=16)
		pl.legend(loc=4)

		pl.savefig('out/QSO_vs_HzRGs/associated_QSO_HzRGs.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/associated_QSO_HzRGs.pdf')

	# N(NV)/N(CIV) vs redshift
	pl.figure(figsize=(10,8))
	ax = pl.gca()

	ax.tick_params(direction='in', right=1, top=1)
	
	for tick in ax.xaxis.get_major_ticks():
		   tick.label.set_fontsize(18)
		   tick.label.set_family('serif')
	
	for tick in ax.yaxis.get_major_ticks():
		   tick.label.set_fontsize(18)
		   tick.label.set_family('serif')

	p = len(N_CIV_no)
	q = len(N_NV_no)

	for i in range(0,p,2):
		for j in range(0,q,2):
			if N_CIV_no[i] == N_NV_no[j]:
				z_abs = (z_abs_CIV[i][0] + z_abs_NV[j][0])/2. 			#average of absorber redshifts in NV and CIV
				z_abs_err = np.sqrt( (z_abs_CIV[i][1])**2 + (z_abs_NV[j][1])**2 )  
				ratio = logN_NV[j]/logN_CIV[i]
				pl.scatter( z_abs, ratio, c='red', label='MRC 0943-242 (MUSE)' if i==2 else "", s=60 )
				yerr = ratio*np.sqrt( (logN_CIV_err[i]/logN_CIV[i])**2 + (logN_NV_err[j]/logN_NV[j])**2 )
				pl.errorbar( z_abs, ratio, c='red', capsize=5, xerr=z_abs_err, yerr=yerr, uplims=True )
				if file == '/Users/skolwa/PHD_WORK/catalogues/Fechner_2009_assoc_abs_1.txt':
					pl.text( 1.0025*z_abs, 0.998*ratio, `int(N_CIV_no[i])`, fontsize=16 )
				else:
					pl.text( 1.006*z_abs, 0.998*ratio, `int(N_CIV_no[i])`, fontsize=16 )
				pl.xlabel('z', family='serif', fontsize=16)
				pl.ylabel(r'$\log{\rm{N}_{NV}}$/$\log{\rm{N}_{CIV}}$', family='serif', fontsize=16 )

	for i in range(len(n)):
		for j in range(len(c)):
		
			#second CIV detections
			if ( len(c[j]) > 1. and (n[i][0][0] == c[j][1][0] and n[i][0][1] == c[j][1][1] and n[i][0][2] == c[j][1][2]) ):
				ratio = n[i][0][4]/c[j][1][4]
				yerr = ratio*np.sqrt( (n[i][0][5]/n[i][0][4])**2 + (c[j][1][5]/c[j][1][4])**2 )
				pl.scatter( n[i][0][1], n[i][0][4]/c[j][1][4], c='blue', s=60 )
				pl.errorbar( n[i][0][1], n[i][0][4]/c[j][1][4], yerr=yerr, capsize=6, c='blue', zorder=0 )

				#all primary detections
			elif ( (n[i][0][0] == c[j][0][0] and n[i][0][1] == c[j][0][1] and n[i][0][2] == c[j][0][2]) ):
				pl.scatter( n[i][0][1], n[i][0][4]/c[j][0][4], c='grey', label='Fechner et al (2009)' if i==1 else "", s=60 )
				ratio = n[i][0][4]/c[j][0][4]
				yerr = ratio*np.sqrt( (n[i][0][5]/n[i][0][4])**2 + (c[j][0][5]/c[j][0][4])**2 )
				pl.errorbar( n[i][0][1], n[i][0][4]/c[j][0][4], yerr=yerr, capsize=6, c='grey', zorder=10 )
	
	if file == direct+'Fechner_2009_interven_abs_1.txt':	
		pl.text( 0.04, 0.92, 'intervening', transform=ax.transAxes, family='serif', fontsize=16 )
		pl.legend(loc=4)

		pl.savefig('out/QSO_vs_HzRGs/intervening_QSO_HzRGs_redshift.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/intervening_QSO_HzRGs_redshift.pdf')


	elif file == direct+'Fechner_2009_assoc_abs_1.txt':
		pl.text( 0.04, 0.92, 'associated', transform=ax.transAxes, family='serif', fontsize=16 )
		pl.legend(loc=4)

		pl.savefig('out/QSO_vs_HzRGs/associated_QSO_HzRGs_redshift.png')
		pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/associated_QSO_HzRGs_redshift.pdf')
