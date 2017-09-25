#velocity_shift_bars.py
#PURPOSE: Plot velocity shifts of fine-structure lines in MRC 0943-242 MUSE spectrum

from math import*
import numpy as np
import matplotlib.pyplot as pl

data = np.loadtxt("out/0943 gaussian fit.txt",dtype=str)

# --------------------------------------------
#   convert numbers from strings to floats
# --------------------------------------------
N = len(data)
rows = [ [] for i in range(N)]
for j in range(N):
	n = len(data[j])
	for i in range(1,n):
		rows[j].append( float(data[j][i]) )

for j in range(N):
	new_data = [ [data[j][0]]+rows[j] for j in range(N)]

# ----------------------------
#     Velocity Offsets
# ----------------------------
#precise speed of light in km/s
c   = 299792.458	

#radial velocity (Doppler)
def vel(wav_obs,wav_em):
	# wav_r = (wav_obs/wav_em)**2			
	# v = c*(wav_r - 1.)/(wav_r + 1)
	v = c*((wav_obs/wav_em) - 1.)
	return v

vel = [ vel(new_data[j][1],new_data[j][10]) for j in range(N) ]

#systemic velocity := HeII 
for j in range(N):
	if new_data[j][0] == 'HeII':
		vel0 = vel[j]
		z = vel0/c

offset_vel = [ (vel[j] - vel0) for j in range(N) ]
new_data = [ new_data[j] + [offset_vel[j]] for j in range(N) ]

fig = pl.figure()
pl.xlim([-700.,700.])
pl.ylim([0.,12.])
pl.xlabel('Velocity Offset (km/s)')
pl.plot([0.,0.],[0.,12.],ls='--',color='grey')
i = 0

v_err = []
#propagate errors - Doppler eqn
for j in range(N):
	wav0 		= new_data[j][1]
	err_wav0 	= new_data[j][2]
	v   		= offset_vel[j]
	v_err.append( abs(v*(err_wav0/wav0)) ) 

for j in range(N):
	i+=1
	pl.scatter([offset_vel[j]],[i],edgecolors='blue',facecolors='white')
	pl.errorbar([offset_vel[j]],[i], xerr=v_err[j], ecolor='blue',capsize=5)
	pl.text(450.,i,new_data[j][0]+' '+str( new_data[j][10]))

[ pl.savefig("out/0943_velocity_shifts."+form) for form in ('png','eps') ]
pl.show()