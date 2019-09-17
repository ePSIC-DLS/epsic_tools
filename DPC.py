
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import h5py
from scipy.ndimage.filters import gaussian_filter
import hyperspy.api as hs

import py4DSTEM
from py4DSTEM.process.calibration import get_probe_size
from py4DSTEM.process.dpc import get_CoM_images, get_rotation_and_flip, get_phase_from_CoM
from py4DSTEM.process.dpc import get_wavenumber, get_interaction_constant

# Load data

fp =  r"Y:\data\2019\mg22549-6\processing\Merlin\Merlin\20190808 145141"
#200e/A data: r"Y:\2019\mg22317-33\processing\Merlin\Merlin\20190527 160131"
fn =r"\MoS2 700 20cm 20M.hdf5"
#200e/A data: r"\binned_ZSM5_300kV_7p6mrad_20Mx_40cm_A2_2p01_072.hdf5"
#r"\binned_a_file_001.hdf5"
#read in arbitrary hdf5 file
#h5d = h5py.File(fp+fn, 'r')
#pass to hyperspy
#d = hs.signals.Signal2D(fp+fn)#h5d['data'][:])#, lazy = True)

d = hs.load(fp + fn, lazy = True)
dc = py4DSTEM.file.datastructure.DataCube(d.data)
#%%
rx,ry = 30,30
power = .5

BF = np.average(dc.data,axis=(2,3))
DP = dc.data[rx,ry,:,:]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(BF)
ax2.matshow(DP**power)
plt.show()


#%%
# Get PACBED

power = 0.1

PACBED = np.average(dc.data,axis=(0,1))

# Show
fig,ax = plt.subplots(figsize=(8,8,))
ax.matshow(PACBED**power)
plt.show()

thresh_lower = 0.001
thresh_upper = 0.01
N = 100

r,x0,y0 = get_probe_size(PACBED, thresh_lower=thresh_lower, thresh_upper=thresh_upper, N=N)

# Show
fig,ax = plt.subplots(figsize=(8,8))
ax.matshow(PACBED**power)
ax.scatter(y0,x0,color='r',s=10)
circle = Circle((y0,x0),r,fill=False,edgecolor='r',linewidth=1)
ax.add_patch(circle)
plt.show()

# get ADF image



# Get mask

#%%
fig,ax = plt.subplots(figsize=(8,8))
ax.matshow((mask*PACBED)**power)
plt.show()

#get ADF
mask_ADF = qr > r + expand

ADF = np.average(mask_ADF * dc.data4D,axis=(2,3))
fig,ax = plt.subplots(figsize=(8,8))
ax.matshow(ADF)
plt.show()
#fig.savefig(fp + r"/ADF_warp.png")


#%%
expand = 30

qy,qx = np.meshgrid(np.arange(dc.Q_Ny),np.arange(dc.Q_Nx))
qr = np.hypot(qx-x0,qy-y0)
mask = qr < r + expand
#%%

normalize = False # set to false for v. low dose data

CoMx,CoMy = get_CoM_images(dc, mask=mask, normalize=normalize)

# Show
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(CoMx)
ax2.matshow(CoMy)
plt.show()
#fig.savefig(fp + r"/CoM_warp.png")
#%%
#remove nans and normalize
#CoMx[CoMx == np.nan] = 
#remove NaN's and normalize - only needed for very low dose data.
CoMx_masked = np.ma.masked_invalid(CoMx)
CoMx_new= CoMx_masked.filled(fill_value =CoMx_masked.mean() )
CoMx_new = CoMx_new - CoMx_new.mean()

CoMy_masked = np.ma.masked_invalid(CoMy)
CoMy_new= CoMy_masked.filled(fill_value =CoMy_masked.mean() )
CoMy_new = CoMy_new - CoMy_new.mean()

CoMx = CoMx_new
CoMy = CoMy_new

f_CoMx = fp + r'/CoMx_warp.txt'
f_CoMy = fp + r'/CoMy_warp.txt'
#np.savetxt(f_CoMx, CoMx)
#np.savetxt(f_CoMy, CoMy)
#%%
n_iter = 100
stepsize = 4
return_costs = True

theta, flip, thetas, costs, thetas_f, costs_f = get_rotation_and_flip(CoMx, CoMy, dc.Q_Nx, dc.Q_Ny, n_iter=n_iter,
                                                                      stepsize=stepsize, return_costs=return_costs)

# Show
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(np.arange(len(thetas)),costs,color='b')
ax1.plot(np.arange(len(thetas_f)),costs_f,color='r')
ax2.plot(np.arange(len(thetas)),np.degrees(thetas),color='b',label='flip = F')
ax2.plot(np.arange(len(thetas_f)),np.degrees(thetas_f),color='r',label='flip = T')
ax1.set_ylabel("Costs")
ax1.set_xlabel("Iteration")
ax2.set_ylabel("Rotations (deg)")
ax2.set_xlabel("Iteration")
ax2.legend()
plt.show()

print("Rotational offset = {:.4} degrees".format(np.degrees(theta)))
print("Flip is set to {}".format(flip))
#%%
#if rotation determination fails - set to known values
flip = True
theta = np.deg2rad(64)

paddingfactor = 2
regLowPass =100
regHighPass = 10
stepsize = 1
n_iter =100

phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=flip, regLowPass=regLowPass, regHighPass=regHighPass,
                                  paddingfactor=paddingfactor, stepsize=stepsize, n_iter=n_iter)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(phase)
ax2.plot(np.arange(n_iter),error)
ax2.set_ylabel('Error')
ax2.set_xlabel('Iteration')
plt.show()
fig.savefig(fp + r"/phase_warp.png")

#save dpc
f_dpc = fp + r'/dpc_warp.txt'
np.savetxt(f_dpc, phase)



