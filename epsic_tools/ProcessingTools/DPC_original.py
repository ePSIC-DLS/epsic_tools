
# coding: utf-8

# # Module: dpc
# 
# This module contains functions for performing differential phase contrast imaging (DPC).
# 
# In DPC, the sample potential is reconstructed from shifts in the diffraction space center of mass of the electron beam at each scan position.  The idea is that deflection of the center of mass, under certain assumptions, scales linearly with the gradient of the phase of the sample transmittance.  When this correspondence holds, it is thus possible to invert the differential equation and extract the phase itself.* The phase scales with the potential according the the electron interaction constant.  The primary assumption made is that the sample is well described as a pure phase object (i.e. the real part of the transmittance is 1).  The inversion is performed here in Fourier space, i.e. using the Fourier transform property that derivatives in real space are turned into multiplication in Fourier space.
# 
# For more detailed discussion of the relavant theory, see, e.g.:
# - Ishizuka et al, Microscopy (2017) 397-405
# - Close et al, Ultramicroscopy 159 (2015) 124-137
# - Waddell and Chapman, Optik 54 (1979) No. 2, 83-96
# 
# This notebook demos:
# * Getting the CBED center and radius
# * Finding the centers of mass
# * Determining any rotational offset between the real and diffraction space coordinates
# * Recontructing the phase from the centers of mass
# 
# *Note: because in DPC a differential equation is being inverted - i.e. the fundamental theorem of calculus is being invoked - one might be tempted to call this "integrated differential phase contrast".  Strictly speaking, this term is redundant - performing an integration is simply how DPC works.  Anyone who tells you otherwise is selling something.
# 

# ### Import packages, load data

# In[1]:


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


# In[2]:


import os


# In[3]:


#os.chdir(r'/dls/e02/data/2019/mg21597-1/processing/Merlin/triplepoint_anti_ws_1/20190620 174154')
os.chdir(r'Y:\data\2019\cm22979-6\processing\Merlin\au_crossgrating_refData\20190822 135045')


# In[4]:


#fp = 'defocus_neg100nm_10Mx_8cmCL.hdf5'
fp = 'binned_au_crossgrating_15Mx_100umCL_0defocus.hdf5'
d = hs.load(fp, lazy = True)


# In[5]:


print(d)


# In[11]:


d_sub = d.inav[:100, :100]
d_sub.compute()

f
# In[12]:


print(d_sub)


# In[4]:


# Load data

fp = r'binned_focus_10Mx_15cmCL_10umAp.hdf5'
d = hs.load(fp)
#%%
dc = py4DSTEM.file.datastructure.DataCube(d_sub_skip.data)


# In[5]:


rx,ry = 15,15
power = .5

BF = np.average(dc.data,axis=(2,3))
DP = dc.data[rx,ry,:,:]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(BF)
ax2.matshow(DP**power)
plt.show()


# In[ ]:





# ### Get CBED center and radius
# 
# These are used for
# 1. Calibrating the diffraction plane pixel size
# 2. Masking the central disk

# In[6]:


# Get PACBED

power = 0.1

PACBED = np.average(dc.data,axis=(0,1))

# Show
fig,ax = plt.subplots(figsize=(8,8,))
ax.matshow(PACBED**power)
plt.show()


# In[7]:


thresh_lower = 0.01
thresh_upper = 0.1
N = 100

r,x0,y0 = get_probe_size(PACBED, thresh_lower=thresh_lower, thresh_upper=thresh_upper, N=N)

# Show
fig,ax = plt.subplots(figsize=(8,8))
ax.matshow(PACBED**power)
ax.scatter(y0,x0,color='r',s=10)
circle = Circle((y0,x0),r,fill=False,edgecolor='r',linewidth=1)
ax.add_patch(circle)
plt.show()


# In[ ]:





# ### Get centers of mass

# In[13]:


# Get mask

expand = 10

qy,qx = np.meshgrid(np.arange(dc.Q_Ny),np.arange(dc.Q_Nx))
qr = np.hypot(qx-x0,qy-y0)
mask = qr < r + expand

fig,ax = plt.subplots(figsize=(8,8))
ax.matshow((mask*PACBED)**power)
plt.show()


# In[14]:


normalize = True

CoMx,CoMy = get_CoM_images(dc, mask=mask , normalize=normalize)

# Show
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(CoMx)
ax2.matshow(CoMy)
plt.show()


# In[ ]:





# ### Get rotation and flip

# In[10]:


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


# In[ ]:





# ### Reconstruction

# In[11]:


get_ipython().magic('matplotlib qt5')


# In[11]:


paddingfactor = 2
regLowPass = 200
regHighPass = 0.3
stepSize = 0.5
n_iter = 20

phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=flip, regLowPass=regLowPass, regHighPass=regHighPass,
                                  paddingfactor=paddingfactor, stepsize=stepsize, n_iter=n_iter)


# In[12]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.matshow(phase)
ax2.plot(np.arange(n_iter),error)
ax2.set_ylabel('Error')
ax2.set_xlabel('Iteration')
plt.show()


# In[14]:


whos


# In[16]:


get_ipython().magic('matplotlib qt5')


# In[17]:


import matplotlib


# In[19]:


plt.figure()
plt.imshow(d_sub.sum(),norm=matplotlib.cm.colors.LogNorm(vmin=20, vmax=500),cmap='inferno')


# In[19]:


plt.savefig('neg100nm.png')


# In[ ]:




