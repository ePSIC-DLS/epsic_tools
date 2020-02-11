# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:15 2020

@author: gys37319
"""

import hyperspy.api as hs
from scipy.ndimage import gaussian_filter
import numpy as np
import Find_2Dpeaks as peaks
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import warp_3d as wp 
from decimal import Decimal
from skimage.feature import match_template
from scipy.ndimage.morphology import binary_fill_holes
from scipy.optimize import curve_fit
#from skimage.filters import gaussian
#%%

#digital micrograph file path
f_dm = r'Y:/2020/cm26481-1/raw/DM/20200130_80kV_600C_graphene/HAADF _30MX _0003.dm4'
#Merlin data file path
f_m = r'Y:/2020/cm26481-1/processing/Merlin/20200130_80kV_graphene_600C_pty/20200131 154940/80kV_600C_CLA_10um_CL_20cm_8C_20Mx_A2_4p71_df0nm_scan_array_255by255_diff_plane_515by515_.hdf5'
#r'Y:/2020/cm26481-1/processing/Merlin/20200130_80kV_graphene_600C_pty/20200131 154940/binned_diff_80kV_600C_CLA_10um_CL_20cm_8C_20Mx_A2_4p71_df0nm_scan_array_255by255_diff_plane_128by128_.hdf5'
#r'Y:/2020/cm26481-1/processing/Merlin/20200130_80kV_graphene_600C_pty/20200131 140133/binned_diff_80kV_600C_CLA_40um_CL_8cm_8C_20Mx_A2_4p71_dfneg10nm_scan_array_255by255_diff_plane_128by128_.hdf5'
alpha = 0.00775#convergence semi-angle rad
electron_energy_eV = 80000
electron_rest_mass_eV = 510998.9461
hc_m_eV = 1.23984197e-6 
wavelength = wavelength = hc_m_eV / np.sqrt(electron_energy_eV*(electron_energy_eV + 2*electron_rest_mass_eV))
print('electron wavlength (m) : ', "{:.2E}".format(Decimal((wavelength))))
atomic_spacing_m =1.23e-10#2.13e-10
atomic_spacing_rad = wavelength / atomic_spacing_m
print('reflection position (mrad) : ',  "{:.2f}".format(atomic_spacing_rad*1000))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 
#%%
def get_first_peaks(data, x, y, search_px = 2):
    cen = [pos / 2 for pos in data.shape]
    r = [np.sqrt((x - cen[0])**2 + (y - cen[1])**2)  for x,y in zip(x,y)]
    #print(r)
    center_i = np.argmin(r)# get rid of middle point
    cen = [x[center_i], y[center_i]]
    r= np.ma.array(r, mask = False)
    r.mask[center_i] = True
    low = r.min() - search_px
    high = r.min() +search_px
    #ef_inds = [i for i in range(len(r)) if low<= r[i] <= high ] #only take first reflections
    #x2  = [x[ref_ind] for ref_ind in ref_inds] 
    #y2 = [y[ref_ind] for ref_ind in ref_inds] 
    
    x2 = x[(r.data >= low) & (r.data <= high)]
    y2 = y[(r.data >= low) & (r.data <= high)]
    return x2, y2, cen
#%%
dm = hs.load(f_dm)
dm.plot()
sq_roi = hs.roi.RectangularROI(0, 0, 4, 4)
dm_crop = sq_roi.interactive(dm)
#dm_crop.plot()
dm_crop_blur = gaussian_filter(dm_crop.data , sigma =2 )
dm_crop_blur = hs.signals.Signal2D(dm_crop_blur)
dm_crop_blur.plot()
#%%
crop_fft = np.fft.fft2(dm_crop_blur.data)#, norm = 'ortho')
crop_fft = np.fft.fftshift(crop_fft)
crop_fft = hs.signals.Signal2D((np.log10(np.abs(crop_fft))**2))
#crop_fft.plot()
x,y = peaks.detect_peaks(crop_fft.data, neighborhood_size = 5,threshold =25, max_min = 'max' )
x = np.array(x)
y = np.array(y)
#print(x,y)
#ax = plt.gca()
plt.figure()
plt.imshow(crop_fft.data)
plt.plot(x,y, 'rx')
plt.show()

x2, y2, cen = get_first_peaks(crop_fft.data, x, y)
plt.plot(x2, y2, 'bo')
angles = np.zeros_like(x2)
#cen = [pos / 2 for pos in crop_fft.data.shape]
for i in range(x2.shape[0]):
    angles[i] = angle_between([x2[i] - cen[0], y2[i] - cen[0]], [0,-1])
    if x2[i]-cen[0] < 0:
        #print(x2[i])
        angles[i] = 2*np.pi- angles[i]
#angles = np.arctan((y2 - cen[1]) / (x2 - cen[0]))
print('DM angles clockwise from vertical : ' ,np.sort( np.rad2deg(angles)))
#%%
m = hs.load(f_m, lazy = True)
m_sub = m.inav[::4, ::4]
m_sub.compute()
m_sum = m_sub.sum()
m_sum.data = m_sum.data**0.1
#%%
#remove crosses
m_sum.data[256:259,:] = (m_sum.data[255,:]+ m_sum.data[259,:]) / 2
m_sum.data[:,256] = (m_sum.data[:,255]+ m_sum.data[:,259]) / 2
m_sum.data[:,257] = (m_sum.data[:,255]+ m_sum.data[:,259]) / 2
m_sum.data[:,258] = (m_sum.data[:,255]+ m_sum.data[:,259]) / 2
m_sum.plot()

#%%
# find bfd radius, calculate graphene radial positions 
res = wp.fit_ellipse(m_sum.data,threshold = 2.8, sigma = 2, plot_me = True)
px_rad = alpha / res[2].mean()
ref_pos_guess = atomic_spacing_rad  / px_rad

wp.plot_ellipse(m_sum.data,[res[0],0,[ref_pos_guess, ref_pos_guess] ] )
#%%
bfd_template = m_sum.data[int(np.floor(res[0][1] - res[2][1])-2):int(np.ceil(res[0][1] + res[2][1])+2),int(np.floor(res[0][0] - res[2][0])-2):int(np.ceil( res[0][0] + res[2][0])+2)]
bfd_template = np.where(bfd_template >bfd_template.mean(), 1,0)
bfd_template = binary_fill_holes(bfd_template)
#plt.figure();
#plt.imshow(bfd_template)
m_sum_smooth= gaussian_filter(m_sum.data, 2)
bfd_smooth = gaussian_filter(bfd_template.astype('float'),2)
results = match_template(m_sum_smooth, bfd_smooth,pad_input=True)
plt.figure()
plt.imshow(m_sum.data)
x_tem,y_tem = peaks.detect_peaks(results, neighborhood_size = 20,threshold =0.1, max_min = 'max' )
x_tem = np.array(x_tem)
y_tem = np.array(y_tem)
plt.plot(x_tem, y_tem, 'rx')
x3, y3, cen_m = get_first_peaks(results, x_tem, y_tem, 40)
plt.plot(x3, y3, 'bo')
plt.plot(res[0][0], res[0][1], 'g*')
angles_m = np.zeros_like(x3)

for i in range(x3.shape[0]):
    angles_m[i] = angle_between([x3[i] - res[0][0], y3[i] - res[0][0]], [0,-1])
    if x3[i]-cen_m[0] < 0:
        #print(x2[i])
        angles_m[i] = 2*np.pi- angles_m[i]
deg_angles_m = np.rad2deg(angles_m)
np.set_printoptions(precision=2, suppress=True)
print('Merlin angles clockwise from vertical : ',deg_angles_m)
#%%
#manually find circles
m_pow = hs.signals.Signal2D(data = m_sum.data)
m_pow.plot()
C_roi = C_roi = hs.roi.CircleROI(cen_m[0],cen_m[1], res[2].max())
c = C_roi.interactive(m_pow)
#%%
x_man = np.array([253.,345.,346.,248.,154.,157.])
y_man = np.array([159.,204.,317.,374.,322.,210.])

angles_man = np.zeros_like(x_man)
for i in range(x_man.shape[0]):
    angles_man[i] = angle_between([x_man[i] - res[0][0], y_man[i] - res[0][1]], [0,-1])
    if x_man[i]-cen_m[0] < 0:
        #print(x2[i])
        angles_man[i] = 2*np.pi- angles_man[i]
plt.figure()
plt.imshow(m_sum.data)       
plt.plot(x_man, y_man, 'bo')
plt.plot(res[0][0], res[0][1], 'g*')
print('Merlin angles clockwise from vertical (manual) : ' , np.rad2deg(angles_man))
#%%
a_dm = np.sort( np.rad2deg(angles))
a_m = np.sort(np.rad2deg(angles_man))
print(np.ediff1d(a_dm))
print(np.ediff1d(a_m))
print(a_dm - a_m)
#%%
dm_res = wp.fitEllipse(x2, y2)
dm_center = wp.ellipse_center(dm_res)
dm_axis = wp.ellipse_axis_length(dm_res)
dm_angle = wp.ellipse_angle_of_rotation(dm_res)

m_man_res = wp.fitEllipse(x_man, y_man)
m_man_cen = wp.ellipse_center(m_man_res)
m_man_axis = wp.ellipse_axis_length(m_man_res)
m_man_angle = wp.ellipse_angle_of_rotation(m_man_res)

print('dm ratio: ',dm_axis[0]/dm_axis[1] )
print('merlin ratio : ', m_man_axis[1]/ m_man_axis[0])
print('bfd ratio : ', res[2][0]/res[2][1])