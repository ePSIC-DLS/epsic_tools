# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:08:22 2020

@author: gys37319
"""
import numpy as np
from decimal import Decimal
#%%
electron_energy_eV = 80000
electron_rest_mass_eV = 510998.9461
hc_m_eV = 1.23984197e-6 
wavelength = wavelength = hc_m_eV / np.sqrt(electron_energy_eV*(electron_energy_eV + 2*electron_rest_mass_eV))
print('electron wavlength (m) : ', "{:.2E}".format(Decimal((wavelength))))
atomic_spacing_m =1.23e-10#2.13e-10
atomic_spacing_rad = wavelength / atomic_spacing_m
print('reflection position (mrad) : ',  "{:.2f}".format(atomic_spacing_rad*1000))
#%%

r =3.2e-10#0.0155 #1e-10 #probe radius or convergence semi angle
d = 4*4.5e-11#0.0196#4*3.72e-11# probe spacing or reflection in mrad (atomic_spacing_rad)
x_pos  = 0.5 * d # x coordinate of circle intersection
y_pos = np.sqrt(r**2 - x_pos**2) # y coordinate of circle intersection
theta = 2*np.arctan(y_pos / x_pos) # angle subtended by overlap
A_overlap =  (r**2 * theta) - (2*x_pos*y_pos) # area of overlap
A_probe = np.pi * r**2
Percentage_overlap =100 *  A_overlap / A_probe
print('x, y : ', x_pos, y_pos)
print('theta : ', theta )
print('overlap area : ', A_overlap)
print('probe area : ', A_probe)
print('overlap  % : ', Percentage_overlap)