# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:08:22 2020

@author: gys37319
Calculate the overlap percentage of two circles
"""
import numpy as np
r = 4e-10 #probe radius
d = 0.73e-10 # probe spacing
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
