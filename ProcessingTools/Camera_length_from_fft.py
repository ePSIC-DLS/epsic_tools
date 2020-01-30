# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:01:43 2019

@author: gys37319
"""

import hyperspy.api as hs
#px size from epie:cl = 11.4 cm , dpx =  4.013e-11
#cl = 12.85 cm ,dpx = 4.52e-11
#have to have dat_fft  in memeory already
fft_hs = hs.signals.Signal2D(data = np.log10(np.abs(dat_fft)**2))
fft_hs.plot()
line_scan = hs.roi.Line2DROI(200,200,400,400,linewidth =10)
ls = line_scan.interactive(fft_hs)
ls.plot()
#first reflection distances
#cl =11.4 cm ,  r1 = np.array([309 - 36, 278-24, 295-35])
#cl = 12.85
r1 = np.array([287 - 34 ,262-25, 257 - 24 ])
#another measurement of the same data: r1b = np.array([270-25, 263 - 27, 271 - 35])
  #second reflection distances
#cl = 11.3 cm r2 = np.array([487-53, 496 - 50, 524-58 ])

r1_mean = r1.mean() /2
r1_err = 0.5* r1.std() / np.sqrt(r1.shape)

r2_mean = r2.mean() / 2
r2_err = r2.std() / np.sqrt(r2.shape)

#1st graphene reflection at 2.14Ang
d1 = 2.14e-10
#2nd graphene reflection at 1.23Ang
d2= 1.23e-10

#convert to reciprical space
k1 = 1 / d1
k2 = 1 / d2

#get recip px size
kpx1 = k1 / r1_mean
kpx2 = k2 / r2_mean

#FOV in image 
FOV1 = 1/kpx1
FOV2 = 1/kpx2

#px size in image 
ipx1 = 1/(kpx1 * dat_fft.shape[0])
ipx2 = 1/(kpx2 * dat_fft.shape[0])

#OR px in real space correspondign to d1
d1_Npx = dat.shape[0] / r1_mean 
ipx1b = d1 / d1_Npx
