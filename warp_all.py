# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:04:01 2019
Warp a full dataset on the cluster
@author: gys37319
"""
import hyperspy.api as hs
import warp_3d as wp
from skimage import transform as tf
import time
import argparse

#%%
#load data
def warp_all(pn, fn4d):
    pn = r'Y:\2019\mg22549-3\processing\Merlin\20190706 MoS2 700C\20190706 153407'
    fn4d = r'\30M_25cmCL.hdf5'
    
    d_4d = hs.load(pn+fn4d, lazy = True)
    da_4d_cut = d_4d.data[100:150, 100:150, :,:]
    #compute PACBED of cut data
    da_4d_sum = da_4d_cut.sum(axis = (0,1))
    d= hs.signals.Signal2D(da_4d_sum.compute())
    #d.plot()
    #%%
    #need flat_field and mask dead px here. 
    #get rid of hot pixels
    d.data[d.data > 20* d.data.mean()] = d.data.mean()
    #d.plot()
    
    #%%
    #fit ellipse and transform PACBED
    d.data = wp.remove_cross(d.data)
    params = wp.fit_ellipse(d.data, threshold = 0.15, plot_me = False)
    print('orignal params : ', params)
    transform = wp.compute_transform(d.data, params)
    dst = tf.warp(d.data, transform, order=1)
    
    #plt.figure()
    #plt.imshow(dst)
    #plt.figure()
    #plt.imshow(d.data)
    
    params_2 = wp.fit_ellipse(dst, threshold = 0.1, plot_me = True)
    print ('final params : ', params_2)
    
    
    #%%
    #apply to entire (cut) data_set
    #get coordinates of transform
    d_4d_shape = d_4d.shape
    coords =  tf.warp_coords(transform, (d_4d_shape.shape[-2], d_4d_shape.shape[-1]))
    t0 = time.time()
    #warp data
    warped_data= d_4d.map_blocks(wp.warp_all_np, dtype = 'float32', coords = coords)
    #save to hdf5
    wf = fn4d[:-5] + '_warp.hdf5'
    warped_data.to_hdf5(pn +wf,'data', compression = 'gzip')
    print(time.time() - t0)
    print(pn +wf)
#%%
#check fit
#new_sum = warped_data.sum(axis= (0,1))
#wp.plot_ellipse(new_sum, params_2)
#check CoM
#print('old CoM ; ' , sc.ndimage.measurements.center_of_mass(da_4d_cut[0,0,:,:].compute()))
#print('new CoM : ', sc.ndimage.measurements.center_of_mass(warped_data[0,0,:,:].compute()))
def main(pn, fn4d):
    warp_all(pn, fn4d)

if __name__ == "__main__":
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers =20, memory_limit = 100e9)
    client = Client(cluster)
    parser = argparse.ArgumentParser()
    parser.add_argument('pn', help='path')
    parser.add_argument('fn4d', help='hdf5 4d data set')

    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()
    
    main(args.pn, args.fn4d)
    #watch_convert(args.beamline, args.year, args.visit, args.folder, args.STEM_flag, args.scan_X)