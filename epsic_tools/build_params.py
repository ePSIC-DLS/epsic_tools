# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:51:27 2020

@author: gys37319
"""

import collections

class NestedDefaultDict(collections.defaultdict):
    '''
    Initialisation of nested dictionary
    '''

    def __init__(self, *args, **kwargs):

        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)



    def __repr__(self):

        return repr(dict(self))
    
    
def init_ptyrex():
    ''' 
    Builds a dictionary of default parameters required for a ptyrex json file
    
    Parameters
    ----------
    
    Returns
    -------
    
    params : dict of parameters
    

    '''
    params = NestedDefaultDict()
    
    params['process']['gpu_flag'] = 1

    params['process']['save_interval'] = 10

    params['process']['PIE']['iterations'] = 100

    params['process']['common']['source']['energy'] = 200e3

    params['process']['common']['source']['radiation'] = 'electron'

    params['process']['common']['source']['flux'] = -1
    
    params['process']['PIE']['decay'] = [1, 0, 0]

    

    params['process']['common']['detector']['pix_pitch'] = [55e-6, 55e-6]

    #params['process']['common']['detector']['distance'] = 0.125

    params['process']['common']['detector']['bin'] = [1, 1]

    params['process']['common']['detector']['min_max'] = [0, 1000000]

    params['process']['common']['detector']['optic_axis']= [-1 , -1]

    params['process']['common']['detector']['crop'] = [512,512]

    params['process']['common']['detector']['orientation'] = '10'

    params['process']['common']['detector']['mask_flag'] = 0

    

    params['process']['common']['probe']['convergence'] = 0.050

    #params['process']['common']['probe']['distance'] = -1

    #params['process']['common']['probe']['focal_dist'] = -1

    params['process']['common']['probe']['load_flag'] = 0
    
    params['process']['common']['probe']['path']  = 'None'   

    #params['process']['common']['probe']['diffuser'] = 0

    #params['process']['common']['probe']['aperture_shape'] = 'circ'

    #params['process']['common']['probe']['aperture_size'] = exp_dict['pupil_rad(pixels)']*exp_dict['detector_pixel_size(m)']



    params['process']['common']['object']['load_flag'] = 0

    params['process']['common']['object']['path'] = 'None'

    params['process']['common']['scan']['rotation'] = 54

    params['process']['common']['scan']['fast_axis'] = 1

    params['process']['common']['scan']['orientation'] = '11'

    params['process']['common']['scan']['type'] = 'tv'

    params['process']['common']['scan']['load_flag'] = 0
    
    params['process']['common']['scan']['path'] = 'None'

    params['process']['common']['scan']['dR'] = [1e-11, 1e-11]

    params['process']['common']['scan']['N'] = [64,64]

    

    params['experiment']['data']['data_path'] = 'None'
    
    params['experiment']['data']['load_flag'] = 1
    
    params['experiment']['data']['dead_pixel_flag'] = 0 

    params['experiment']['data']['flat_field_flag'] = 0 

    params['experiment']['data']['dead_pixel_path'] = 'None'

    params['experiment']['data']['flat_field_path'] = 'None'



    params['experiment']['data']['meta_type'] = 'hdf'

    params['experiment']['data']['key'] = ''

    

    params['experiment']['sample']['position'] = [0, 0, 0]



    params['experiment']['detector']['position'] = [0,0,0.14]



    params['experiment']['optics']['lens']['alpha'] = 0.050

    params['experiment']['optics']['lens']['defocus'] = [0,0]

    params['experiment']['optics']['lens']['use'] = 1

    #params['experiment']['optics']['diffuser']['use'] = 0

    #params['experiment']['optics']['FZP']['use'] = 0

    #params['experiment']['optics']['pinhole']['use'] = 0



    params['base_dir'] = 'None'

    params['process']['save_dir'] = 'None'

    params['process']['cores'] = 1
    
    return params