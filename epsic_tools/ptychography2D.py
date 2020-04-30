# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:37:55 2020

@author: gys37319
"""

from epsic_tools import build_params
from epsic_tools.toolbox.ptychography import ptyrex
import json
import hyperspy.api as hs

def is_number(s):
    ''' 
    checks if input is a number
    
    Parameters
    ----------
    
    
    s: input 
    
    
    Returns
    -------
    
    True or False  
    '''
    
    try:
        float(s)
        return True
    except ValueError:
        return False

class Ptychography2D(object):
    ''' 
    Ptychography object 
    '''
    
    def __init__(self):
        self.ptyrex = build_params.init_ptyrex()
        self.recon = []
        self.recon_object = []
        self.recon_probe = []
        self.recon_object_fft = []
        self.recon_radial_object_fft = []
        self.recon_scan_rotation = []
        self.recon_scan_step = []
        self.recon_error = []
        #self.recon_full_error = []
        #self.ptypy = build.ptypy()
        
    def load_series(self, pn,crop_to, sort_by = 'rot', blur = 0, verbose = False, plot_me = True):
        
        d_s, p_s, d_s_fft, rad_fft, r_s, s_s, e_s = ptyrex.load_series(pn,crop_to, sort_by = sort_by, blur = blur, verbose = verbose)
        if plot_me:
            hs.plot.plot_signals([d_s,p_s,d_s_fft, rad_fft], navigator_list=[r_s,s_s, e_s,None])
        self.recon_object = d_s
        self.recon_probe = p_s
        self.recon_object_fft = d_s_fft
        self.recon_radial_object_fft = rad_fft
        self.recon_scan_rotation = r_s
        self.recon_scan_step = s_s
        self.recon_error = e_s
        #self.recon_full_error = fe_s
    
    def load_recon(self, fn):
    #load a ptyrex reconstruction
 
        d, p, e = ptyrex.load_recon(fn)
        self.recon_object = d
        self.recon_probe = p
        #self.recon_object_fft = d_s_fft
        #self.recon_radial_object_fft = rad_fft
        #self.recon_scan_rotation = r_s
        #self.recon_scan_step = s_s
        self.recon_error = e
    #self.recon_error = ptyrex.load_error(fn)

    
    def get_interaction(self):
        return self.ptyrex['process']['interaction']
        
    def set_interaction(self, val):
        if isinstance(val , str):
            self.ptyrex['process']['interaction'] = val
        else:
            raise ValueError('interaction must be a string')
        
    def get_save_prefix(self):
        return self.ptyrex['process']['save_prefix']
        
    def set_save_prefix(self, val):
        if isinstance(val , str):
            self.ptyrex['process']['save_prefix'] = val
        else:
            raise ValueError('save prefix must be a string')
            
    def get_experiment_ID(self):
        return self.ptyrex['experiment']['experiment_ID']
        
    def set_experiment_ID(self, val):
        if isinstance(val , str):
            self.ptyrex['experiment']['experiment_ID'] = val
        else:
            raise ValueError('experiment ID must be a string')
        
    def get_gpu_flag(self):
        return self.ptyrex['process']['gpu_flag']
    
    def set_gpu_flag(self, val):
        if val == 1 or val == 0:
            self.ptyrex['process']['gpu_flag'] = val
        else:
            raise ValueError("gpu flag takes a value of either 1 or 0")
            
    def get_save_interval(self):
        return self.ptyrex['process']['save_interval']
    
    def set_save_interval(self, val):
        if int(val) == val:
            self.ptyrex['process']['save_interval'] = val
        else:
            raise ValueError("save_interval must be an integer")
            
    def get_iterations(self):
        return self.ptyrex['process']['PIE']['iterations']
        
    def get_decay(self):
        return self.ptyrex['process']['PIE']['decay']
        
    def set_decay(self, val):
        if len(val) == 3:
            self.ptyrex['process']['PIE']['decay'] = val
        else:
            raise ValueError("decay must be of the form [max, min, power]")
        
    def set_iterations(self, val):
        if int(val) == val:
            self.ptyrex['process']['PIE']['iterations'] = val
        else:
            raise ValueError("iterations must be an integer")
    
    def get_energy(self):
        return self.ptyrex['process']['common']['source']['energy']
        
    def set_energy(self, val):
        if val > 0:
            self.ptyrex['process']['common']['source']['energy'] = [val]
        else:
            raise ValueError("energy must be a positive real number")
            
    def get_px_pitch(self, val):
        return self.ptyrex['process']['common']['detector']['pix_pitch']
        
    def set_px_pitch(self, val):
        if len(val) == 2: 
            self.ptyrex['process']['common']['detector']['pix_pitch'] = val
        else:
            raise ValueError("px pitch must take the form [dx, dy]")
    
    def get_detector_bin(self):
        return self.ptyrex['process']['common']['detector']['bin']
        
    def set_detector_bin(self, val):
        if len(val) == 2: 
            self.ptyrex['process']['common']['detector']['bin'] = val
        else:
            raise ValueError("detector bin must take the form [binx, biny]")
            
    def get_detector_min_max(self):
        return self.ptyrex['process']['common']['detector']['min_max']
        
    def set_detector_min_max(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['detector']['min_max'] = val
        else:
            raise ValueError("detector min max must take the form [min, max]")
    
    def get_detector_optic_axis(self):
        return self.ptyrex['process']['common']['detector']['optic_axis']
    
    def set_detector_optic_axis(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['detector']['optic_axis'] = val
        else:
            raise ValueError("optic axis must take the form [x, y]")
    
    def get_detector_crop(self):
        return self.ptyrex['process']['common']['detector']['crop']
        
    def set_detector_crop(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['detector']['crop'] = val
        else:
            raise ValueError("detector crop must take the form [x,y]")
            
    def get_detector_orientation(self):
        return self.ptyrex['process']['common']['detector']['orientation']
        
    def set_detector_orientation(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['detector']['orientation'] = val
        else:
            raise ValueError("detector orientation must take the form [x,y]")
    
    def get_convergence(self):
        return self.ptyrex['process']['common']['probe']['convergence']
        
    def set_convergence(self, val):
        if is_number(val) == True and val > 0:
            #do we need both of these??
            self.ptyrex['process']['common']['probe']['convergence'] = val
            self.ptyrex['experiment']['optics']['lens']['alpha'] = val
        else:
            raise ValueError("convergence must be a positive number")
            
    def get_load_probe_flag(self):
        return self.ptyrex['process']['common']['probe']['load_flag']
        
    def set_load_probe_flag(self, val):
        if val ==1 or val == 0:
            self.ptyrex['process']['common']['probe']['load_flag'] = val
        else:
            raise ValueError("load flag must take a value of 1 or 0")
    
    def get_load_probe_path(self):
        return self.ptyrex['process']['common']['probe']['path']
        
    def set_load_probe_path(self, val):
        #todo : check if path is valid here??
        if isinstance(val , str):
            self.ptyrex['process']['common']['probe']['path'] = val
        else:
            raise ValueError("probe path must be a string")
    
    def get_load_object_flag(self):
        return self.ptyrex['process']['common']['object']['load_flag']
        
    def set_load_object_flag(self, val):
        if val == 1 or val == 0:
            self.ptyrex['process']['common']['object']['load_flag'] = val
        else:
            raise ValueError("object flag must take a value of 1 or 0")
    
    def get_load_object_path(self):
        return self.ptyrex['process']['common']['object']['path']
        
    def set_load_object_path(self,val):
        if isinstance(val , str): 
            self.ptyrex['process']['common']['object']['path'] = val
        else:
            raise ValueError("object path must be a string")
            
    def get_scan_rotation(self):
        return self.ptyrex['process']['common']['scan']['rotation']
        
    def set_scan_rotation(self, val):
        if is_number(val):
            self.ptyrex['process']['common']['scan']['rotation'] = val
        else:
            raise ValueError("scan rotation must be a number")
    
    def get_scan_step(self):
        return self.ptyrex['process']['common']['scan']['dR']
        
    def set_scan_step(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['scan']['dR'] = val
        else:
            raise ValueError("scan step must take the form [dx, dy]")
            
    def get_scan_size(self):
        return self.ptyrex['process']['common']['scan']['N']
        
    def set_scan_size(self, val):
        if len(val) == 2:
            self.ptyrex['process']['common']['scan']['N'] = val
        else:
            raise ValueError("scan size must take the form [x, y]")
            
    def get_load_data_flag(self):
        return self.ptyrex['experiment']['data']['load_flag']
        
    def set_load_data_flag(self, val):
        if val == 1 or val == 0 :
            self.ptyrex['experiment']['data']['load_flag'] = val
        else:
            raise ValueError("load data flag must take a value or 0 or 1")
            
    def get_data_path(self):
        return self.ptyrex['experiment']['data']['data_path']
        
    def set_data_path(self, val):
        if isinstance(val , str):
            self.ptyrex['experiment']['data']['data_path'] = val
        else:
            raise ValueError("data path must be a string")
            
    def get_dead_px_flag(self):
        return self.ptyrex['experiment']['data']['dead_pixel_flag']
        
    def set_dead_px_flag(self, val):
        if val == 0 or val == 1:
            self.ptyrex['experiment']['data']['dead_pixel_flag'] = val
        else:
            raise ValueError("dead px flag must take a value of 0 or 1")
        
    def get_dead_px_path(self):
        return self.ptyrex['experiment']['data']['dead_pixel_path']
        
    def set_dead_px_path(self, val):
        if isinstance(val , str):
            self.ptyrex['experiment']['data']['dead_pixel_path'] = val
        else:
            raise ValueError("dead px path must be a string")
    
    def get_flat_field_flag(self):
        return self.ptyrex['experiment']['data']['flat_field_flag']
        
    def set_flat_field_flag(self, val):
        if val == 0 or val == 1:
            self.ptyrex['experiment']['data']['flat_field_flag'] = val
        else:
            raise ValueError("flat field flag must take a value of 0 or 1")
            
    def get_flat_field_path(self):
        return self.ptyrex['experiment']['data']['flat_field_flag']
        
    def set_flat_field_path(self, val):
        if isinstance(val , str):
            self.ptyrex['experiment']['data']['flat_field_flag'] = val
        else:
            raise ValueError("flat field path must be a string")
    
    def get_data_key(self):
        return self.ptyrex['experiment']['data']['key']
        
    def set_data_key(self, val):
        if isinstance(val , str):
            self.ptyrex['experiment']['data']['key']  = val
  
        else:
            raise ValueError("data key must be a string")
    
    def get_detector_pos(self):
        return self.ptyrex['experiment']['detector']['position']
        
    def set_detector_pos(self, val):
        if len(val) == 3:
            self.ptyrex['experiment']['detector']['position'] = val
        else:
            raise ValueError("detector position must take the value [x_pos, y_pos_, z_pos]")
    
    def get_defocus(self):
        return self.ptyrex['experiment']['optics']['lens']['defocus']
        
    def set_defocus(self, val):
        if is_number(val):
            self.ptyrex['experiment']['optics']['lens']['defocus'] = [val, val]
        else:
            raise ValueError("defocus must be a number")
            
    def get_source_modes(self):
        return self.ptyrex['process']['PIE']['source']['sx'], self.ptyrex['process']['PIE']['source']['sy']
        
    def set_source_modes(self, val):
        if len(val) == 2: 
            self.ptyrex['process']['PIE']['source']['sx'] = val[0]
            self.ptyrex['process']['PIE']['source']['sy'] = val[1]
        else:
            raise ValueError('source modes must take the form [sx, sy]')
            
    def get_n_cores(self):
        return self.ptyrex['process']['cores']
        
    def set_n_cores(self, val):
        if val > 0 and int(val) == val:
            self.ptyrex['process']['cores'] = val
        else:
            raise ValueError("number of cores must be a positive integer")
            
    def get_base_dir(self):
        return self.ptyrex['base_dir']
        
    def set_base_dir(self, val):
        if isinstance(val , str):
            self.ptyrex['base_dir'] = val
        else:
            raise ValueError("base diretory must be a string")
            
    def get_save_dir(self):
        return self.ptyrex['process']['save_dir']
        
    def set_save_dir(self, val):
        if isinstance(val , str):
            self.ptyrex['process']['save_dir'] = val
        else:
            raise ValueError("save diretory must be a string")
            
    def save_json(self, file_path):
        with open(file_path, 'w+') as outfile:
            json.dump(self.ptyrex, outfile, indent = 4)
        print("json file saved to : " , file_path)
    
            
