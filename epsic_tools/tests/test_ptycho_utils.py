import os
import numpy.testing as nptest
import numpy.testing as nptest
from epsic_tools import api as epsic

def test_plot_ptyREX_output():
    root_path = os.path.dirname(epsic.__file__)
    test_json_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.json')
    epsic.ptycho_utils.plot_ptyREX_output(test_json_path)
    
def test_crop_recon_obj():
    root_path = os.path.dirname(epsic.__file__)
    test_json_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.json')
    epsic.ptycho_utils.crop_recon_obj(test_json_path)
  
def test_get_json_pixelSize():
    root_path = os.path.dirname(epsic.__file__)
    test_json_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.json')
    epsic.ptycho_utils.get_json_pixelSize(test_json_path)
    
    
def test_json_to_dict():
    root_path = os.path.dirname(epsic.__file__)
    test_json_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.json')
    epsic.ptycho_utils.json_to_dict(test_json_path)
    
def test_get_error():
    root_path = os.path.dirname(epsic.__file__)
    test_recon_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.hdf')
    error = epsic.ptycho_utils.get_error(test_recon_path)
    nptest.assert_equal(error.shape, (2000,))
    
def test_get_probe_array():
    root_path = os.path.dirname(epsic.__file__)
    test_recon_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.hdf')
    probe = epsic.ptycho_utils.get_probe_array(test_recon_path)
    nptest.assert_equal(probe.shape, (256,256))
    
def test_get_obj_array():
    root_path = os.path.dirname(epsic.__file__)
    test_recon_path = os.path.join(root_path , 'tests', 'test_data', 'test_recon_sim.hdf')
    obj = epsic.ptycho_utils.get_obj_array(test_recon_path)
    nptest.assert_equal(obj.shape, (484,484))
    
    
    
    
