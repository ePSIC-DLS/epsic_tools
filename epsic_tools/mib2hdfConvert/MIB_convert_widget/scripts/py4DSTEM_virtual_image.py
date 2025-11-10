import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import tifffile
import py4DSTEM
import hyperspy.api as hs

print(py4DSTEM.__version__)

def Meta2Config(acc,nCL,aps):
    '''This function converts the meta data from the 4DSTEM data set into parameters to be used in a ptyREX json file'''

    '''The rotation angles noted here are from ptychographic reconstructions which have been successful. see the 
    following directory for example reconstruction from which these values are derived:
     /dls/science/groups/imaging/ePSIC_ptychography/experimental_data'''
    if acc == 80e3:
        rot_angle = 238.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 41.65e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 31.74e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 24.80e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =15.44e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    elif acc == 200e3:
        rot_angle = 90
        print('Rotation angle = ' + str(rot_angle) +' Warning: This rotation angle need further calibration')
        if aps == 1:
            conv_angle = 37.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 28.8e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 22.4e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle = 14.0
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 5:
            conv_angle = 6.4
            print('Condenser aperture size is 10um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
    elif acc == 300e3:
        rot_angle = -85.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 44.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 34.1e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 26.7e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =16.7e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    else:
        print('Rotation angle for this acceleration voltage is unknown, please collect calibration data. Rotation angle being set to zero')
        rot_angle = 0

    '''this is incorrect way of calucating the actual camera length but okay for this prototype code'''
    '''TODO: add py4DSTEM workflow which automatic determines the camera length from a small amount of reference data and the known convergence angle'''
    camera_length = 1.5*nCL
    print('camera length estimated to be ' + str(camera_length))

    return rot_angle,camera_length,conv_angle


# load the information
info_path = sys.argv[1]
index = int(sys.argv[2])
info = {}
with open(info_path, 'r') as f:
    for line in f:
        tmp = line.split(" ")
        if tmp[0] == 'to_convert_paths':
            info[tmp[0]] = line.split(" = ")[1].split('\n')[:-1]
            print(tmp[0], line.split(" = ")[1].split('\n')[:-1])
        else:
            info[tmp[0]] = tmp[-1].split("\n")[0]
            print(tmp[0], tmp[-1].split("\n")[0])

            
data_path = eval(info['to_convert_paths'][0])[index]
meta_path = data_path[:-10]+".hdf"
data_name = data_path.split("/")[-1].split(".")[0]
time_stamp = data_name[:-5]
if info["mask_path"] == '':
    mask_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/29042024_12bitmask.h5'
else:
    mask_path = info["mask_path"]
save_dir = os.path.dirname(data_path) # directory for saving the results
print(meta_path)
print(data_path)
print(data_name)
print(mask_path)

device = info["device"]

if meta_path != '':
    try:
        with h5py.File(meta_path,'r') as f:
            print("----------------------------------------------------------")
            print(f['metadata']["defocus(nm)"])
            print(f['metadata']["defocus(nm)"][()])
            defocus_exp = f['metadata']["defocus(nm)"][()]*10 # Angstrom
            print("----------------------------------------------------------")
            print(f['metadata']["ht_value(V)"])
            print(f['metadata']["ht_value(V)"][()])
            HT = f['metadata']["ht_value(V)"][()]
            print("----------------------------------------------------------")
            print(f['metadata']["step_size(m)"])
            print(f['metadata']["step_size(m)"][()])
            scan_step = f['metadata']["step_size(m)"][()] * 1E10 # Angstrom
            print("----------------------------------------------------------")
            meta_values = f['metadata']
            print(meta_values['aperture_size'][()])
            print(meta_values['nominal_camera_length(m)'][()])
            print(meta_values['ht_value(V)'][()])
            acc = meta_values['ht_value(V)'][()]
            nCL = meta_values['nominal_camera_length(m)'][()]
            aps = meta_values['aperture_size'][()]
            rot_angle,camera_length,conv_angle = Meta2Config(acc, nCL, aps)
    except:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            HT = metadata['process']['common']['source']['energy'][0]
            print("HT: ", HT)
            defocus_exp = metadata['experiment']['optics']['lens']['defocus'][0]*1E10
            print("defocus: ", defocus_exp)
            scan_step = metadata['process']['common']['scan']['dR'][0]*1E10
            print("scan step: ", scan_step)

if mask_path != '':
    try:
        with h5py.File(mask_path,'r') as f:
            mask = f['data']['mask'][()]

    except:
        with h5py.File(mask_path,'r') as f:
            mask = f['root']['np.array']['data'][()]
    
    mask = np.invert(mask)
    mask = mask.astype(np.float32)
    
    print(type(mask))
    print(mask.dtype)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(mask)
    fig.tight_layout()
    plt.savefig(save_dir+"/mask.png")

### VERY IMPORTANT VARIABLE ###
semiangle = conv_angle*1000 # mrad
### ####################### ###

if data_path.split(".")[-1] == "hspy": 
# This is for the simulated 4DSTEM data using 'submit_abTEM_4DSTEM_simulation.ipynb'
# stored in /dls/science/groups/e02/Ryu/RYU_at_ePSIC/multislice_simulation/submit_abtem/submit_abtem_4DSTEM_simulation.ipynb
    original_stack = hs.load(data_path)
    print(original_stack)
    n_dim = len(original_stack.data.shape)
    scale = []
    origin = []
    unit = []
    size = []
    
    
    for i in range(n_dim):
        print(original_stack.axes_manager[i].scale, original_stack.axes_manager[i].offset, original_stack.axes_manager[i].units, original_stack.axes_manager[i].size)
        scale.append(original_stack.axes_manager[i].scale)
        origin.append(original_stack.axes_manager[i].offset)
        unit.append(original_stack.axes_manager[i].units)
        size.append(original_stack.axes_manager[i].size)
    
    HT = eval(original_stack.metadata["HT"])
    defocus_exp = eval(original_stack.metadata["defocus"])
    semiangle = eval(original_stack.metadata["semiangle"])
    scan_step = scale[0]
    print("HT: ", HT)
    print("experimental defocus: ", defocus_exp)
    print("semiangle: ", semiangle)
    print("scan step: ", scan_step)
    original_stack = original_stack.data
    det_name = 'ePSIC_EDX'
    data_key = 'Experiments/__unnamed__/data'


elif data_path.split(".")[-1] == "hdf" or data_path.split(".")[-1] == "hdf5" or data_path.split(".")[-1] == "h5":
    # try:
    #     original_stack = hs.load(data_path, reader="HSPY", lazy=True)
    #     print(original_stack)
    #     original_stack = original_stack.data
    try:    
        f = h5py.File(data_path,'r')
        print(f)
        original_stack = f['Experiments']['__unnamed__']['data'][:]
        f.close()
        det_name = 'ePSIC_EDX'
        data_key = 'Experiments/__unnamed__/data'
    
    except:
        f = h5py.File(data_path,'r')
        print(f)
        original_stack = f['data']['frames'][:]
        f.close()
        det_name = 'pty_data'
        data_key = "data/frames"
    
    #performing a check to see if the data in single chip or quad chip
    n_dim = np.shape(original_stack)
    print("checking number merlin chips used \n")
    if n_dim[3] == 256:
        chip = 1   
    else:
        chip = 4
    print("... the number of chips used is " + str(chip)+"\n")
    
    if chip == 1:
        mask = mask[0:256,0:256]
        print("subsection of the mask has been taken from top left hand corner\n")

elif data_path.split(".")[-1] == "dm4":
    original_stack = hs.load(data_path)
    print(original_stack)
    n_dim = len(original_stack.data.shape)
    scale = []
    origin = []
    unit = []
    size = []
    
    for i in range(n_dim):
        print(original_stack.axes_manager[i].scale, original_stack.axes_manager[i].offset, original_stack.axes_manager[i].units, original_stack.axes_manager[i].size)
        scale.append(original_stack.axes_manager[i].scale)
        origin.append(original_stack.axes_manager[i].offset)
        unit.append(original_stack.axes_manager[i].units)
        size.append(original_stack.axes_manager[i].size)
    
    HT = 1000 * original_stack.metadata['Acquisition_instrument']['TEM']['beam_energy']
    scan_step = scale[0] * 10
    defocus_exp = eval(info['defocus'])
    print("HT: ", HT)
    print("experimental defocus: ", defocus_exp)
    print("semiangle: ", semiangle)
    print("scan step: ", scan_step)
    original_stack = original_stack.data
    original_stack = original_stack.astype(np.float32)
    original_stack -= np.min(original_stack)
    original_stack /= np.max(original_stack)
    original_stack *= 128.0
    # det_name = 'ePSIC_EDX'
    # data_key = 'Experiments/__unnamed__/data' 

else:
    print("Wrong data format!")

original_stack = original_stack.astype(np.float32)
print(original_stack.dtype)
print(original_stack.shape)
print(np.min(original_stack), np.max(original_stack))

# masking
if mask_path != '' and type(mask) == np.ndarray:
    for i in range(original_stack.shape[0]):
        for j in range(original_stack.shape[1]):
            original_stack[i, j] = np.multiply(original_stack[i, j], mask)


dataset = py4DSTEM.DataCube(data=original_stack)
print("original dataset")
print(dataset)

del original_stack

dataset.get_dp_mean()
dataset.get_dp_max()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(dataset.tree('dp_mean')[:, :], cmap='jet')
fig.tight_layout()
plt.savefig(save_dir+"/pacbed.png")

probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(thresh_lower=eval(info["disk_lower_thresh"]), thresh_upper=eval(info["disk_upper_thresh"]), N=100, plot=True)
plt.savefig(save_dir+"/disk_detection.png")

dataset.calibration._params['Q_pixel_size'] = semiangle / probe_radius_pixels
dataset.calibration._params['Q_pixel_units'] = "mrad"
dataset.calibration._params['R_pixel_size'] = scan_step
dataset.calibration._params['R_pixel_units'] = "A"

print(dataset)
print(dataset.calibration)

# Make a virtual bright field and dark field image
center = (probe_qx0, probe_qy0)
radius_BF = probe_radius_pixels
radii_DF = (probe_radius_pixels, int(dataset.Q_Nx/2))

dataset.get_virtual_image(
    mode = 'circle',
    geometry = (center,radius_BF),
    name = 'bright_field',
    shift_center = False,
)
dataset.get_virtual_image(
    mode = 'annulus',
    geometry = (center,radii_DF),
    name = 'dark_field',
    shift_center = False,
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(dataset.tree('bright_field')[:, :], cmap="inferno")
ax[0].set_title("BF image")
ax[1].imshow(dataset.tree('dark_field')[:, :], cmap="inferno")
ax[1].set_title("ADF image [%.1f, %.1f] mrad"%(radii_DF[0]*dataset.Q_pixel_size, radii_DF[1]*dataset.Q_pixel_size))
fig.tight_layout()
plt.savefig(save_dir+"/STEM_image.png")
# tifffile.imwrite(save_dir+"/BF_image.tif", dataset.tree('bright_field')[:, :])
# tifffile.imwrite(save_dir+"/ADF_image.tif", dataset.tree('dark_field')[:, :])
vBF = dataset.tree('bright_field')[:, :]
vADF = dataset.tree('dark_field')[:, :]

if eval(info["DPC"]):
    dpc = py4DSTEM.process.phase.DPC(
        datacube=dataset,
        energy = HT,
    ).preprocess(force_com_rotation=rot_angle)
    plt.savefig(save_dir+"/DPC_optimization.png")

    dpc.reconstruct(
        max_iter=8,
        store_iterations=True,
        reset=True,
        gaussian_filter_sigma=0.1,
        gaussian_filter=True,
        q_lowpass=eval(info["dpc_lpass"]),
        q_highpass=eval(info["dpc_hpass"])
    ).visualize(
        iterations_grid='auto',
        figsize=(16, 10)
    )
    plt.savefig(save_dir+"/iDPC_image.png")

    dpc_cor = py4DSTEM.process.phase.DPC(
        datacube=dataset,
        energy=HT,
        verbose=False,
    ).preprocess(
        force_com_rotation = np.rad2deg(dpc._rotation_best_rad),
        force_com_transpose = False,
    )
    plt.savefig(save_dir+"/corrected_DPC_optimization.png")

    dpc_cor.reconstruct(
        max_iter=8,
        store_iterations=True,
        reset=True,
        gaussian_filter_sigma=0.1,
        gaussian_filter=True,
        q_lowpass=eval(info["dpc_lpass"]),
        q_highpass=eval(info["dpc_hpass"])
    ).visualize(
        iterations_grid='auto',
        figsize=(16, 10)
    )
    plt.savefig(save_dir+"/corrected_iDPC_image.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(dpc._com_normalized_y, cmap="bwr")
    ax[0].set_title("CoMx")
    ax[1].imshow(dpc._com_normalized_x, cmap="bwr")
    ax[1].set_title("CoMy")
    ax[2].imshow(np.sqrt(dpc._com_normalized_y**2 + dpc._com_normalized_x**2), cmap="inferno")
    ax[2].set_title("Magnitude of CoM")
    fig.tight_layout()
    plt.savefig(save_dir+"/CoM_image.png")


    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(dpc_cor._com_normalized_y, cmap="bwr")
    ax[0].set_title("CoMx - rotation corrected")
    ax[1].imshow(dpc_cor._com_normalized_x, cmap="bwr")
    ax[1].set_title("CoMy - rotation corrected")
    ax[2].imshow(np.sqrt(dpc_cor._com_normalized_y**2 + dpc_cor._com_normalized_x**2), cmap="inferno")
    ax[2].set_title("Magnitude of CoM - rotation corrected")
    fig.tight_layout()
    plt.savefig(save_dir+"/corrected_CoM_image.png")


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(dpc.object_phase, cmap="inferno")
    ax[0].set_title("iCoM")
    ax[1].imshow(dpc_cor.object_phase, cmap="inferno")
    ax[1].set_title("iCoM - rotation corrected")
    fig.tight_layout()
    plt.savefig(save_dir+"/iDPC_comparison.png")
    # tifffile.imwrite(save_dir+"/iDPC_corrected.tif", dpc_cor.object_phase)

    
if eval(info["parallax"]):
    print("rebin by 2, implemented for parallax to reduce the memory usage")
    dataset.bin_Q(2) 
    parallax = py4DSTEM.process.phase.Parallax(
        datacube=dataset,
        energy = HT,
        device = device, 
        verbose = True
    ).preprocess(
        threshold_intensity=eval(info["disk_upper_thresh"]),
        normalize_images=True,
        plot_average_bf=False,
        edge_blend=8,
    )

    parallax = parallax.reconstruct(
        reset=True,
        regularizer_matrix_size=(1,1),
        regularize_shifts=True,
        running_average=True,
        min_alignment_bin = 16,
        num_iter_at_min_bin = 2,
    )
    plt.savefig(save_dir+"/parallax_rough.png")

    parallax.show_shifts()
    plt.savefig(save_dir+"/parallax_shift.png")

    parallax.subpixel_alignment(
        # kde_upsample_factor=2.0,
        kde_sigma_px=0.125,
        plot_upsampled_BF_comparison=True,
        plot_upsampled_FFT_comparison=True,
    )
    plt.savefig(save_dir+"/parallax_subpixel_alignment.png")

    parallax.aberration_fit(
        plot_CTF_comparison=True,
    )
    plt.savefig(save_dir+"/parallax_CTF.png")

    parallax.aberration_correct(figsize=(5, 5))
    plt.savefig(save_dir+"/parallax_aberration_corrected.png")

    # Get the probe convergence semiangle from the pixel size and estimated radius in pixels
    semiangle_cutoff_estimated = dataset.calibration.get_Q_pixel_size() * probe_radius_pixels
    print('semiangle cutoff estimate = ' + str(np.round(semiangle_cutoff_estimated, decimals=1)) + ' mrads')

    # Get the estimated defocus from the parallax reconstruction - note that defocus dF has the opposite sign as the C1 aberration!
    defocus_estimated = -parallax.aberration_C1
    print('estimated defocus         = ' + str(np.round(defocus_estimated)) + ' Angstroms')

    rotation_degrees_estimated = np.rad2deg(parallax.rotation_Q_to_R_rads)
    print('estimated rotation        = ' + str(np.round(rotation_degrees_estimated)) + ' deg')
    
#     tifffile.imwrite(save_dir+"/parallax_aberration_corrected.tif", parallax.recon_phase_corrected)
    
#     with open(save_dir+"/parallax_estimates.txt", 'w') as fp:
#         fp.write('semiangle cutoff estimate = ' + str(np.round(semiangle_cutoff_estimated, decimals=1)) + ' mrads\n')
#         fp.write('estimated defocus         = ' + str(np.round(defocus_estimated)) + ' Angstroms\n')
#         fp.write('estimated rotation        = ' + str(np.round(rotation_degrees_estimated)) + ' deg')


with h5py.File(save_dir+"/"+time_stamp+"_py4DSTEM_processed.hdf5", 'w') as vi_save:
    vi_save.create_dataset('data_path', data=data_path)
    vi_save.create_dataset('defocus(nm)', data=defocus_exp)
    vi_save.create_dataset('ht_value(V)', data=acc)
    vi_save.create_dataset('nominal_camera_length(m)', data=nCL)
    vi_save.create_dataset('aperture_size', data=aps)
    vi_save.create_dataset('convergence_angle(mrad)', data=semiangle)
    vi_save.create_dataset('pixel_size(Å)', data=scan_step)
    vi_save.create_dataset('vBF', data=vBF)
    vi_save.create_dataset('vADF', data=vADF)
    if eval(info["DPC"]):
        vi_save.create_dataset('CoMx', data=dpc._com_normalized_y)
        vi_save.create_dataset('CoMy', data=dpc._com_normalized_x)
        vi_save.create_dataset('iDPC', data=dpc_cor.object_phase)

    if eval(info["parallax"]):
        vi_save.create_dataset('parallax', data=parallax._recon_phase_corrected)
        vi_save.create_dataset('parallax_estimated_defocus(Å)', data=defocus_estimated)
        vi_save.create_dataset('parallax_estimated_rotation(deg)', data=rotation_degrees_estimated)