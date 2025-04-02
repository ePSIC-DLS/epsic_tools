from itertools import product
from math import floor
import logging

import numpy as np
import h5py
from hyperspy.signal import BaseSignal

formatter = logging.Formatter("%(asctime)s    %(process)5d %(processName)-12s %(threadName)-12s                   %(levelname)-8s %(pathname)s:%(lineno)d %(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set the debug log level.
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

# Make a logger for this module.
logger = logging.getLogger(__name__)


PIXEL_DEPTH_NPY_TYPE = {"U01": np.uint8,
                        "U08": np.uint8,
                        "U16": np.uint16,
                        "U32": np.uint32,
                        "U64": np.uint64,
                        }

PIXEL_DEPTH_NPY_TYPE_PROMOTED = {"U01": np.uint8,
                                 "U08": np.uint16,
                                 "U16": np.uint32,
                                 "U32": np.uint64,
                                 "U64": np.uint64,
                                 }
def _add_crosses(a):
    """
    Adds 3 pixel buffer cross to quad chip data.

    Parameters
    ----------
    a : numpy.ndarray
        Stack of raw frames or reshaped dask array object, prior to dimension reshaping, to insert
        3 pixel buffer cross into.

    Returns
    -------
    b : numpy.ndarray
        Stack of frames or reshaped 4DSTEM object including 3 pixel buffer cross in the diffraction plane.
    """
    original_shape = a.shape

    if len(original_shape) == 4:
        a = a.reshape(
            original_shape[0] * original_shape[1], original_shape[2], original_shape[3]
        )

    a_half = int(original_shape[-1] / 2), int(original_shape[-2] / 2)
    # Define 3 pixel wide cross of zeros to pad raw data
    if len(original_shape) == 4:
        z_array = np.zeros(
            (original_shape[0] * original_shape[1], original_shape[-2], 3),
            dtype=a.dtype,
        )
        z_array2 = np.zeros(
            (original_shape[0] * original_shape[1], 3, original_shape[-1] + 3),
            dtype=a.dtype,
        )
    else:
        z_array = np.zeros((original_shape[0], original_shape[-2], 3), dtype=a.dtype)
        z_array2 = np.zeros(
            (original_shape[0], 3, original_shape[-1] + 3), dtype=a.dtype
        )

    # Insert blank cross into raw data
    b = np.concatenate((a[:, :, : a_half[1]], z_array, a[:, :, a_half[1] :]), axis=-1)

    b = np.concatenate((b[:, : a_half[0], :], z_array2, b[:, a_half[0] :, :]), axis=-2)

    if len(original_shape) == 4:
        b = b.reshape(
            original_shape[0],
            original_shape[1],
            original_shape[2] + 3,
            original_shape[3] + 3,
        )

    return b

def STEM_flag_dict(exp_times_list):
    """
    Determines whether a .mib file contains STEM or TEM data and how many
    frames to skip due to triggering from a list of exposure times.

    Parameters
    ----------
    exp_times_list : list
        List of exposure times extracted from a .mib file.

    Returns
    -------
    output : dict
        Dictionary containing - STEM_flag, scan_X, exposure_time,
                                number_of_frames_to_skip, flyback_times
    Example
    -------
    {'STEM_flag': 1,
     'scan_X': 256,
     'exposure time': 0.0007,
     'number of frames_to_skip': 136,
     'flyback_times': [0.0392, 0.0413, 0.012625, 0.042]}
    """
    output = {}
    times_set = set(exp_times_list)
    # If single exposure times in header, treat as TEM data.
    if len(times_set) == 1:
        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = list(times_set)
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # In case exp times not appearing in header treat as TEM data
    elif len(times_set) == 0:

        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = None
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # Otherwise, treat as STEM data.
    else:
        STEM_flag = 1
        # Check that the smallest time is the majority of the values
        exp_time = max(times_set, key=exp_times_list.count)
        if exp_times_list.count(exp_time) < int(0.9 * len(exp_times_list)):
            logger.debug('Something wrong with the triggering!')
        peaks = [i for i, e in enumerate(exp_times_list) if e > 5 * exp_time]
        # Diff between consecutive elements of the array
        lines = np.ediff1d(peaks)
        if len(set(lines)) == 1:
            scan_X = lines[0]
            frames_to_skip = peaks[0]
            # if frames_to_skip is 1 less than scan_X we do not need to skip any frames
            # if frames_to_skip == peaks[0]:
            #     frames_to_skip = 0
        else:
            # Assuming the last element to be the line length
            scan_X = lines[-1]
            check = np.ravel(np.where(lines == scan_X, True, False))
            # Checking line lengths
            start_ind = np.where(check == False)[0][-1] + 2
            frames_to_skip = peaks[start_ind]

        flyback_times = list(times_set)
        flyback_times.remove(exp_time)
        output['STEM_flag'] = STEM_flag
        output['scan_X'] = scan_X
        output['exposure time'] = exp_time
        output['number of frames_to_skip'] = frames_to_skip
        output['flyback_times'] = flyback_times

    return output


def get_scan_y(nframe, start_frame, scan_x):
    """Return the number of rows (scan y)."""
    return floor((nframe - start_frame) / scan_x)


def bright_flyback_frame(start_frame, scan_y, scan_x):
    """Return the indices of flyback frame in a stack (the first column)."""
    return start_frame + np.arange(scan_y)*scan_x


def grid_chunk_offsets(scan_shape):
    """A generator for grid chunk offsets."""
    scan_y = scan_shape[0]
    scan_x = scan_shape[1]

    for syx in product(range(scan_y), range(scan_x)):
        yield (syx[0], syx[1], 0, 0)


def stack_chunk_offsets(scan_shape):
    """A generator for stack chunk offsets."""
    scan_yx = scan_shape[0]

    for syx in range(scan_yx):
        yield (syx, 0, 0)


def binned_nav_indices(linear_index, ncol, bw, row_shift=0, col_shift=0):
    """Get the bin indices from a linear index.

    If I have the 5x5 array below and bin it by 2 (bw):

    -------------------------------
    |  0  |  1  |  2  |  3  |  4  |
    -------------------------------
    |  5  |  6  |  7  |  8  |  9  |
    -------------------------------
    |  10 |  11 |  12 |  13 |  14 |
    -------------------------------
    |  15 |  16 |  17 |  18 |  19 |
    -------------------------------
    |  20 |  21 |  22 |  23 |  24 |
    -------------------------------

    (the indices are linear)

    the binned array will be a 2x2 array, and if row_shift and col_shift
    are both 1, the above data will be in the following bin:

    -----------------------------
    |             |             |
    |  6,7,11,12  |  8,9,13,14  |
    |             |             |
    -----------------------------
    |             |             |
    | 16,17,21,22 | 18,19,23,24 |
    |             |             |
    -----------------------------

    the row_shift and col_shift is equivalent to the number of cropping
    from top row and left column, respectively.

    The function returns:
        - (0, 0) for linear indices 6, 7, 11, 12;
        - (0, 1) for linear indices 8, 9, 13, 14;
        - (1, 0) for linear indices 16, 17, 21, 22;
        - (1, 1) for linear indices 18, 19, 23, 24;

    Parameters
    ----------
    linear_index : int
        the index (linear) of the 2D array to be binned
    ncol : int
        the number of column of the 2D array
    bw : int
        the width of the bin
    row_shift, col_shift : int
        the number of rows and columns to be shifted, equivalent to
        cropping to the same number of top rows and left columns
        respectively

    Returns
    -------
    binned_row_idx, binned_col_idx
        the row and column indices of the bin
    """

    norm_idx = linear_index - col_shift - ncol*row_shift

    binned_row_idx = norm_idx // (ncol*bw)
    binned_col_idx = (norm_idx % ncol) // bw

    return binned_row_idx, binned_col_idx

def empty_hspy_hdf5(output_path, shape, data_dict=None):
    """Create an empty hdf5 file following HyperSpy hierarchy with metadata.

    Parameters
    ----------
    output_path : str
        the output hdf5 file
    shape : tuple
        the shape of the dataset, with 4 members (scan_y, scan_x, det_y,
        det_x).
    data_dict : dict, optional
        a dictionary contains some values for the metadata

    Returns
    -------
    the dataset key where the actual data will be saved.

    """
    axes_dict = _get_axes_dict(shape)

    # construct metadata dictionary
    metadata_dict = {"Signal": {}}
    metadata_dict["Signal"]["flip"] = "True"
    # this is what pyxem would have set at the end
    # so skip setting "STEM" or "TEM"
    metadata_dict["Signal"]["signal_type"] = "electron_diffraction"

    if data_dict is not None:
        metadata_dict["Signal"]["scan_X"] = data_dict["scan_X"]
        metadata_dict["Signal"]["frames_number_skipped"] = data_dict["number of frames_to_skip"]
        # in ms
        metadata_dict["Signal"]["exposure_time"] = data_dict["exposure time"]
        metadata_dict["Signal"]["flyback_times"] = data_dict["flyback_times"]

    # fake HyperSpy signal
    # its content and dims are not important (it is not saved)
    s = BaseSignal(np.empty((1,2,3,4)),
                   axes=axes_dict,
                   metadata=metadata_dict
                   )

    # this creates a file consistent with HyerSpy signal
    # "write_dataset=False" to skip writing the fake data
    # we just want the hierarchy
    s.save(output_path,
           overwrite=True,
           file_format="HSPY",
           write_dataset=False)

    # inspect the created hdf5 file and return the dataset where "data"
    # should be saved
    with h5py.File(output_path, "r") as f:
        # this should be fixed by Hyperspy
        expg = f["/Experiments"]
        # this could depend on "title" in the metadata
        # and it has one member only
        dset_name = list(expg.keys())[0]

    return f"/Experiments/{dset_name}/data"

def _get_axes_dict(shape):
    if len(shape) == 3:
        # a stack
        syx = shape[0]
        dety = shape[1]
        detx = shape[2]

        # construct HyperSpy axes dictionary
        ax_syx = {"size": syx, "navigate": True}
        ax_dy = {"size": dety}
        ax_dx = {"size": detx}

        return [ax_syx, ax_dy, ax_dx]
    elif len(shape) == 4:
        # a grid
        sy = shape[0]
        sx = shape[1]
        dety = shape[2]
        detx = shape[3]

        # construct HyperSpy axes dictionary
        ax_sy = {"size": sy, "navigate": True}
        ax_sx = {"size": sx, "navigate": True}
        ax_dy = {"size": dety}
        ax_dx = {"size": detx}

        return [ax_sy, ax_sx, ax_dy, ax_dx]

    msg = "It only supports saving 3D or 4D data"
    raise ValueError(msg)
