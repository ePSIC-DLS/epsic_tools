import os

import numpy as np
import h5py
import blosc
from hyperspy.signals import Signal2D
from hyperspy.misc.array_tools import rebin

# C extensions
from mib_prop import mib_props
try:
    from fast_binning import fast_bin
except ImportError:
    print("Cannot import fast binning function.")

from mib_flyback_utils import (PIXEL_DEPTH_NPY_TYPE,
                               PIXEL_DEPTH_NPY_TYPE_PROMOTED,
                               STEM_flag_dict,
                               get_scan_y,
                               bright_flyback_frame,
                               grid_chunk_offsets,
                               stack_chunk_offsets,
                               empty_hspy_hdf5,
                               binned_nav_indices,
                               _add_crosses,
                               )

def main():
    # 1-bit
    # mib_path = "/dls/e02/data/2024/mg37328-1/Merlin/Pd_ZnO/20240626_120132/20240626_120126_data.mib"

    # 6-bit
    mib_path = "/dls/e02/data/2024/mg35950-3/Merlin/PTO_SRO/20240705_120423/20240705_120418_data.mib"

    # 12-bit
    # mib_path = "/dls/e02/data/2023/cm33902-4/Merlin/au_xgrating_ptycho_12cmCL/20230822_135857/20230822_140140_data.mib"

    hdf5_path = "/dls/e02/data/2024/mg35950-3/processing/Merlin/PTO_SRO/20240705_120418/20240705_120418_data.hdf5"
    ibf_path = "/dls/e02/data/2024/mg35950-3/processing/Merlin/PTO_SRO/20240705_120418/result_iBF.jpg"
    bin_nav_path = "/dls/e02/data/2024/mg35950-3/processing/Merlin/PTO_SRO/20240705_120418/result_bin_nav_4.hspy"
    bin_sig_path = "/dls/e02/data/2024/mg35950-3/processing/Merlin/PTO_SRO/20240705_120418/result_bin_sig_4.hspy"


    no_reshaping = False
    use_fly_back = True
    known_shape = False
    Scan_X = 256
    Scan_Y = 256
    iBF = True
    bin_sig_flag = True
    bin_sig_factor = 4
    bin_nav_flag = True
    bin_nav_factor = 4
    add_cross = True


    # check provided reshaping options
    if sum([bool(no_reshaping), bool(use_fly_back), bool(known_shape)]) != 1:
        msg = (f"Only one of the options 'no_reshaping' ({no_reshaping}), "
               f"'use_fly_back' ({use_fly_back}) or 'known_shape' "
               f"({known_shape}) should be True.")
        raise ValueError(msg)

    # the Blosc filter registered ID (for h5py)
    compression_id = 32001
    # maximum compression level (0-9)
    clevel = 9
    # use "blosclz" compressor
    compressor = "blosclz"
    compressor_code = blosc.name_to_code(compressor)

    # set the block size for Blosc compression
    # the default is 32 kB and modfiied by clevel and others although
    # the docs said setting to L2 (but source code use L1 as a start?)
    # cache should provide some optimisation, some experimentation told
    # us that 1 kB is quite good for our compression setting
    try:
        blksz = int(os.environ["BLOSC_BLOCKSIZE"])
    except (KeyError, ValueError):
        blksz = 1024
    blosc.set_blocksize(blksz)
    print(f"Blosc block size: {blksz} B")

    try:
        blosc_nthreads = int(os.environ["BLOSC_NTHREADS"])
    except (KeyError, ValueError):
        # leave it as default, i.e. the maximum detected number
        blosc_nthreads = blosc.detect_number_of_cores()
    blosc.set_nthreads(blosc_nthreads)
    print(f"Blosc number of threads: {blosc_nthreads}")

    # fetch all useful information from the headers in the mib file
    mib_properties = mib_props(mib_path,
                               sequence_number=True,
                               header_bytes=True,
                               pixel_depth=True,
                               det_x=True,
                               det_y=True,
                               exposure_time_ns=True,
                               bit_depth=True,
                               )

    with open(mib_path, "rb") as mib:
        # determine header size, dtype and detector size from first header
        header_size = mib_properties["header_bytes"][0]
        det_y = mib_properties["det_y"][0]
        det_x = mib_properties["det_x"][0]
        dtype = PIXEL_DEPTH_NPY_TYPE[mib_properties["pixel_depth"][0]]

        # the number of bytes of each frame, including header
        num_frames = len(mib_properties["sequence_number"])
        stride = (header_size + det_y*det_x*np.dtype(dtype).itemsize)
        exposure_time_ms = mib_properties["exposure_time_ns"] * 1e-6

        if use_fly_back:
            # parse all headers to fetch metadata from every frame
            data_dict = STEM_flag_dict(exposure_time_ms.tolist())

            start_frame = data_dict["number of frames_to_skip"]
            scan_x = data_dict["scan_X"]
            scan_y = get_scan_y(num_frames, start_frame, scan_x)
            flyback_frames = bright_flyback_frame(start_frame, scan_y, scan_x)
            end_frame = start_frame + scan_y*scan_x
        else:
            data_dict = None

            start_frame = 0
            scan_x = Scan_X
            scan_y = Scan_Y
            flyback_frames = ()
            end_frame = num_frames

        if add_cross:
            # for 2x2 chip configuration, in order to represent
            # correct angular relationship in reciprocal space,
            # a 3 pixel-width cross is added
            width_cross = 3
        else:
            width_cross = 0

        if no_reshaping:
            # a stack
            mib_data_shape = (num_frames,
                              det_y + width_cross,
                              det_x + width_cross
                              )
            chunk_sz = (1, mib_data_shape[-2], mib_data_shape[-1])

            if iBF:
                msg = ("Saving the MIB frames as a stack does not support "
                       "saving the integrated bright field image.")
                raise ValueError(msg)

            if bin_nav_flag:
                msg = ("Saving the MIB frames as a stack does not support "
                       "binning across the navigation dimension.")
                raise ValueError(msg)
        elif use_fly_back:
            # use information from flyback
            # scan_x-1 to account for the flyback column
            mib_data_shape = (scan_y,
                              scan_x - 1,
                              det_y + width_cross,
                              det_x + width_cross
                              )
            chunk_sz = (1, 1, mib_data_shape[-2], mib_data_shape[-1])

        elif known_shape:
            # use the provided shape
            mib_data_shape = (scan_y,
                              scan_x,
                              det_y + width_cross,
                              det_x + width_cross
                              )
            chunk_sz = (1, 1, mib_data_shape[-2], mib_data_shape[-1])

            if scan_y*scan_x != num_frames:
                msg = (f"The requested scan dimension ({scan_y} x {scan_x}) "
                       f"does not match the total number of frames "
                       f"({num_frames}) in the MIB file.")
                raise ValueError(msg)
        else:
            msg = "You have to select one of the actions on reshaping!!!"
            raise ValueError(msg)

        if iBF:
            # guarantee the first two dims are scanning dims
            # fix uint32
            ibf_buffer = np.zeros(mib_data_shape[:2], dtype=np.uint32)

        if bin_sig_flag:
            dtype_bin_sig = PIXEL_DEPTH_NPY_TYPE_PROMOTED[mib_properties["pixel_depth"][0]]

            # determine top row (y) and left col (x) for cropping
            sig_to_cropy = mib_data_shape[-2] % bin_sig_factor
            sig_to_cropx = mib_data_shape[-1] % bin_sig_factor

        if bin_nav_flag:
            dtype_bin_nav = PIXEL_DEPTH_NPY_TYPE_PROMOTED[mib_properties["pixel_depth"][0]]

            # assume enough memory for holding this array
            arr_nav_binned = np.zeros((mib_data_shape[0] // bin_nav_factor,
                                       mib_data_shape[1] // bin_nav_factor,
                                       mib_data_shape[2],
                                       mib_data_shape[3]),
                                      dtype=dtype_bin_nav
                                      )

            # determine top row (y) and left col (x) for cropping
            nav_to_cropy = mib_data_shape[0] % bin_nav_factor
            nav_to_cropx = mib_data_shape[1] % bin_nav_factor

        # create an hdf5 file following HyperSpy hierarchy
        # without saving actual data but with metadata
        # "dset_path" is the key where the actual data will be saved
        dset_path = empty_hspy_hdf5(hdf5_path, mib_data_shape, data_dict)

        with h5py.File(hdf5_path, "r+") as hdf:
            # the last 3 indices:
            # compression level (0-9)
            # NOSHUFFLE, SHUFFLE, BITSHUFFLE
            # compressor, "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"
            if np.dtype(dtype).itemsize == 1:
                # for 1 byte data type don't use bit shuffle
                comopts_mib = (0, 0, 0, 0,
                               clevel, blosc.SHUFFLE, compressor_code)
            else:
                comopts_mib = (0, 0, 0, 0,
                               clevel, blosc.BITSHUFFLE, compressor_code)

            # create the output dataset
            dset = hdf.require_dataset(dset_path,
                                       shape=mib_data_shape,
                                       dtype=dtype,
                                       chunks=chunk_sz,
                                       compression=compression_id,
                                       compression_opts=comopts_mib,
                                       )

            # start at the beginning of the MIB file
            ptr = 0
            # ensure the dtype is big endian (from Merlin manual)
            dtype_be = np.dtype(dtype).newbyteorder(">")
            # some counters for tracking
            count_saved = 0
            count_skipped = 0
            count_sig_binned = 0
            count_nav_binned = 0
            # set up the chunk offset generator
            if no_reshaping:
                ch_offset_gen = stack_chunk_offsets(mib_data_shape)
            else:
                ch_offset_gen = grid_chunk_offsets(mib_data_shape)

            # for each frame, save to the hdf5 dataset
            for k in range(num_frames):
                # if count_saved == 10:
                    # break
                if (start_frame <= k < end_frame) and (k not in flyback_frames):
                    # point to the current frame
                    mib.seek(ptr)
                    # read the frame (header+data), this moves the file
                    # pointer by stride
                    frame = mib.read(stride)

                    # construct only the frame
                    arr = np.frombuffer(frame[header_size:], dtype=dtype_be)

                    # reshape to 2D for flipping
                    resh = arr.reshape(det_y, det_x)

                    # flip it to match those from pyxem (but why?)
                    resh = np.flipud(resh)

                    # reshape to the chunk frame size
                    if no_reshaping:
                        resh = resh.reshape(1, det_y, det_x)
                    else:
                        resh = resh.reshape(1, 1, det_y, det_x)

                    # add cross if needed
                    # final shape will match chunk_sz
                    if add_cross:
                        resh = _add_crosses(resh)

                    if bin_sig_flag:
                        # crop the signal (divisible by bin factor),
                        # return to 2D, make sure C-contiguous
                        sig_cropped = np.ascontiguousarray(np.squeeze(resh)[sig_to_cropy:, sig_to_cropx:])

                        if 2 <= bin_sig_factor <= 8:
                            try:
                                # fast bin if available
                                sig_binned = fast_bin(sig_cropped,
                                                      bin_sig_factor,
                                                      np.dtype(dtype_bin_sig).num
                                                      )
                            except NameError:
                                # fall back to the slower bin
                                sig_binned = rebin(sig_cropped,
                                                   scale=bin_sig_factor,
                                                   dtype=dtype_bin_sig)
                        else:
                            # fall back to the slower bin
                            sig_binned = rebin(sig_cropped,
                                               scale=bin_sig_factor,
                                               dtype=dtype_bin_sig)

                        # promote to dims of chunk frame size
                        if no_reshaping:
                            sig_binned = np.array(sig_binned, copy=False, ndmin=3)
                        else:
                            sig_binned = np.array(sig_binned, copy=False, ndmin=4)

                        # create bin sig file using the binned shape
                        # only necessary the first time
                        if count_sig_binned == 0:
                            if no_reshaping:
                                bin_sig_shape = (mib_data_shape[0],
                                                 *sig_binned.shape[-2:]
                                                 )
                                bin_sig_chunk_sz = (1,
                                                    *sig_binned.shape[-2:]
                                                    )
                            else:
                                bin_sig_shape = (*mib_data_shape[:2],
                                                 *sig_binned.shape[-2:]
                                                 )
                                bin_sig_chunk_sz = (1, 1,
                                                    *sig_binned.shape[-2:]
                                                    )

                            dset_bin_sig_key = empty_hspy_hdf5(bin_sig_path,
                                                               bin_sig_shape,
                                                               data_dict)

                            # remember to close the file handle!
                            f_sig_bin = h5py.File(bin_sig_path, "r+")

                            # set up compression options
                            if np.dtype(dtype_bin_sig).itemsize == 1:
                                # for 1 byte data type don't use bit shuffle
                                comopts_bin_sig = (0, 0, 0, 0,
                                                   clevel, blosc.SHUFFLE, compressor_code)
                            else:
                                comopts_bin_sig = (0, 0, 0, 0,
                                                   clevel, blosc.BITSHUFFLE, compressor_code)

                            # create the output dataset
                            dset_bin_sig = f_sig_bin.require_dataset(dset_bin_sig_key,
                                                                     shape=bin_sig_shape,
                                                                     dtype=dtype_bin_sig,
                                                                     chunks=bin_sig_chunk_sz,
                                                                     compression=compression_id,
                                                                     compression_opts=comopts_bin_sig,
                                                                     )

                            # set up the chunk offset generator
                            if no_reshaping:
                                ch_offset_bin_sig_gen = stack_chunk_offsets(bin_sig_shape)
                            else:
                                ch_offset_bin_sig_gen = grid_chunk_offsets(bin_sig_shape)

                    if bin_nav_flag:
                        # the if condition is equivalent to cropping
                        # such as [nav_to_cropy:, nav_to_cropx:], but operates
                        # on linear index (i.e. count_saved here)
                        if ((count_saved // mib_data_shape[1] >= nav_to_cropy) and
                            (count_saved % mib_data_shape[1] >= nav_to_cropx)):

                            # return the indices of this frame that
                            # belong to the current bin
                            bin_idx = binned_nav_indices(count_saved,
                                                         mib_data_shape[1],
                                                         bin_nav_factor,
                                                         row_shift=nav_to_cropy,
                                                         col_shift=nav_to_cropx,
                                                         )

                            # add the frame to the bin (binning)
                            arr_nav_binned[bin_idx[0], bin_idx[1], :, :] += resh[0, 0, :, :]
                            count_nav_binned += 1

                    # Blosc compression by using pointer
                    resh = np.ascontiguousarray(resh)
                    arr_compressed = blosc.compress_ptr(resh.__array_interface__["data"][0],
                                                        items=resh.size,
                                                        typesize=resh.itemsize,
                                                        clevel=comopts_mib[4],
                                                        shuffle=comopts_mib[5],
                                                        cname=compressor,
                                                        )


                    try:
                        # get the chunk offset for the dataset for this
                        # frame
                        chunk_offset = next(ch_offset_gen)
                    except StopIteration:
                        msg = ("There are more chunks than it should be! "
                               "Check the shape of the dataset.")
                        raise RuntimeError(msg)
                    else:
                        # write the frame to the offset for this frame
                        dset.id.write_direct_chunk(chunk_offset, arr_compressed)
                        count_saved += 1

                        # sum the frame for integrated bright field
                        if iBF:
                            ibf_buffer[chunk_offset[:2]] += resh.sum()


                    # save the bin sig to the dataset
                    if bin_sig_flag:
                        # Blosc compression by using pointer
                        sig_binned = np.ascontiguousarray(sig_binned)
                        sig_bin_compressed = blosc.compress_ptr(sig_binned.__array_interface__["data"][0],
                                                                items=sig_binned.size,
                                                                typesize=sig_binned.itemsize,
                                                                clevel=comopts_bin_sig[4],
                                                                shuffle=comopts_bin_sig[5],
                                                                cname=compressor,
                                                                )
                        try:
                            # get the chunk offset for the dataset for this
                            # frame
                            chunk_offset = next(ch_offset_bin_sig_gen)
                        except StopIteration:
                            msg = ("There are more chunks than it should be! "
                                   "Check the shape of the dataset.")
                            raise RuntimeError(msg)
                        else:
                            # write the binned sig to the offset
                            dset_bin_sig.id.write_direct_chunk(chunk_offset, sig_bin_compressed)
                            count_sig_binned += 1
                else:
                    # print(f"Skipped: frame index {k}, sequence number "
                          # f"{mib_properties['sequence_number'][k]}, "
                          # f"exposure time {mib_properties['exposure_time_ns'][k]} ns")
                    count_skipped += 1

                # point to the next frame
                ptr += stride

            print(f"Number of frames saved: {count_saved}")
            print(f"Number of frames skipped: {count_skipped}")

        # save iBF after rescale to 8-bit and change dtype
        if iBF:
            ibf = ibf_buffer - ibf_buffer.min()
            if ibf.max() != 0:
                ibf = ibf * (255 / ibf.max())
            ibf = ibf.astype(np.uint8)

            s_ibf = Signal2D(ibf)
            s_ibf.save(ibf_path, overwrite=True)

        # save binned array across the navigation axes (first 2)
        if bin_nav_flag:
            # although not streaming the chunk, using HyperSpy save
            # method still much slower than iterating the NumPy array
            # and direct chunk write with Blosc compression

            # dummy file
            dset_bin_nav_path = empty_hspy_hdf5(bin_nav_path,
                                                arr_nav_binned.shape,
                                                data_dict
                                                )

            with h5py.File(bin_nav_path, "r+") as bin_nav_hdf:
                # set up compression options
                if np.dtype(arr_nav_binned.dtype).itemsize == 1:
                    # for 1 byte data type don't use bit shuffle
                    comopts_bin_nav = (0, 0, 0, 0,
                                       clevel, blosc.SHUFFLE, compressor_code)
                else:
                    comopts_bin_nav = (0, 0, 0, 0,
                                       clevel, blosc.BITSHUFFLE, compressor_code)

                # create the output dataset
                chunk_sz = (1, 1, arr_nav_binned.shape[-2], arr_nav_binned.shape[-1])
                dset = bin_nav_hdf.require_dataset(dset_bin_nav_path,
                                                   shape=arr_nav_binned.shape,
                                                   dtype=arr_nav_binned.dtype,
                                                   chunks=chunk_sz,
                                                   compression=compression_id,
                                                   compression_opts=comopts_bin_nav,
                                                   )

                # set chunk offset generator
                ch_offset_gen = grid_chunk_offsets(arr_nav_binned.shape)

                # for each frame use direct chunk with compression
                for chunk_offset in ch_offset_gen:
                    row = chunk_offset[0]
                    col = chunk_offset[1]

                    arr = arr_nav_binned[row, col, :, :].reshape(chunk_sz)

                    # Blosc compression by using pointer
                    arr = np.ascontiguousarray(arr)
                    arr_compressed = blosc.compress_ptr(arr.__array_interface__["data"][0],
                                                        items=arr.size,
                                                        typesize=arr.itemsize,
                                                        clevel=comopts_bin_nav[4],
                                                        shuffle=comopts_bin_nav[5],
                                                        cname=compressor,
                                                        )

                    # write the frame to the offset for this frame
                    dset.id.write_direct_chunk(chunk_offset, arr_compressed)

        if bin_sig_flag:
            f_sig_bin.close()

if __name__ == "__main__":
    main()
