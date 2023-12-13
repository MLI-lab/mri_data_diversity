"""
Converts the original Stanford Fullysampled 3D FSE Knees dataset to a format 
easily accessed by fastMRI modules.

kspace data is scaled by 1e-7 
target is obtained via RSS reconstruction

Stanford 2D FSE dataset: http://mridata.org/list?project=Stanford%20Fullysampled%203D%20FSE%20Knees
https://github.com/z-fabian/MRAugment/blob/master/data/stanford/convert_stanford3d.py
"""
import os
import numpy as np
import pathlib
import h5py, ismrmrd
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser

import xml.etree.ElementTree as etree
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def isrmrmd_user_param_to_dict(header):
    """
    Store ISMRMRD header user parameters in a dictionary.
    Parameter
    ---------
    header : ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    Returns
    -------
    dict
        Dictionary containing custom user parameters
    """
    user_dict = {}
    user_long = list(header.userParameters.userParameterLong)
    user_double = list(header.userParameters.userParameterDouble)
    user_string = list(header.userParameters.userParameterString)
    user_base64 = list(header.userParameters.userParameterBase64)

    for entry in user_long + user_double + user_string + user_base64:
        user_dict[entry.name] = entry.value_

    return user_dict

def load_ismrmrd_to_np(file_name, verbose=False):
    """
    Load data from an ISMRMRD file to a numpy array.
    Raw data from the ISMRMRD file is loaded into a numpy array. If the ISMRMRD file includes the array 'rec_std' that contains the standard deviation of the noise, this information is used to pre-whiten the k-space data. If applicable, a basic phase correction is performed on the loaded k-space data.
    Parameters
    ----------
    file_name : str
        Name of ISMRMRD file
    verbose : bool, optional
        Turn on/off verbose print out
    Returns
    -------
    np.array
        k-space data in an np.array of dimensions [phase, echo, slice, coils, kz, ky, kx]
    ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    """
    dataset = ismrmrd.Dataset(file_name, create_if_needed=False)
    xml_header = dataset.read_xml_header()
    header = ismrmrd.xsd.CreateFromDocument(xml_header)
    param_dict = isrmrmd_user_param_to_dict(header)

    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_kz = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum

    num_channels = header.acquisitionSystemInformation.receiverChannels
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_echoes = header.encoding[0].encodingLimits.contrast.maximum + 1
    num_phases = header.encoding[0].encodingLimits.phase.maximum + 1

    is_fse_with_calib = False

    chop_y = 1 - int(param_dict.get('ChopY', 1))
    chop_z = 1 - int(param_dict.get('ChopZ', 1))

    try:
        rec_std = dataset.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)

    if verbose:
        print("Data dims: (%d, %d, %d, %d, %d, %d, %d)" % (num_kx, num_ky, num_kz,
                                                           num_channels, num_slices,
                                                           num_echoes, num_phases))
    kspace = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                       num_kz, num_ky, num_kx], dtype=np.complex64)

    if is_fse_with_calib:
        echo_train = np.zeros([num_phases, num_echoes, num_slices, 1, num_kz, num_ky, 1],
                              dtype=np.uint)
        kspace_fse_cal = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                                   num_segments, num_kx], dtype=np.complex64)
        echo_train_fse_cal = np.zeros([num_phases, num_echoes, num_slices, 1, num_segments, 1],
                                      dtype=np.uint)

    max_slice = 0
    wrap = lambda x: x
    if verbose:
        print("Loading data...")
        wrap = tqdm
    try:
        num_acq = dataset.number_of_acquisitions()
    except:
        print("Unable to determine number of acquisitions! Empty?")
        return
    for i in wrap(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1 # pylint: disable=E1101
        i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_echo = acq.idx.contrast           # pylint: disable=E1101
        i_phase = acq.idx.phase             # pylint: disable=E1101
        i_slice = acq.idx.slice             # pylint: disable=E1101
        if i_slice > max_slice:
            max_slice = i_slice
        sign = (-1) ** (i_ky * chop_y + i_kz * chop_z)
        data = np.matmul(opt_mat.T, acq.data) * sign
        if i_kz < num_kz:
            i_segment = acq.idx.segment # pylint: disable=E1101
            if i_ky < num_ky:
                kspace[i_phase, i_echo, i_slice, :, i_kz, i_ky, :] = data
                if is_fse_with_calib:
                    echo_train[i_phase, i_echo, i_slice, 0, i_kz, i_ky, 0] = i_segment
            elif is_fse_with_calib:
                kspace_fse_cal[i_phase, i_echo, i_slice, :, i_ky - num_ky, :] = data
                echo_train_fse_cal[i_phase, i_echo, i_slice, 0, i_ky - num_ky, 0] = i_segment
    dataset.close()

    max_slice += 1
    if num_slices != max_slice:
        if verbose:
            print("Actual number of slices different: %d/%d" % (max_slice, num_slices))
        kspace = kspace[:, :, :max_slice, :, :, :, :]
        if is_fse_with_calib:
            echo_train = echo_train[:, :, :max_slice, :, :, :, :]
            kspace_fse_cal = kspace_fse_cal[:, :, :max_slice, :, :, :]
            echo_train_fse_cal = echo_train_fse_cal[:, :, :max_slice, :, :, :]

    if is_fse_with_calib:
        if verbose:
            print("FSE phase correction...")
        if 0:
            print("writing files for debugging...")
            cfl.write("kspace", kspace)
            cfl.write("echo_train", echo_train)
            cfl.write("kspace_fse_cal", kspace_fse_cal)
            cfl.write("echo_train_fse_cal", echo_train_fse_cal)
        kspace_cor = fse.phase_correction(kspace, echo_train, kspace_fse_cal, echo_train_fse_cal)
        # for debugging
        if 0:
            cfl.write("kspace_orig" , kspace)
        kspace = kspace_cor

    return kspace, header, xml_header

# Functions to generate reconstruction target
def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def kspace_to_target(x):
    return np.sqrt(np.sum(np.square(np.abs(ifft2_np(x))), axis=-3))

def center_crop(data, shape):
    """
    Adjusted code from fastMRI github repository
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape
    Returns:
        The center cropped image.
    """
    assert shape[0] > 0 and shape[1] > 0, "Desired shape of ({:d},{:d}) invalid. shape must contain positive integers".format(*shape)
    w_from = (data.shape[-2] - shape[0]) // 2
    w_from = w_from if w_from > 0 else 0
    
    h_from = (data.shape[-1] - shape[1]) // 2
    h_from = h_from if h_from > 0 else 0

    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

# Main conversion code
def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
        
    mri_files = list(pathlib.Path(args.input_dir).glob('*.h5'))
    print(mri_files)
    for i, mri_f in tqdm(zip(range(len(mri_files)), sorted(mri_files)), total=len(mri_files)):
        kspace, header, xml_header = load_ismrmrd_to_np(mri_f, verbose=False)
        # print('converting ', i+1, '/', len(mri_files))
        scaling = 1e7  # scale measurements to similar range as fastMRI
        kspace = kspace[0, 0, :, :, 0, :, :] / scaling
        kspace = kspace.transpose(0,1,-1,-2) # axis in fastmri dataset are swapped: x direction in ismrmrd header corresponds to y (vertical) direction in numpy array

        # Zero padding kspace if needed
        et_root = etree.fromstring(xml_header)
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        z = np.zeros((kspace.shape[0], kspace.shape[1], enc_size[0], enc_size[1])).astype(np.complex64)
        z[...,padding_left:padding_right] = kspace

        kspace = z

        # Center crop target to have recon size given in ismrmrd header
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
        )

        target = kspace_to_target(kspace)
        target = center_crop(target, recon_size).astype(np.float32)
        kspace = kspace.astype(np.complex64)
        
        save_file = os.path.join(args.output_dir, str(mri_f.name))
        data = h5py.File(save_file, 'w')
        data.create_dataset('ismrmrd_header', data=xml_header)
        data.create_dataset('kspace', data=kspace)
        data.create_dataset('reconstruction_rss', data=target)
        data.attrs.create('max', data=target.max())
        data.close()
    print('Finished.')
    
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the original Stanford3D dataset is located.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the converted dataset.",
    )
    
    args = parser.parse_args()
    return args
    
def run_cli():
    args = build_args()
    # run conversion
    cli_main(args)

if __name__ == "__main__":
    run_cli()