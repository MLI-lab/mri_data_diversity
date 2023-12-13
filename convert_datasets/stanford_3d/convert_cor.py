"""
Retrospectively synthesized coronal views of the stanford 3D dataset.
Dataformat for use with fastMRI code.
"""

import os
import numpy as np
import pathlib
import h5py
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

# Functions to generate reconstruction target
def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def fft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

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
        with h5py.File(mri_f, "r") as hf:
            k = hf['kspace'][()]
            i = ifft2_np(k)
            i_cor = i.transpose(3,1,2,0)
            k_cor = fft2_np(i_cor)
            kspace = np.fft.fftshift(k_cor, axes=-1) # kspace for new view is shifted for some reason, i.e, low. freqs are at the edges
            target = kspace_to_target(kspace)
            
            target = target.astype(np.float32)
            kspace = kspace.astype(np.complex64)
            
            save_file = os.path.join(args.output_dir, str(mri_f.name))
            data = h5py.File(save_file, 'w')
            data.create_dataset('kspace', data=kspace)
            data.create_dataset('reconstruction_rss', data=target)
            data.attrs.__setitem__('max', target.max())
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