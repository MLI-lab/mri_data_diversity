import h5py
import glob
import os
import pathlib
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser
import numpy as np

def ifft2(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))

def fft2(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
    
    mri_files = list(pathlib.Path(args.input_dir).glob('*.h5'))
    for i, mri_f in tqdm(zip(range(len(mri_files)), sorted(mri_files)), total=len(mri_files)):
        with h5py.File(mri_f, 'r') as hf:
            kspace_hf = hf['kspace'][()]
            kspace_cpx = kspace_hf[...,::2] + 1j*kspace_hf[...,1::2]
            kspace_cpx = kspace_cpx.transpose(0,-1,1,2)
            kspace_cpx *= 1e-10

            img = ifft2(kspace_cpx)
            img_shifted = np.fft.ifftshift(img, axes=(-2,-1))

            # Coronal view
            img_cor = img_shifted.transpose(2,1,0,3)
            kspace = fft2(img_cor).astype(np.complex64)
            target = np.sqrt(np.sum(np.abs(img_cor)**2, axis=1)).astype(np.float32)

            save_file = os.path.join(args.output_dir, str(mri_f.name))
            data = h5py.File(save_file, 'w')
            data.create_dataset('kspace', data=kspace)
            data.create_dataset('reconstruction_rss', data=target)
            data.attrs.create('max', data=target.max())
            data.close()
        

def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the original dataset is located.",
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