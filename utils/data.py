import fastmri
from fastmri.data import transforms
from fastmri.data.subsample import MaskFunc
import pandas as pd
import h5py
import numpy as np
import torch
from pathlib import Path
import xml.etree.ElementTree as etree

from scipy.ndimage import fourier_shift

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

def fft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def rss_np(x, axis=0):
    return np.sqrt(np.sum(np.square(np.abs(x)), axis=axis))

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

class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    
class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        files,
        challenge: str = 'multicoil',
        transform: Optional[Callable] = None,
        slice_indices = None,
        slice_range = None,
        augment_data = False,
        batch_size = 1,
    ):
        """
        Args:
            files: list of filepaths
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            slice_range: list of tuples [(start_slice_idx, end_slice_idx), ...],
                i.e., one tuple for each file
                start_slice_idx is included in slice slection
                end_slice_idx is excluded in slice selection
            batch_size: batch_size-1 additional number of random_samples within a volume corresponding to a selected slice
                if batch_size exceeds slice_range then slice_range number of samples are returned
        """
        assert slice_indices is None or slice_range is None, "Use either slice_indices or slice_range or neither"
        
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.batch_size = batch_size
        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.augment_data = augment_data
        self.raw_samples = []
        files = [Path(f) for f in files]
        if slice_indices is None and slice_range is None:
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                new_raw_samples = []
                for slice_ind in range(num_slices):
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples
        elif slice_range is not None:
            for fname, (start, end) in sorted(zip(files, slice_range)):
                metadata, _ = self._retrieve_metadata(fname)
                metadata['start_slice'] = start
                metadata['end_slice'] = end
                new_raw_samples = []
                for slice_ind in range(start, end):
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples
        else:
            assert len(files) == len(slice_indices), "Each file has to have a corresponding slice number"
            for fname, slice_ind in sorted(zip(files, slice_indices)):
                metadata, _ = self._retrieve_metadata(fname)
                if slice_ind == -1:
                    new_raw_samples = []
                    for idx in range(num_slices):
                        raw_sample = FastMRIRawDataSample(fname, idx, metadata)
                        new_raw_samples.append(raw_sample)
                    self.raw_samples += new_raw_samples
                else:
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    self.raw_samples.append(raw_sample)


    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices, h, w = hf["kspace"].shape[0], hf["kspace"].shape[-2], hf["kspace"].shape[-1]    
            if "ismrmrd_header" in list(hf.keys()):
                et_root = etree.fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )

                lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, lims + ["center"]))
                enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            else:
                attrs = dict(hf.attrs)
                if 'padding_left' in attrs:
                    padding_left = attrs['padding_left']
                else:
                    padding_left = 0
                    
                if 'padding_right' in attrs:
                    padding_right = attrs['padding_right']
                else:
                    padding_right = hf["kspace"].shape[-1]
                enc_size = (h, w, 1) 
                recon_size = enc_size
                            
            padding_left = 0 if padding_left < 0 else padding_left
            padding_right = w if padding_right > w else padding_right

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)//self.batch_size

    def __getitem__(self, i: int):
        if self.batch_size > 1:
            i = np.random.randint(len(self.raw_samples))
            
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][[dataslice]]
            
            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][[dataslice]] if self.recons_key in hf else None
            
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            
            h = target.shape[-2] if attrs["recon_size"][0] > target.shape[-2] else attrs["recon_size"][0]
            w = target.shape[-1] if attrs["recon_size"][1] > target.shape[-1] else attrs["recon_size"][1]
                
            attrs["recon_size"] = (h, w, 1) 
            
            if 'start_slice' in metadata:
                start_slice = metadata['start_slice']
            else:
                start_slice = 0
                
            if 'end_slice' in metadata:
                end_slice = metadata['end_slice']
            else:
                end_slice = hf["kspace"].shape[0]
                
            if end_slice - start_slice < self.batch_size:
                batch_size = end_slice - start_slice
            else:
                batch_size = self.batch_size
                
            slice_indices = np.arange(hf["kspace"].shape[0])
            slice_indices = np.concatenate((slice_indices[start_slice:dataslice], slice_indices[dataslice+1:end_slice]))
            slice_indices = np.random.choice(slice_indices, batch_size-1, replace=False)
            
            for i in slice_indices:
                kspace = np.concatenate((kspace, hf["kspace"][[i]]))
                target = np.concatenate((target, hf[self.recons_key][[i]]))
            

            if self.augment_data:
                if np.random.rand() > 0.5: # vertical flip               
                    target = np.flip(target, axis=-2)
                    kspace = np.flip(kspace, axis=-2)
                    kspace = fourier_shift(kspace, (0,0,-1,0))
                    
                if np.random.rand() > 0.5: # horizontal flip
                    target = np.flip(target, axis=-1)
                    kspace = np.flip(kspace, axis=-1)
                    kspace = fourier_shift(kspace, (0,0,0,-1))
                    pad_left = attrs["padding_left"]
                    attrs["padding_left"] = target.shape[-1] - attrs["padding_right"]
                    attrs["padding_right"] = target.shape[-1] - pad_left

                if np.random.rand() > 0.5: # 90 degree rotation
                    target = np.rot90(target, axes=(-2, -1))
                    kspace = np.rot90(kspace, axes=(-2, -1))
                    kspace = fourier_shift(kspace, (0,0,-1,0))
                    attrs["padding_left"] = 0
                    attrs["padding_right"] = target.shape[-1]
                    h, w, z = attrs["encoding_size"]
                    attrs["encoding_size"] = (w, h, z)
                    h, w, z = attrs["recon_size"]
                    attrs["recon_size"] = (w, h, z)
                    
                target = np.ascontiguousarray(target)

            if self.transform is None:
                sample = (kspace, mask, target, attrs, fname.name, np.concatenate(([dataslice], slice_indices)))
            else:
                sample = self.transform(kspace, mask, target, attrs, fname.name, np.concatenate(([dataslice], slice_indices)))

        return sample
    
class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.
    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: list
    max_value: float
    
class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the filename, and the slice number.

        """
        kspace_torch = transforms.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = transforms.apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[-1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image, -3)

        # normalize target
        if target is not None:
            target_torch = transforms.to_tensor(target)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )
    
def slice_df(df_dataset, attributes):
    ''' 
    attributes: dictionary to extract elements based on keys and values
    if attributes is list of dicts then logical-and-reduce (extract from df) 
    for each dict but then logical-or-reduce (combine) all extracted subsets to create final set
    e.g, attributes = {'protocolName': 'CORPD_FBK'} for extracting all CORPD_FBK data,
    attributes = ({'protocolName': 'CORPD_FBK'}, {'protocolName': 'CORPDFS_FBK'}) 
    for extracting all CORPD_FBK data and CORPDFS_FBK data
    '''
    assert isinstance(attributes, tuple) or isinstance(attributes, dict), "attributes must be given as dict or tuple of dicts"
    attributes = attributes if isinstance(attributes, tuple) else (attributes,)
    mask = []
    for attr in attributes:
        and_mask = np.logical_and.reduce([df_dataset[k] == v for k, v in attr.items()])
        assert and_mask.any(), 'Slicing dataframe with given attributes yields empty set. Check for validity of attributes (e.g, existence, typos, combinations).'
        mask.append(and_mask)
    df_dataset = df_dataset[np.logical_or.reduce(mask)]
    return df_dataset

def load_dataset(path_to_data, attr=None):
    ''' 
    attr: extract elements based on condition, e.g.,
    attr = {'protocolName': 'CORPD_FBK'} for extracting all CORPD_FBK data
    '''
    df_dataset = pd.read_csv(path_to_data)
    if attr is not None:
        df_dataset = slice_df(df_dataset, attr)
    return df_dataset



def train_test_split(dataset, split_ratio):
     # split_ratio is training_data/total_data
    num_train = round(split_ratio*len(dataset))
    trainset = dataset[:num_train]
    testset = dataset[num_train:]
    return trainset, testset