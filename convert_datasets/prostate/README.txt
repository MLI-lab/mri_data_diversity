This is not the fastMRI prostate dataset in its original form as described in the paper
https://fastmri.med.nyu.edu/, https://arxiv.org/pdf/2304.09254.pdf, https://github.com/cai2r/fastMRI_prostate/tree/main

For the dataset with T2-weighted images, in its original form, the raw k-space has an additional dimension corresposing to
three so-called 'averages'. Each average is 2x undersampled, where two of these 'averages' correspond
to the odd k-space lines and one 'average' corresponds to even k-space lines.  The reconstructions 
are based on GRAPPA estimatation of missing k-space values and the images (targets) are flipped. 
Furthermore, the provided k-space is not zero-padded. In the fastMRI brain and knee dataset, the raw k-space
is fully-sampled and zero-padded. The image reconstruction is done by directly applying 2D ifft to this k-space.

Here, the format of the T2 prostate fastMRI dataset as been modified such that it is more consistent with
the original fastMRI brain and knee dataset.

1. the 3 so-called 'averages' are combined into one full kspace by averaging the two 'averages' corresponding
to the odd kspace lines and filling the even kspace lines with the 'average' corresposding to the
even kspace lines.

2. kspace is then padded according to ismrmrd header file

3. a new target is then created by applying 2D ifft to the modifed kspace and center cropping the 
modified kspace and target are then saved as a new .h5 file

The conversion can be done by running 'convert.py' on the original prostate dataset.