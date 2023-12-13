This folder contains code to convert the Calgary Campinas challenge dataset in a format that is compatible with fastMRI code.
Source: https://www.ccdataset.com/mr-reconstruction-challenge

To recreate the datasset download the datset and then run convert.py.

The dataset in its original form is intended for reconstruction of axial views of the brain.
However, since the data are 3D MRI scans we can retropectively synthesize other views: Coronal and sagittal.

To recreate the dataset of coronal views from the downloaded dataset, run 'convert_cor.py'.

To recreate the dataset of sagittal views from the downloaded converted dataset, run 'convert_sag.py'.
