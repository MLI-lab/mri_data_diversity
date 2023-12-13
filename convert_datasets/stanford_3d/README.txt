This folder contains the Stansford 3D dataset converted in a format that is compatible with fastMRI code.
To recreate the dataset run 'download_datasets.py' and then convert.py.
Source: http://mridata.org/list?project=Stanford%203D%20FSE

The Stanford 3D dataset in its original form is intended for reconstruction of sagittal views of the knee.
However, since the data are 3D MRI scans we can retropectively synthesize other views: Coronal and axial.

To recreate the dataset of coronal views from the already converted dataset, run 'convert_cor.py'.

To recreate the dataset of axial views from the already converted dataset, run 'convert_ax.py'.