Each folder contains coversion code for the dataset corresponding to the folder name. The conversion code will transform the data
in a format usable with fastMRI code.

After the conversion, the data in stanford_2d, stanford_3d, nyu_2d, cc-359 and M4Raw (except the M4Raw GRE dataset), skm-tea
should go through another conversion given by convert_flip.py. This will flip the images and transform the corresponding kspace
such that the orientation is consistent with fastRMI data (where all images are upside down).
