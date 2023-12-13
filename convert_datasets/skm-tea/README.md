Download the SKM-TEA dataset from the source:
https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7

convert.py separates the two echoes for each file of the skm-tea dataset and saves each echo as separate .h5 files. 
Additionally, the target is computes as the rss of the coil images and padding information is added to the .h5 files.