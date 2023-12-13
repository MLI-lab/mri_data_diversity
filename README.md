# Robustness of Deep Learning for Accelerated MRI: Benefits of Diverse Training Data
This repository provides code for reproducing the results of the paper: [Robustness of Deep Learning for Accelerated MRI: Benefits of Diverse Training Data](https://arxiv.org/pdf/2207.---.pdf), by Kang Lin and Reinhard Heckel. `TODO: UPDATE URL`

## Datasets
The majority of experiments from the paper are based on the [fastMRI dataset](https://fastmri.med.nyu.edu/). A detailed list of all datasets used and their references can be found in Table 1 in the paper.

## Installation
First, install PyTorch version 1.12.1 for your operating system and CUDA setup from the
[PyTorch website](https://pytorch.org/get-started/previous-versions/).

Then, install the following depedencies in the given order using `pip`:

- fastmri==0.3.0
- ismrmrd==1.12.0
- pandas==2.0.3
- tqdm==4.66.1
- timm==0.9.7
- PyYAML==6.0.1
- numpy==1.21.1
- h5py==2.10.0

## Usage
First download the datasets listed in Table 1 in the paper. Then convert all the datasets to a usable format by following the instructions in the folder `convert_datasets`.

To reproduce the results from Section 3, 4, and 5, only the fastMRI knee and fastMRI brain datasets are needed.

To obtain the models, simply run `main.py` after specifying the experimental setup and model configurations in a separate configuration file. A detailed explanation on how to setup the configurations files is found in the folder `training_examples`. 

In the following we provide some examples on how to run the code for obtaining the models in the paper:

- To obtain the U-nets from Section 3 where $P$ is PD-weighted knee and $Q$ is PDFS knee run
  
  - `python main.py --config_file training_examples/unet-small_pd.yml --device cuda:0`

    to obtain the U-net trained on $P$

  - `python main.py --config_file training_examples/unet-small_pdfs.yml --device cuda:0`

    to obtain the U-net trained on $Q$

  - `python main.py --config_file training_examples/unet-small_pd-pdfs.yml --device cuda:0`

    to obtain the U-net trained on $P$ and $Q$

- To obtain the U-nets from Section 7, run
  - `python main.py --config_file training_examples/unet-large_fmknee.yml --device cuda:0`
 
    to obtain the U-net trained on fastMRI knee

  - `python main.py --config_file training_examples/unet-large_fmbrain.yml --device cuda:0`
 
    to obtain the U-net trained on fastMRI brain

  - `python main.py --config_file training_examples/unet-large_diverse.yml --device cuda:0`
 
    to obtain the U-net trained on the collection of datasets
    
#### Multi GPU training

It is also possible to train the models using multiple GPUs. Here we rely on the Distributed Data Parallel package from PyTorch.

To do so, specify the number of GPUs by passing it to the keyword `--world_size`. For example, in the case of 4 GPUs, run

`python main.py --config_file training_examples/unet-large_diverse.yml --world_size 4`  

For other models and their configurations see Appendix D in the paper.

## Citation
```
TODO
```
## License
```
TODO
```


