# Turfgrass Divot Analysis with Sequoia+ Camera
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repository contains the code implementation for our IMVIP Paper 2024. If you find this code useful in your research, please consider citing us.
Self-supervised detection and classification of turfgrass abnormalities using vision foundation models and NDVI data using Parrot Sequoia+ camera

<!-- TOC -->

  - [Divot in PyTorch](#divot-in-pytorch)
  - [Requirements](#requirements)
  - [Code Structure](#code-structure)
  - [Config File Format](#config-file-format)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

<!-- /TOC -->

## Requirements
Conda is used for managing the environment. Install the dependencies as follows:

```bash
pip install -r requirements.txt
```
or for a local installation:

```bash
pip install --user -r requirements.txt
```

### Manual Installation
To set up the environment manually:

```bash
conda create --name your_env_name
conda install -c conda-forge pytorch-gpu
conda install torchvision
conda install exiftool  # For image tags, adjust based on your system, 
```

## Code Structure
An overview of the code structure is as follows:

```plaintext
code-base/
├── README.md
├── LICENSE
├── .gitignore
├── setup.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── rectify_images.py     # 1. Process the images of image in folders
│   ├── align_images.py       # 2. Image alignment of image in folders & NDVI
│   ├── create_crop_pairs.py  # 3. Generate Sliding windows pairs for RGB and NDVI_Aligned
│   ├── create_embeddings.py  # 4. Generate Deep learning embeddings for RGB crops 
│   ├── cluster_and_report.py # 5. Cluster data and generate reports aginst NDVI groundtruth 
│   ├── config.py             # configuration global vars
├── tests/ # Exploration and testing on the data
│   ├── __init__.py
│   ├── test_rectify_images.py
│   ├── test_align_images.py
│   ├── test_calculate_ndvi.py
    ├── .......              # Other scripts
├── notebooks/
│   ├── exploratory_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│
└── saved/                       # Save plots etc here
```

## To-Do List
- [x] Baseline code to process_sequia
- [x] Baseline PyTorch import deep learning mode 
- [x] Conda environment setup with repository
- [x] Test PyTorch Vision Dino V2
- [x] TSNE with PyTorch Vision Dino V2
- [x] Finalize Conda environment
- [x] Refactor code for paper release
- []  Tests 
- []  Paper link for IMVIP 2024
- [] MORE research on  

## Acknowledgement
- [Tutorial for Parrot Sequoia and Sequoia+ corrections](https://github.com/stevefoy/micasense_imageprocessing_sequoia/tree/master)
- [PyTorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.md)
- [TSNE](#)

## Citation
```bibtex
@inproceedings{IMVIP2024,
    author = {Stephen Foy and Simon McLoughlin},
    title = {A Self-Supervised Approach for the Detection and Classification of Turfgrass Abnormalities},
    booktitle = {Irish Machine Vision and Image Processing Conference (IMVIP)},
    year = {2024}
}
