## Introduction
This project is developed based on the paper *"You Should Learn to Stop Denoising on Point Clouds in Advance"*. 

## Data
Our data is the same as *Score-Based Point Cloud Denoising* by Shitong Luo and Wei Hu. Kudos to them for their excellent implementation and resources. You can check out their [GitHub repository](https://github.com/). We will also make the data available as a zip file for ease of use. Please download and place it in the `./data` directory.

## Environment Setup
To set up the project environment, follow these steps:

```bash
conda create -n ASDN python=3.10
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
pip install torch_geometric
pip install point-cloud-utils
pip install pandas
pip install tensorboard

cd pointops
python setup.py install
cd Chamfer3D
python setup.py install
```

## Training
To train the proposed ASDN network, run the following command:

```bash
python train_ASDN.py
```

If you want to train the classifier, run:

```bash
python train_classifier.py
```

## Inference
We provide pre-trained models for inference. You can use them by running the following commands:

```bash
python test_ASDN.py --niters=2 --resolutions="['10000_poisson', '50000_poisson']" --noise_lvls="['0.01', '0.02']"
python test_ASDN.py --niters=3 --resolution="['10000_poisson', '50000_poisson']" --noise="['0.03']"
```
