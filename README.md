## Introduction
This project is developed based on the paper *"You Should Learn to Stop Denoising on Point Clouds in Advance"*. 

## Data
Our data is the same as *Score-Based Point Cloud Denoising* by Shitong Luo and Wei Hu. Kudos to them for their excellent implementation and resources. You can check out their [GitHub repository](https://github.com/luost26/score-denoise). We will also make the data available as a zip file for ease of use. Please download and place it in the `./data` directory.

## Environment Setup
To set up the project environment, follow these steps:

```bash
conda create -n ASDN python=3.10
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning=2.5.0.post0 -c conda-forge
conda install iopath=0.1.10 -c conda-forge
conda install pytorch3d=0.7.8 -c pytorch3d
pip install pyg-lib==0.4.0+pt23cu118
pip install torch-scatter==2.1.2+pt23cu118
pip install torch-sparse==0.6.18+pt23cu118
pip install torch-cluster==1.6.3+pt23cu118
pip install torch-spline-conv==1.2.2+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
pip install torch-geometric==2.6.1
pip install point-cloud-utils==0.31.0
pip install pandas
pip install tensorboard


cd Chamfer3D
python setup.py install
cd ..

cd pointops
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
python test_ASDN.py --niters 2  --noise_lvls 0.01 
python test_ASDN.py --niters 2  --noise_lvls 0.02
python test_ASDN.py --niters 3  --noise_lvls 0.03
```
