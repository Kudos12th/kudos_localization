
### [AtLoc: Attention Guided Camera Localization](https://arxiv.org/abs/1909.03557) - AAAI 2020 (Oral).
<br/>

![image](https://github.com/Kudos12th/kudos_localization/assets/63325450/60675056-3ead-4371-ba81-cba0d410b319)


## Introduction 

This is the PyTorch implementation of **AtLoc**, a simple and efficient neural architecture for robust visual localization.

## Setup

AtLoc uses a Conda environment that makes it easy to install all dependencies.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 2.7.

2. Create the `AtLoc` Conda environment: `conda env create -f environment.yml`.

3. Activate the environment: `conda activate py27pt04`.

4. Note that our code has been tested with PyTorch v0.4.1 (the environment.yml file should take care of installing the appropriate version).

## Data

### Robocup Datset

Pixel and Pose statistics must be calculated before any training. Use the `data/dataset_mean.py`, which also saves the information at the proper location.

```
cd data
python3 dataset_mean.py --data_dir ./
```

## Running the code

### Training
The executable script is `train.py`. For example:

- AtLoc on `received_images`: 
```
python3 train.py --dataset Robocup --scene received_images --model AtLoc --gpus 0
```

- AtLocLstm on `received_images`: 
```
python3 train.py --scene received_images --model AtLoc --lstm True --gpus 0
```

The meanings of various command-line parameters are documented in train.py. The values of various hyperparameters are defined in `tools/options.py`.

### Inference
The trained models for partial experiments presented in the paper can be downloaded [here](https://drive.google.com/drive/folders/1inY29zupeCmvIF5SsJhQDEzo_jzY0j6Q). The inference script is `eval.py`. Here are some examples, assuming the models are downloaded in `logs`.

- AtLoc on `loop`: 
```
python3 eval.py --scene received_images --model AtLoc --gpus 0 --weights {WEIGHTS_PATH}.pth.tar
```

Calculates the network attention visualizations and saves them in a video

- For the AtLoc model trained on `received_images`:
```
python3 saliency_map.py --scene received_images --model AtLoc --gpus 0 --weights {WEIGHTS_PATH}.pth.tar 
```

## Citation

```
@article{wang2019atloc,
  title={AtLoc: Attention Guided Camera Localization},
  author={Wang, Bing and Chen, Changhao and Lu, Chris Xiaoxuan and Zhao, Peijun and Trigoni, Niki and Markham, Andrew},
  journal={arXiv preprint arXiv:1909.03557},
  year={2019}
}
```