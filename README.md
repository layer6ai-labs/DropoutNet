
<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logo.svg" width="180"></a>
</p>

# DropoutNet
Python + TensorFlow implementation for the NIPS'17 paper 
["DropoutNet: Addressing Cold Start in Recommender Systems"](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf).

Authors: [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs), [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu), Tomi Poutanen

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
3. [Training](#demo)

<a name="intro"/>

## Introduction
This repository contains full implementation of the DropoutNet model and includes both training and evaluation routines. We also provide the [ACM RecSys 2017 Challenge](http://2017.recsyschallenge.com) dataset that we further split into three subsets for warm start, user cold start and item cold start evaluation. The aim is to train a *single* model that can be applied to all three tasks and we report validation accuracy on each task during training. If you use this model in your research please cite this paper:
```
@inproceedings{Volkovs2017,
	author = {Maksims Volkovs and Guangwei Yu and Tomi Poutanen},
	title = {DropoutNet: {Addressing} Cold Start in Recommender Systems},
	booktitle = {Neural Information Processing Systems},
	year = {2017}
}
```
and also this paper if you use the RecSys dataset:
```
@inproceedings{Abel2017,
	author = {Fabian Abel and Yashar Deldjoo and Mehdi Elahi and Daniel Kohlsdorf},
	title = {RecSys Challenge 2017: {Offline} and Online Evaluation},
	booktitle = {ACM Conference on Recommender Systems},
	year = {2017}
}
```


<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* python 2.7
* tensorflow-gpu 1.3.0
* Intel Xeon E5-2630
* 128GB ram (around 30GB is required)
* Titan X (Pascal) 12GB, driver ver. 384.81
* CUDA 9 and CUDNN 7

<a name="dataset"/>

## Dataset

To run the model, download the dataset from [here](https://s3.amazonaws.com/public.layer6.ai/DropoutNet/recsys2017.pub.tar.gz).
With this dataset we have also included pre-trained Weighted 
Factorization model (WMF)\[Hu et al., 2008\], that is used as preference input to the DropoutNet. WMF produces competitive performance on warm start but doesn't generalize to cold start. So this code demonstrates how to apply DropoutNet to provide cold start capability to WMF. The format of the data is as follows:
```
recsys2017.pub				
└─ eval					// use path to this folder in --data-dir
   ├─ trained				// WMF model
   │  └─ warm				
   │     ├─ U.csv.bin			// numpy binarized WMF user preference latent vectors (U)
   │     └─ V.csv.bin			// numpy binarized WMF item preference latent vectors (V)
   ├─ warm				
   │  ├─ test_cold_item.csv		// validation interactions for item cold start 
   │  ├─ test_cold_item_item_ids.csv	// targets item ids for item cold start
   │  ├─ test_cold_user.csv    		// validation interactions for user cold start
   │  ├─ test_cold_user_item_ids.csv	// target user ids for user cold start
   │  ├─ test_warm.csv			// validation interactions for warm start
   │  ├─ test_warm_item_ids.csv		// target item ids for warm start
   │  └─ train.csv			// training interactions
   ├─ item_features_0based.txt		// item features in libsvm format
   └─ user_features_0based.txt		// user features in libsvm format
      
interactions are stored in csv as:
  <USER_ID>,<ITEM_ID>,<INTERACTION_TYPE>,<TIMESTAMP>
where INTERACTION_TYPE is one of:
  0: impression
  1: click
  2: bookmark
  3: reply
  5: recruiter interest
```

<a name="demo"/>

## Running training code

1. Download the dataset, extract and keep the directory structure.

2. run `main.py`
    * for usage, run with `main.py --help`
    * default setting trains a two layer neural network with hyperparameters selected for the RecSys data
    * gpu is used for training by default and cpu for inference
3. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=<log_path>`

During training recall@50,100,...,500 accuracy is shown every 50K updates for warm start, user cold start and item cold start validation sets.

Notes:

* Make sure `--data-dir` points to the `eval/` folder, not the root
* On our environment (described above) 50K updates takes approximately 14 minutes with the default GPU/CPU setting.
* By default, training happens on GPU while inference and batch generation is on CPU.

## Validation Curves
<p align="center">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/warm.png" width="500">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/cold_user.png" width="500">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/cold_item.png" width="500">
</p>

