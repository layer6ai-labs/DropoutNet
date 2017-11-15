# DropoutNet <a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logo_fill.png" width="100"></a>
Python + TensorFlow implementation of the [DropoutNet](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf) - a deep neural network model for cold start in recommender systems.

Authors: [Maksims Volkovs](www.cs.toronto.edu/~mvolkovs), [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu), Tomi Poutanen

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
3. [Training](#demo)
3. [Notes](#notes)

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
* 128 GB ram (need about 50)
* Titan X (Pascal) 12gb, driver ver. 384.81

<a name="dataset"/>

## Dataset

The code runs on the [ACM RecSys 2017 challenge dataset](http://2017.recsyschallenge.com/) used in the paper.

To run the demo, download the dataset from [here](https://s3.amazonaws.com/public.layer6.ai/DropoutNet/recsys2017.pub.tar.gz).
With this dataset we have also included a pre-trained Weighted Matrix Factorization model (WMF)\[Hu et al., 2008\], that is used as preference input to DropoutNet. WMF produces competitive performance on the warm start but can't be applied to cold start so this code demonstrates how to apply DropoutNet to provide cold start capability to WMF. The format of the data is as follows:
```
- recsys2017.pub			// root folder when tar is extracted
  -eval					// this is where the model should point to
    *item_features_0based.txt		// our extracted item features in libsvm text form
    *user_features_0based.txt		// our extracted user features in libsvm text form
    -trained				// we have included our trained latent model to start DropoutNet
      -warm				// only warm is included for preference latent vectors
        *U.csv.bin			// numpy binarized user preference latent vectors (U)
	*V.csv.bin			// numpy binarized item preference latent vectors (V)
    -warm				// this folder contains all datasets
      *test_cold_item.csv		// sparse matrix for cold-start-item testing
      *test_cold_item_item_ids.csv	// item ids used for cold-start-item testing
      *test_cold_user.csv    		// sparse matrix for cold-start-user testing
      *test_cold_user_item_ids.csv	// item ids used for cold-start-user testing
      *test_warm.csv			// sparse matrix for warm-start testing
      *test_warm_item_ids.csv		// item ids used for warm-start testing
      *train.csv			// sparse matrix for training
      
sparse matrix are stored in csv as:
  <USER_ID>,<ITEM_ID>,<INTERACTION_TYPE>,<TIMESTAMP>
where INTERACTION_TYPE is one of:
  0: impression
  1: click
  2: bookmark
  3: reply
  4: delete
  5: recruiter interest
```

<a name="demo"/>

## Running training code

1. Download dataset above, extract and keep the directory structure.

2. run `main.py`
    * for usage, run with `main.py --help`
    * default setting trains a two layer neural network with hyperparameters selected for the RecSys data
    * gpu is used for training by default and cpu for inference
3. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=<log_path>`

during training recall@50,100,...,500 accuracy is shown every 50K updates for warm start, user cold start and item cold start validation sets

<a name="notes"/>

## Notes

* Make sure `--data-dir` points to the `eval/` folder, not the root
* On the setup outlined above, 2 full user batches (50,000 batches with 100 updates each) takes approximately 14 minutes.
## Training Curve
<p align="center">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/warm.png" width="500">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/cold_user.png" width="500">
<img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/cold_item.png" width="500">
</p>

