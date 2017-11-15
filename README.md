# DropoutNet
Python + TensorFlow implementation of the DropoutNet model

[Maksims Volkovs](www.cs.toronto.edu/~mvolkovs), Guangwei Yu, Tomi Poutanen

[layer6.ai](http://layer6.ai)

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
3. [Training](#demo)
3. [Notes](#notes)

<a name="intro"/>

## Introduction
This repository contains full implementation of the DropoutNet model and icludes both training and evaluation routines. We also provide the [ACM RecSys 2017 Challenge](http://2017.recsyschallenge.com) dataset that we further split into three subsets for warm start, user cold start and item cold start evaluation. The aim is to train a *single* model that can be applied to all three tasks and we report validation accuracy on each task during training. If you use this model in your research please cite this paper:
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
@inproceedings{Volkovs2017,
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

To run the demo, download the dataset from [here](https://s3.amazonaws.com/public.layer6.ai/DropoutNet/recsys2017.pub.tar.gz)
In this dataset, we have trained a simple WMF model, whose performance is outlined in the paper. The model gives competitive results on the warm dataset but near random results on the cold user or item sets. These are compressed in binary form for convenience.

<a name="demo"/>

## Running training code

1. Download dataset above, extract and keep the directory structures.

2. run `main.py`
    * for usage, run with `main.py --help`
    * this script demos DropoutNet on RecSys dataset, producing results outlined in the paper
    * default setting runs a simple two layer network with appropriate setting
    * default setting uses gpu for training and cpu for inference
3. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=<log_path>`

the output is evaluation of recall at 50,100,...,500 for:
- warm
- cold user
- cold item

<a name="notes"/>

## Notes

* Make sure `--data-dir` points to the `eval/` folder, not the root
* On the machine outlined above, 2 full user batch (50,000 batches with 100 updates each) takes about 14 minutes, reaching approximately 0.45 recall@500 for warm, and 0.30 for both cold user and cold item.
