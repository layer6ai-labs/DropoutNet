# DropoutNet
python + tensorflow implementation of DropoutNet paper from NIPS 2017

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
3. [Running the Demo](#demo)

<a name="intro"/>

## Introduction
This repository contains python implementation of DropoutNet paper.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* python 2.7
* tensorflow-gpu==1.3.0
* Intel Xeon E5-2630
* Titan X (Pascal) 12gb, driver ver. 375.66

<a name="dataset"/>

## Dataset

The code runs on the [ACM RecSys 2017 challenge dataset](http://2017.recsyschallenge.com/) used in the paper.

To run the demo, download the dataset from [here](https://s3.amazonaws.com/public.layer6.ai/DropoutNet/recsys2017.pub.tar.gz)
In this dataset, we have trained a simple WMF model, whose performance is outlined in the paper. The model gives competitive results on the warm dataset but near random results on the cold user or item sets. These are compressed in binary form for convenience.

## Running training code

1. Download dataset above, extract and keep the directory structures.

2. Set parameters at the beginning of `main.py`
    * set appropriate paths under `# SETTINGS`
    * specifically, `DATA_PATH` must point to the `recsys.pub/eval` folder in the downloaded dataset
    * `CHECKPOINT_PATH` should point to the directory to dump model files, or `None` if not dumping model files
    * `TB_LOG_PATH` should point to the path to save Tensorboard log files, or `None` if not saving logs
    * parameters can be changed under the `# DEFINITIONS` section
  
3. Runing the main script will start training.
4. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=/datadrive/tmp/citeu.tf.log`

## Notes
