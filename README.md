# A Abstractive Summarization Implementation by Transformer and Pointer-generator
when I wanted to get summary by neural network, I tried many ways to generator abstract summary, but the result was not good.
when I heared 2018 byte cup, I found some information about it, and the champion's solution attracted me, but I found some websites,
like github gitlab, I didn't find the official code, so I decided to implement it.

## Requirements
* python==3.x (Let's move on to python 3 if you still use python 2)
* tensorflow==1.12.0
* numpy>=1.15.4
* sentencepiece==0.1.8
* tqdm>=4.28.1

## model structure
<img src="fig/structure.jpg">

## Training
* STEP 1. [download](https://pan.baidu.com/s/1szq0Wa60AS5ISpM_SNPcbA) the dataset, pwd is ayn6, the dataset is lcsts by pre processed, so you will see very different dataset structure with lcsts in the file
each line is abstract and article, they split by ",", if you worry the amount of the dataset is different between my and lcsts, don't 
worry, the amout of the dataset is same as lcsts. 
* STEP 2. Check hparams.py to see which parameters are possible. 

| name | type | detail |
|--------------------|------|-------------|
vocab_size | int | vocab size
train1 | str | train dataset dir
eval1 | str| eval dataset dir
eval3 | str| calculate rouge score dataset dir
vocab | str| vocabulary file path
batch_size | int| train batch size
eval_batch_size | int| eval batch size
lr | float| learning rate
warmup_steps | int| warmup steps by learing rate
logdir | str| log directory
num_epochs | int| the number of train epoch
evaldir | str| evaluation dir
d_model | int| hidden dimension of encoder/decoder
d_ff | int| hidden dimension of feedforward layer
num_blocks | int| number of encoder/decoder blocks
num_heads | int| number of attention heads
maxlen1 | int| maximum length of a source sequence
maxlen2 | int| maximum length of a target sequence
dropout_rate | float| dropout rate
beam_size | int| beam size for decode

