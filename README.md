# ogbn-papers100m-sage

This repository contains the code to reproduce the performance of ogbn-papers100m-sage on ogbn-papers100M dataset. Hyperparameters are empirically tunned the validation set. All experiments were runned with a Tesla V100 with 16GB memory.

We made some modifications based on the original GraphSAGE:

1. skip connection: the each sage conv layer, the inputs will be added to the outputs with a linear projection.
2. inception-like structure: the hidden representations of different perceptive field are concatenated for classification.

## Requirements
Dependencies with python 3.7.3:
```
torch==1.5.1
ogb==1.2.5
dgl-cu92==0.6.0
```

## Usage
```shell
pip install -r requirements.txt
python train.py
```
Training log will be written into directory "./log/[version]-[timestamp]-seed[seed]"

## Result

```
runned 10 times
valid accus: [0.7032690695725062, 0.7041711571468486, 0.7021753881770646, 0.7013850636650302, 0.7045064463337724, 0.7027581527162415, 0.7025905081227797, 0.7025026942881092, 0.7046421586237177, 0.7044186324991019]
test  accus: [0.6716261232259328, 0.6691860519366608, 0.6670539055137213, 0.6693773385960492, 0.6720086965447097, 0.6727411844843192, 0.6715141505472665, 0.6695546286706043, 0.6725172391269864, 0.6700165159701033]
average valid accu: 0.7032 ± 0.0011
average test  accu: 0.6706 ± 0.0017
numbers of params: 5755172
```


