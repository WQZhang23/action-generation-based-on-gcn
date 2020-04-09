# Action-Generation based on ST-GCN

## Introduction
This repository is the implementation of the idea:
skeleton-based-action data GENERATION based on ST-GCN.

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- Other Python libraries can be installed by `pip install -r requirements.txt`
- Install torchlight
```shell
cd torchlight; python setup.py install; cd ..
```

## Data Preparation
The data generation experiment are implemented based on the NTU RGB+D.
The pre-processed data is available by [MMSkeleton] (https://github.com/open-mmlab/mmskeleton).

## Training
To train the generative model, 

a) GCN base model
```
python main.py generation_gcn -c config/st_gcn/ntu-xsub/generation_train_local.yaml
```

b) GCN attention model
```
python main.py generation_attention -c config/st_gcn/ntu-xsub/generation_train_local.yaml
```


Configurations and logging files, will be saved under the `./work_dir` by default or `<work folder>` if you appoint it.

The training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` could be modified in the command line or configuration files.

## Inference
Use the pretrained generative model to generate action data by running
a) GCN base model
```
python main.py generation_gcn -c config/st_gcn//ntu-xsub/test_local.yaml
```

b) GCN attention model
```
python main.py generation_attention -c config/st_gcn//ntu-xsub/test_local.yaml
```

## Visualization of the action data

By running the `./tools/NTU_visu.ipynb` the skeleton based action data could be visualized.

## Reference
This repo is based on the skeleton-based action recognition, [MMSkeleton](https://github.com/open-mmlab/mmskeleton).