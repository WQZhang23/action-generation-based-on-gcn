# Action-Generation based on ST-GCN

## Introduction
This repository is the implementation of the idea:
skeleton-based-action data generation based on ST-GCN.

Please refer to [MMSkeleton](https://github.com/open-mmlab/mmskeleton) for the original repo of ST-GCN.

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
To train the generative model, run

```
python main.py generation -c config/st_gcn/ntu-xsub/generation_train_local.yaml
```

Configurations and logging files, will be saved under the `./work_dir` by default or `<work folder>` if you appoint it.

The training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` could be modified in the command line or configuration files.

## Inference
Use the pretrained generative model to generate action data by running

```
python main.py generative -c config/st_gcn//ntu-xsub/test_local.yaml
```

## Visualization of the action data

By running the `./tools/NTU_visu.ipynb` the skeleton based action data could be visualized.