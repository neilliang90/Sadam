# Sadam
This repository contains our pytorch implementation of Sadam in the paper [Calibrating the Learning Rate for Adaptive Gradient Methods to Improve Generalization Performance](https://arxiv.org/abs/1908.00700).



## Command Line Arguments:
--hist True : record information of A-LR

## Prerequisites:
* pytorch
* tensorboard


## Usage examples
### sgd
CUDA_VISIBLE_DEVICES=0 python main_CIFAR.py --b 128 --NNtype ResNet20 --optimizer sgd --reduceLRtype manual0 --weight_decay 5e-4  --lr 0.1

### Adam
CUDA_VISIBLE_DEVICES=1 python main_CIFAR.py --b 128 --NNtype ResNet20 --optimizer Sadam --reduceLRtype manual0 --weight_decay 5e-4  --transformer Padam --partial 0.25 --grad_transf square --lr 0.001

### Padam
CUDA_VISIBLE_DEVICES=1 python main_CIFAR.py --b 128 --NNtype ResNet20 --optimizer Sadam --reduceLRtype manual0 --weight_decay 5e-4  --transformer Padam --partial 0.125 --grad_transf square --lr 0.1


### adabound
CUDA_VISIBLE_DEVICES=0 python main_CIFAR.py --b 128 --NNtype ResNet20 --optimizer adabound --reduceLRtype manual0 --weight_decay 5e-4  --lr 0.01

### Sadam ( our methods ) 
CUDA_VISIBLE_DEVICES=1 python main_CIFAR.py --b 128 --NNtype ResNet20 --optimizer Sadam --reduceLRtype manual0 --weight_decay 5e-4  --transformer softplus --smooth 50 --lr 0.01 --partial 0.5 --grad_transf square 


## Results:
### Anisotrpoic A-LR, which cause "small learning rate dillema" in Adam 
![Alt text](figure1_adam_over4model.png?raw=true "Title")
### Performance of softplus function to calibrate A-LR
![Alt text](Behavior_softplus_function.png?raw=true "Title")
### Comparison of different methods
![Alt text](cifar10.png?raw=true "Title")

