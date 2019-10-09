# Hierarchical Multi-Task Network For Race, Gender and Facial Attractiveness Recognition
## Introduction
This repository holds the PyTorch implementation of our paper [Hierarchical Multi-Task Network For Race, Gender and Facial Attractiveness Recognition](https://ieeexplore.ieee.org/abstract/document/8803614).

![HMTNet](./hmt_architecture.png)


### New!!!
1. By leveraging ``FiveCrops`` inference, we are able to achieve better performance at ``149th Epoch``! 
2. We also report ``5 cross validation`` results, since we find newly proposed models often use this metric instead of ``6/4 splitting strategy``.


## How to use
* Install 3rd party libraries   
    ````sudo pip3 install -r requirements.txt````
* Modify [cfg.py](./config/cfg.py) to fit your path


## Hyper-param Selection
| Loss | MAE | RMSE | PC | Acc_R | Acc_G| Epoch | WD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MSE | 0.2556 | 0.3372 | 0.8693 | 99.68% | 98.53% | 170 | 5e-2|
| L1 | 0.2500 | 0.3299 | 0.8753 | 99.26% | 98.16% | 150 | 5e-2|
| Smooth L1 | 0.2531 | 0.3313 | 0.8738 | 99.54% | 98.58% | 170 | 5e-2|
| Smooth Huber | 0.2501 | 0.3263 | 0.8783 | 99.26% | 98.16% | 170 | 5e-2|
| **FiveCrops (New)** | 0.2439 | 0.3226 | 0.8801 | 99.45% | 98.58% | 149 | 5e-2|


## Performance Comparison
### 6/4 Splitting
| Methods | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: |
| ResNeXt-50 | 0.2518 | 0.3325 | 0.8777 |
| ResNet-18 | 0.2818 | 0.3703 | 0.8513 |
| AlexNet | 0.2938 | 0.3819 | 0.8298 |
| CRNet | 0.2816 | 0.3669 | 0.8450 |
| **HMTNet (Ours)** | **0.2500** | **0.3299** | **0.8753** |
| **HMTNet (Ours)** | **0.2501** | **0.3263** | **0.8783** |
| **HMTNet (Ours)** | **0.2439** | **0.3226** | **0.8801** |

### 5 Fold Cross Validation
#### MSE Loss

| Round | Acc_r | Acc_g | MAE | RMSE | PC | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 99.54% | 98.53% | 0.2431 | 0.3150 | 0.8916 |
| 2 | 99.82% | 98.71% | 0.2425 | 0.3246 | 0.8813 |
| 3 | 99.17% | 98.71% | 0.2468 | 0.3245 | 0.8883 |
| 4 | 99.63% | 98.25% | 0.2331 | 0.3033 | 0.9002 |
| 5 | 99.26% | 98.99% | 0.2465 | 0.3242 | 0.8859 |
| Avg | 99.48% | 98.64%	| 0.2424 | 0.3183 |	0.8895 |

#### Smooth Huber Loss 

| Round | Acc_r | Acc_g | MAE | RMSE | PC | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 99.54% | 98.53% | 0.2357 | 0.3091 | 0.8915 |
| 2 | 99.72% | 98.62% | 0.2365 | 0.3150 | 0.8884 |
| 3 | 99.54% | 99.17% | 0.2442 | 0.3235 | 0.8863 |
| 4 | 99.63% | 98.16% | 0.2335 | 0.3053 | 0.9006 |
| 5 | 99.36% | 99.26% | 0.2403 | 0.3178 | 0.8892 |
| Avg | 99.56% | 98.75% | 0.2380 | 0.3141 | 0.8912 |


| Methods | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: |
| ResNeXt-50 | 0.2291 | 0.3017 | 0.8997 |
| ResNet-18 | 0.2419 | 0.3166 | 0.8900 |
| AlexNet | 0.2651 | 0.3481 | 0.8634 |
| **HMTNet** | 0.2380 | 0.3141 | 0.8912 |


## Samples
![Prediction](./fbp_pred.png)

![TikTok Video](./TikTok.gif)

## Deep Feature Visualization
![Feature Visualization](./feature_vis.png)


## Resources
- [x] [Pretrained HMTNet](https://drive.google.com/file/d/1S11I3LlIpIW0PZusmTz52kETlW4u_ODF/view?usp=sharing)


## Citation
If you find this repository helps your research, please cite our paper:
```
@inproceedings{xu2019hierarchical,
  title={Hierarchical Multi-Task Network For Race, Gender and Facial Attractiveness Recognition},
  author={Xu, Lu and Fan, Heng and Xiang, Jinhai},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={3861--3865},
  year={2019},
  organization={IEEE}
}
```
