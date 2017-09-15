# Multi-level Scene Description Network


This is our PyTorch Implementation of [Scene Graph Generation from Objects, Phrases and Region Captions](http://cvboy.com/pdf/iccv2017_msdn.pdf). The project is based on PyTorch implementation of [faster R-CNN](https://github.com/longcw/faster_rcnn_pytorch). 


## Progress
- [x] README for training 
- [x] README for project settings
- [ ] our trained RPN
- [ ] our trained Full Model
- [ ] Our cleansed Visual Genome Dataset
- [x] training codes
- [x] evaluation codes
- [ ] Model acceleration
- [ ] Multi-GPU support

We are still working on the project. If you are interested, please **Follow** our project. 


## Project Settings

1. Install the requirements (you can use pip or [Anaconda](https://www.continuum.io/downloads)):

    ```
    conda install pip pyyaml sympy h5py cython numpy scipy
    conda install -c menpo opencv3
    pip install easydict
    ```

2. Clone the Faster R-CNN repository
    ```bash
    git clone git@github.com:yikang-li/MSDN.git
    ```

3. Build the Cython modules for nms and the roi_pooling layer
    ```bash
    cd MSDN/faster_rcnn
    ./make.sh
    ```
4. Download the [trained model](https://drive.google.com/open?id=0B4pXCfnYmG1WOXdpYVFybWxiZFE) 

5. Download [our cleansed Visual Genome dataset](https://www.dropbox.com/sh/anewjszk97caes1/AAB3IBziBnQTclv-iHkfZezca?dl=0). p.s. Our ipython [scripts](https://github.com/yikang-li/vg_cleansing) for data cleansing is also released. 
6. 


## Data perparation

We have offer our cleansed Visual Genome dataset. Please download the images by yourself. Our [ipython scripts for data cleansing](https://github.com/yikang-li/vg_cleansing) is also released for your reference. 

Please [Click](https://www.dropbox.com/sh/anewjszk97caes1/AAB3IBziBnQTclv-iHkfZezca?dl=0) to download from Dropbox. 

## Training
- Training from scratch
	1. Training RPN for object proposals and caption region proposals (the **shared conv layers** are fixed)

	by default, the training is done on a small part of the full dataset:
	```
	CUDA_VISIBLE_DEVICES=0 python train_rpn_region.py
	```

	For full Dataset Training:
	```
	CUDA_VISIBLE_DEVICES=0 python train_rpn_region.py --max_epoch=10 --step_size=2 --dataset_option=normal --model_name=RPN_full_region
	```

	```--step_size``` is set to indicate the number of epochs to decay the learning rate, ```dataset_option``` is to indicate the *small*, *fat* or *normal* subset. 
	2. Training MSDN

	Here, we use SGD (controled by ```--optimizer```)by default:
	```
	CUDA_VISIBLE_DEVICES=4 python train_hdn.py --load_RPN --saved_model_path=./output/RPN/RPN_region_full_best.h5 \
	--dataset_option=normal --enable_clip_gradient \
	--step_size=3 --caption_use_bias --caption_use_dropout 
	```
- Furthermore, we can directly use end-to-end training from scratch. The result is not good. 
	```
	CUDA_VISIBLE_DEVICES=7 python train_hdn.py \
	--dataset_option=normal --enable_clip_gradient \
	--step_size=3 --MPS_iter=1 --caption_use_bias --caption_use_dropout \
	--max_epoch=11 --optimizer=1 --lr=0.001
	```


## Evaluation 

Since our project only support one-GPU training, the training process is really slow (take 1 week per experiment). Therefore, we provide [our pretrained full Model](https://www.dropbox.com/sh/0dvf5igsbn5t2k5/AACBzeivC8r6tiOQUCVD6MPHa?dl=0). 


```
CUDA_VISIBLE_DEVICES=5 python train_hdn.py \ 
	--resume_training --resume_model ./pretrained_models/HDN_1_iters_alt_normal_H_LSTM_with_bias_with_dropout_0_5_nembed_256_nhidden_512_with_region_regression_resume_SGD_best.h5 \
	--dataset_option=normal  --MPS_iter=1 \
	--caption_use_bias --caption_use_dropout \
	--rnn_type LSTM_normal
```


## Acknowledgement

We thank [longcw](https://github.com/longcw/faster_rcnn_pytorch) for his generously releasing the [PyTorch Implementation of Faster R-CNN](https://github.com/longcw/faster_rcnn_pytorch). 


## Reference

@inproceedings{li2017msdn,  
	author={Li, Yikang and Ouyang, Wanli and Zhou, Bolei and Wang, Kun and Wang, Xiaogang},  
	title={Scene graph generation from objects, phrases and region captions},  
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision},  
	year      = {2017}  
}

## License:

The pre-trained models and the MSDN technique are released for uncommercial use.

Contact [Yikang LI](http://www.cvboy.com) if you have questions.
