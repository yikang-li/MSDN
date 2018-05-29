# Multi-level Scene Description Network


This is our implementation of **Multi-level Scene Description Network** in [Scene Graph Generation from Objects, Phrases and Region Captions](http://cvboy.com/pdf/iccv2017_msdn.pdf). The project is based on PyTorch version of [faster R-CNN](https://github.com/longcw/faster_rcnn_pytorch). 


## Progress
- [x] README for training 
- [x] README for project settings
- [x] our trained RPN
- [x] our trained Full Model
- [x] Our cleansed Visual Genome Dataset
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
    conda install -c soumith pytorch torchvision cuda80 
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
    cd ..
    ```
4. Download the [trained full model](https://drive.google.com/file/d/1NWjVMKfa6_ce2MguRLG6lGdrEF3TvwlI/view?usp=sharing) and [trained RPN](https://drive.google.com/file/d/1-Jewgg9tfZm5c-beAsCdNWEWKh9GuKa7/view?usp=sharing), and place it to ```output/trained_model```

5. Download our [cleansed Visual Genome dataset](https://drive.google.com/file/d/1RtYidFZRgX1_iYPCaP2buHI1bHacjRTD/view?usp=sharing). And unzip it:
```
tar xzvf top_150_50.tgz
```
- p.s. Our ipython [scripts](https://github.com/yikang-li/vg_cleansing) for data cleansing is also released. 


6. Download [Visual Genome images](http://visualgenome.org/api/v0/api_home.html)

7. Place *Images and* *cleansed annotations* to coresponding folders:
```
mkdir -p data/visual_genome
cd data/visual_genome
ln -s /path/to/VG_100K_images_folder VG_100K_images
ln -s /path/to/downloaded_folder top_150_50
```
- p.s. You can change the default data directory by modifying ```__C.IMG_DATA_DIR``` in ```faster_rcnn/fast_rcnn/config.py``` 

## Training
- Training in multiple stages. (Single-GPU training may take about one week.)
	1. Training RPN for object proposals and caption region proposals (the **shared conv layers** are fixed). We also provide [our pretrained RPN](https://www.dropbox.com/s/fazqfcs6bhbe051/RPN_region_full_best.h5?dl=0) model. 

	by default, the training is done on a small part of the full dataset:
	```
	CUDA_VISIBLE_DEVICES=0 python train_rpn.py
	```

	For full Dataset Training:
	```
	CUDA_VISIBLE_DEVICES=0 python train_rpn_region.py --max_epoch=10 --step_size=2 --dataset_option=normal --model_name=RPN_full_region
	```

	```--step_size``` is set to indicate the number of epochs to decay the learning rate, ```dataset_option``` is to indicate the ```\[ small | fat | normal \]``` subset. 

	2. Training MSDN

	Here, we use SGD (controled by ```--optimizer```)by default:
	```
	CUDA_VISIBLE_DEVICES=0 python train_hdn.py --load_RPN --saved_model_path=./output/RPN/RPN_region_full_best.h5  --dataset_option=normal --enable_clip_gradient --step_size=2 --MPS_iter=1 --caption_use_bias --caption_use_dropout --rnn_type LSTM_normal 
	```
- Furthermore, we can directly use end-to-end training from scratch (not recommended). The result is not good. 
	```
	CUDA_VISIBLE_DEVICES=0 python train_hdn.py  --dataset_option=normal --enable_clip_gradient  --step_size=3 --MPS_iter=1 --caption_use_bias --caption_use_dropout --max_epoch=11 --optimizer=1 --lr=0.001
	```


## Evaluation 

Our [pretrained full Model](https://www.dropbox.com/s/vg1lseklk1f86z0/HDN_1_iters_alt_normal_I_LSTM_with_bias_with_dropout_0_5_nembed_256_nhidden_512_with_region_regression_resume_SGD_best.h5?dl=0) is provided for your evaluation for further implementation. (Please download the related files in advance.)


```
./eval.sh
```

Currently, the accuracy of our released version is slightly different from the reported results in the paper:Recall@50: **11.705%**; Recall@100: **14.085%**.

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

Contact [Yikang LI](http://www.cvboy.com/) if you have questions.
