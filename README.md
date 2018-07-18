# A Pytorch Implementation of RetinaNet for object detection 

## Introduction

This project is a pytorch implementation of RetinaNet. During the implementing, I referred several implementations to 
make this project work:
* [kuangliu/pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet), this repository give several main 
scripts to train RetinaNet, but doesn't give the results of training.
* [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet), this repository completely give the training, 
test, evaluate processes, but it is based on Keras.
* [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch), this repository is a implementation
of Detectron based on pytorch, but it doesn't support RetinaNet in this moment.


For this implementation, it has the following features:
* **It supports multi-image batch training**. Change the original sampler into `MinibatchSampler` to support 
different multi-image size in minibatch.
* **It supports multiple GPUs training**. Change the original `DataParallel` in Pytorch to support minibatch 
supported dataset.

## Results of RetinaNet
Now, I get the result using COCOAPI, the training AP is 29.2, compare to 34.0 in the original paper, it's not good.
I will figure out where the problem is.

## Getting Started
Clone the repo:

```
git clone https://github.com/wsnedy/pytorch-retinanet.git
```
### Requirements

Tested under python3.

- python packages
  - pytorch=0.3.1
  - torchvision=0.2.0
  - matplotlib
  - numpy
  - opencv
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

### Get pretrained model
Download pretrained ResNet50 params from the following url.
```
mkdir pretrained_model
cd pretrained_model
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv resnet50-19c8e357.pth resnet50.pth
```
Get the pretrained RetinaNet by run the script:
```
cd network
python get_state_dict.py
```

### Data Preparing
Download the coco images and annotations from [coco website](http://cocodataset.org/#download).

  And make sure to put the files as the following structure:
  ```
  coco
  ├── annotations
  |   ├── instances_minival2014.json
  │   ├── instances_train2014.json
  │   ├── instances_val2014.json
  │   ├── instances_valminusminival2014.json
  │   ├── ...
  |
  └── images
      ├── train2014
      ├── val2014
      ├── ...
  ```
  When training, change the root path to your own data path.
 
 ### Training model
 For the hyper-parameters, I just put them in the scripts, and I will put all the hyper-parameters in a config file.
 The setting is as follows:
 - For multi-gpus, it will use all the available gpus in default. change the `device_ids` in `bin/train.py` if you want
 to specific gpus.
 - For batch_size, I use `batch_size = 24`, if you want to change, you have to change two places, `batch_size=24` and 
`iteration_per_epoch = int(len(dataloader) / 24.)` in `bin/train.py`
 - For img_per_minibatch, I use `img_per_minibatch = 3` to achieve `batch_size=24`, change it if you want to use other 
 minibatch number.
 - To do, put all the hyper-parameters into config file.
 
 ### Training from scratch
 Training RetinaNet using following code, and after each epoch, it will give a evaluation in `minival2014` dataset:
 ```
 python bin/train.py
```

### Training from a pretrained checkpoint
If you want to load the checkpoint, use the follow code. If you have more than one checkpoint, change the code 
`checkpoint = torch.load('../checkpoint/ckpt.pth')` in `train.py` to load different checkpoint:
```angular2html
python bin/train.py --resume
```

### Inference
- To do: `demo.py`

### Support Dataset
Only COCO supported now, for different dataset, change a little bit in `datasets` will be work.
- To do: support `VOC` dataset.

