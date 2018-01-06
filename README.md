# DSS-pytorch
Pytorch implement of [Deeply Supervised Salient Object Detection with Short Connection](https://arxiv.org/abs/1611.04849)

<p align="center"><img width="80%" src="png/dss.png" /></p>

The official caffe version: [DSS](https://github.com/Andrew-Qibin/DSS)

## Prerequisites

- [Python 3](https://www.continuum.io/downloads)
- [Pytorch 0.3.0](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- [visdom](https://github.com/facebookresearch/visdom) (optional for visualization)

## Results

The information of Loss:

![](./png/loss.png)

Example output:

![](png/example.png)

## Usage

### 1. Clone the repository

```shell
git clone git@github.com:AceCoooool/DSS-pytorch.git
cd DSS-pytorch/
```

### 2. Download the dataset

Note: the original paper use other datasets.

Download the [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset.  (see [NLFD-pytorch](https://github.com/AceCoooool/NLFD-pytorch/blob/master/download.sh))

```shell
bash download.sh
```

### 3. Get pre-trained vgg

```bash
cd tools/
python extract_vgg.py
cd ..
```

### 4. Demo (coming soon)

```shell
python demo.py --demo_img='your_picture' --trained_model='pre_trained pth' --cuda=True
```

Note: 

1. default choose: download and copy the [pretrained model]() to `weights` directory. (add soon)
2. a demo picture is in `png/demo.jpg`

### 5. Train

```shell
python main.py --mode='train' --train_path='you_data' --label_path='you_label' --batch_size=8 --visdom=True
```

Note:

1. `--val=True` add the validation (but your need to add the `--val_path` and `--val_label`)
2. `you_data, you_label` means your training data root. (connect to the step 2)

### 6. Test

```shell
python main.py --mode='test', --test_path='you_data' --test_label='your_label' --batch_size=1 --model='your_trained_model'
```

## TODO

- [ ] add RCF process
- [ ] test other connection situation

