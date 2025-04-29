# SRCNN

This repository is implementation of the ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092).

<center><img src="./thumbnails/fig1.png"></center>

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train
首先应使用prepare.py将自己的数据集处理成train.h5,和eval.h5的形式用于训练和测试。也可以使用SRCNN/SRCNN-pytorch/data/traindata中的数据进行训练和测试

train.py使用方法
```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --num-workers 8 \
                --seed 123                
```

## Test
test.py使用方法
```bash
python test.py --weights-file "BLAH_BLAH/srcnn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```