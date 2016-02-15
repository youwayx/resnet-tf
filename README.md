# ResNet TensorFlow

This is a TensorFlow implementation of ResNet-20, a deep residual network developed by [Kaiming He](http://research.microsoft.com/en-us/um/people/kahe/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en), [Shaoqing Ren](http://home.ustc.edu.cn/~sqren/), [Jian Sun](http://research.microsoft.com/en-us/people/jiansun/). 

Read the original paper: "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385).

## Getting Started

Cloning the repo

```shell
$ git clone http://github.com/xuyuwei/resnet-tf
$ cd resnet-tf
```

Setting up the virtualenv, installing TensorFlow (OS X)
```shell
$ virtualenv venv
$ source venv/bin/activate
(venv)$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py2-none-any.whl
```

If you don't have virtualenv installed, run `pip install virtualenv`.

Currently, running `python main.py` starts the training for ResNet-20, saving the progress after training every 512 images.
