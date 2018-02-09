# Deep-Alignment-Network-tensorflow
A re-implementation of Deep-Alignment-Network using TensorFlow

## Requirements

Tensorflow 1.5.0 (My version)

Python 3.6

Other commonly used libs for image processing

Dataset formatted like 300-W and Menpo

## Usage

### Data preparation

`cd DAN_TF`

`mkdir Model`


`python training\testSetPreparation.py`

### Training or testing

`python train\testDAN.py`

Remember to set the `STAGE` variable and modify the `data path` in trainDAN.py

### MobileNet training or testing

`mkdir Model_mobilenet`


`python trainDAN_mobilenet.py`

Remember to set the `STAGE` variable and modify the `data path` in trainDAN_mobilenet.py


## What's new?

MobileNet: Added on 2018-02-08

## References

Original implementation: https://github.com/MarekKowalski/DeepAlignmentNetwork

Another tensorflow implementation: https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment

## Remarks

Note that I have no GPU in this period, so I'm not sure whether this implementation can achieve the same performance as the paper described. However, the functionality of the self defined layers has been tested and I found no problem. If you're interested, you can try my code and train a model. Feel free to raise an issue if you have any question.