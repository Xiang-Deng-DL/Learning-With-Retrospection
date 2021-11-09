# Learning with Retrospection AAAI'2021
The official PyTorch implementation of Learning with Retrospection.

# Requirements:
pytorch==1.2.0  torchvision==0.4.0

# Run Experiments:
For example, train resnet-18 on CIFAR-100 by LWR: <br>
$python train_LWR.py --model ResNet18 --arch resnet -e 5 --kd_T 5

or train multiple networks:<br>
$sh train_script.sh
