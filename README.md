#Pytorch Implementation for paper 'Learning with Retrospection' AAAI-2021

#Requirements:
pytorch==1.2.0  torchvision==0.4.0

#Run Experiments:
For example, train resnet-18 on CIFAR-100 by LWR:
python train_LWR.py --model ResNet18 --arch resnet -e 5 --kd_T 5

or use 'train_script.sh' to train many networks:
sh train_script.sh