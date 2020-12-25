python train_LWR.py --model ResNet18 --arch resnet -e 5 --kd_T 5
python train_LWR.py --model ShuffleV2 --arch shuffle -e 5 --kd_T 5
python train_LWR.py --model wrn_16_4 --arch wrn -e 1 --kd_T 5
python train_LWR.py --model resnet56 --arch resnet -e 5 --kd_T 5
python train_LWR.py --model vgg16 --arch vgg -e 50 --kd_T 5