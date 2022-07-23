#export CUDA_VISIBLE_DEVICES=0

stage=train # or test
cnn=vgg16 # or mobilenetv2 or resnet18

python coarse_model.py stage config/net_$cnn.cfg 1
python coarse_model.py stage config/net_$cnn.cfg 2
python coarse_model.py stage config/net_$cnn.cfg 3
python coarse_model.py stage config/net_$cnn.cfg 4
python coarse_model.py stage config/net_$cnn.cfg 5
