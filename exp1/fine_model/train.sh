#export CUDA_VISIBLE_DEVICES=0

stage=train # or test
cnn=vgg16 # or mobilenetv2 or resnet18

python fine_model.py stage config/net_$cnn.cfg 1
python fine_model.py stage config/net_$cnn.cfg 2
python fine_model.py stage config/net_$cnn.cfg 3
python fine_model.py stage config/net_$cnn.cfg 4
python fine_model.py stage config/net_$cnn.cfg 5
