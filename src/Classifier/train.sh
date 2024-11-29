# python train.py \
#         --data_root ../../../datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir Logs/AFHQVanilla \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 &



# python train.py \
#         --data_root ../../../datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir Logs/AFHQDenseNet121 \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 150 



# python train.py \
#         --data_root ../../../datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --model resnet18 \
#         --seed 2022 \
#         --logdir Logs/AFHQResNet18 \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 150  

# python train.py \
#         --data_root ../../../datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --biased \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir Logs/AFHQDenseNet121-Biased \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 150 



# python train.py \
#         --data_root ../../../datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --biased \
#         --model resnet18 \
#         --seed 2022 \
#         --logdir Logs/AFHQResNet18-Biased \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250  


# =====================================

# python train.py \
#         --data_root ../../../datasets/FFHQ/data/ \
#         --nclasses 2 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir Logs/FFHQVanilla \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250  &


python train.py \
        --data_root ../../../datasets/FFHQ/data/ \
        --nclasses 2 \
        --batch_size 32 \
        --input_size 128 \
        --model resnet18 \
        --seed 2022 \
        --logdir Logs/FFHQResNet18 \
        --decreasing_lr '80, 120, 160, 200' \
        --epochs 250  


python train.py \
        --data_root ../../../datasets/FFHQ/data/ \
        --nclasses 2 \
        --batch_size 32 \
        --input_size 128 \
        --model densenet121 \
        --seed 2022 \
        --logdir Logs/FFHQDenseNet121 \
        --decreasing_lr '80, 120, 160, 200' \
        --epochs 250  

python train.py \
        --data_root ../../../datasets/FFHQ/data/ \
        --nclasses 2 \
        --batch_size 32 \
        --input_size 128 \
        --model resnet18 \
        --seed 2022 \
        --logdir Logs/FFHQResNet18 \
        --biased \
        --decreasing_lr '80, 120, 160, 200' \
        --epochs 250  


python train.py \
        --data_root ../../../datasets/FFHQ/data/ \
        --nclasses 2 \
        --batch_size 32 \
        --input_size 128 \
        --model densenet121 \
        --seed 2022 \
        --logdir Logs/FFHQDenseNet121 \
        --biased \
        --decreasing_lr '80, 120, 160, 200' \
        --epochs 250  



# =====================================

# python train.py \
#         --data_root ../../../datasets/MNIST  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir Logs/MNISTVanilla \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 &


# python train.py \
#         --data_root ../../../datasets/MNIST  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model resnet18 \
#         --seed 2022 \
#         --logdir Logs/MNISTResNet18 \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 &


# python train.py \
#         --data_root ../../../datasets/MNIST  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir Logs/MNISTDenseNet121 \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 &

# =================================


# python train.py \
#         --data_root ../../../datasets/SHAPES/shapes  \
#         --nclasses 4 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir Logs/SHAPESVanilla \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 &


# python train.py \
#         --data_root ../../../datasets/SHAPES/shapes  \
#         --nclasses 4 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model resnet18 \
#         --seed 2022 \
#         --logdir Logs/SHAPESResNet18 \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 &


# python train.py \
#         --data_root /../../../datasets/SHAPES/shapes  \
#         --nclasses 4 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir Logs/SHAPESDenseNet121 \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 

# ==============