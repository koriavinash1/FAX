
DATASET='MNIST'
./slurm.sh $DATASET 4 False False vanilla
./slurm.sh $DATASET 4 False False resnet
./slurm.sh $DATASET 4 False False densenet

./slurm.sh $DATASET 6 False False vanilla
./slurm.sh $DATASET 6 False False resnet
./slurm.sh $DATASET 6 False False densenet

./slurm.sh $DATASET 10 False False vanilla
./slurm.sh $DATASET 10 False False resnet
./slurm.sh $DATASET 10 False False densenet


# ===================================

./slurm.sh $DATASET 4 True False vanilla
./slurm.sh $DATASET 4 True False resnet
./slurm.sh $DATASET 4 True False densenet

./slurm.sh $DATASET 6 True False vanilla
./slurm.sh $DATASET 6 True False resnet
./slurm.sh $DATASET 6 True False densenet

./slurm.sh $DATASET 10 True False vanilla
./slurm.sh $DATASET 10 True False resnet
./slurm.sh $DATASET 10 True False densenet

# ===================================



DATASET='AFHQ'
./slurm.sh $DATASET 4 False False vanilla
./slurm.sh $DATASET 4 False False resnet
./slurm.sh $DATASET 4 False False densenet

./slurm.sh $DATASET 6 False False vanilla
./slurm.sh $DATASET 6 False False resnet
./slurm.sh $DATASET 6 False False densenet

./slurm.sh $DATASET 10 False False vanilla
./slurm.sh $DATASET 10 False False resnet
./slurm.sh $DATASET 10 False False densenet


# ===================================

./slurm.sh $DATASET 4 True False vanilla
./slurm.sh $DATASET 4 True False resnet
./slurm.sh $DATASET 4 True False densenet

./slurm.sh $DATASET 6 True False vanilla
./slurm.sh $DATASET 6 True False resnet
./slurm.sh $DATASET 6 True False densenet

./slurm.sh $DATASET 10 True False vanilla
./slurm.sh $DATASET 10 True False resnet
./slurm.sh $DATASET 10 True False densenet

# ===================================


DATASET='FFHQ'
./slurm.sh $DATASET 4 False False vanilla
./slurm.sh $DATASET 4 False False resnet
./slurm.sh $DATASET 4 False False densenet

./slurm.sh $DATASET 6 False False vanilla
./slurm.sh $DATASET 6 False False resnet
./slurm.sh $DATASET 6 False False densenet

./slurm.sh $DATASET 10 False False vanilla
./slurm.sh $DATASET 10 False False resnet
./slurm.sh $DATASET 10 False False densenet


# ===================================

./slurm.sh $DATASET 4 True False vanilla
./slurm.sh $DATASET 4 True False resnet
./slurm.sh $DATASET 4 True False densenet

./slurm.sh $DATASET 6 True False vanilla
./slurm.sh $DATASET 6 True False resnet
./slurm.sh $DATASET 6 True False densenet

./slurm.sh $DATASET 10 True False vanilla
./slurm.sh $DATASET 10 True False resnet
./slurm.sh $DATASET 10 True False densenet


# ===================================


DATASET='SHAPES'
./slurm.sh $DATASET 4 False False vanilla
./slurm.sh $DATASET 4 False False resnet
./slurm.sh $DATASET 4 False False densenet

./slurm.sh $DATASET 6 False False vanilla
./slurm.sh $DATASET 6 False False resnet
./slurm.sh $DATASET 6 False False densenet

./slurm.sh $DATASET 10 False False vanilla
./slurm.sh $DATASET 10 False False resnet
./slurm.sh $DATASET 10 False False densenet


# ===================================

./slurm.sh $DATASET 4 True False vanilla
./slurm.sh $DATASET 4 True False resnet
./slurm.sh $DATASET 4 True False densenet

./slurm.sh $DATASET 6 True False vanilla
./slurm.sh $DATASET 6 True False resnet
./slurm.sh $DATASET 6 True False densenet

./slurm.sh $DATASET 10 True False vanilla
./slurm.sh $DATASET 10 True False resnet
./slurm.sh $DATASET 10 True False densenet

# ===================================