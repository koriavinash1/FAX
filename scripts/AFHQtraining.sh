QUANTIZE='spatial'
LOGSDIR='/AFHQ/'$4'-C'$2'-G'$3'-DS'$1


FEATUREEXTRACTOR=$4
SPLITEPOCH=10
ENDEPOCH=15

RNNHIDDEN=128
RNNINPUT=128


NCONCEPTS=32
MODULATEDCHANNELS=64
CDIM=$MODULATEDCHANNELS


if [[ "$FEATUREEXTRACTOR" == "vanilla" ]]; then
        NFEATURES=64
        IMAGESIZE=32
elif [[ "$FEATUREEXTRACTOR" == "resnet" ]]; then
        NFEATURES=512
        IMAGESIZE=128
else
        NFEATURES=1024
        IMAGESIZE=128
fi


# python main.py \
#         --data_dir= datasets/AFHQ/afhq \
#         --ckpt_dir=$LOGSDIR/ckpt \
#         --plot_dir=$LOGSDIR/plots \
#         --img_size=$IMAGESIZE \
#         --epoch=$SPLITEPOCH \
#         --nfeatures=$NFEATURES \
#         --cdim=$CDIM \
#         --rnn_hidden=$RNNHIDDEN \
#         --rnn_input_size=$RNNINPUT \
#         --nconcepts=$NCONCEPTS \
#         --modulated_channels=$MODULATEDCHANNELS \
#         --feature_extractor=$FEATUREEXTRACTOR \
#         --quantize=$QUANTIZE \
#         --contrastive=False \
#         --batch_size=16 \
#         --device=0 \
#         --print_freq=50 \
#         --plot_freq=1 \
#         --n_class=3 \
#         --init_lr=1e-3 \
#         --narguments=$1 \
#         --cosine=$2 \
#         --gumbel=$3 \
#         --name=$4 



# echo Debate Completed


# python main.py \
#         --data_dir=datasets/AFHQ/afhq \
#         --ckpt_dir=$LOGSDIR/ckpt \
#         --plot_dir=$LOGSDIR/plots \
#         --img_size=$IMAGESIZE \
#         --epoch=$ENDEPOCH \
#         --nfeatures=$NFEATURES \
#         --cdim=$CDIM \
#         --rnn_hidden=$RNNHIDDEN \
#         --rnn_input_size=$RNNINPUT \
#         --nconcepts=$NCONCEPTS \
#         --modulated_channels=$MODULATEDCHANNELS \
#         --feature_extractor=$FEATUREEXTRACTOR \
#         --quantize=$QUANTIZE \
#         --contrastive=True \
#         --resume=True \
#         --batch_size=16 \
#         --device=0 \
#         --print_freq=50 \
#         --plot_freq=1 \
#         --n_class=3 \
#         --init_lr=5e-5 \
#         --narguments=$1 \
#         --cosine=$2 \
#         --gumbel=$3 \
#         --name=$4 

# echo Debate Completed


echo Plotting logs from:---------
echo $LOGSDIR/plots/Exp-$4-Debate:GRU_${1}_${NCONCEPTS}_${RNNINPUT}_${RNNHIDDEN}_1/Exp-$4-Debate:GRU_${1}_${NCONCEPTS}_${RNNINPUT}_${RNNHIDDEN}_1


python plots.py \
       --plot_dir $LOGSDIR/plots/Exp-$4-Debate:GRU_${1}_${NCONCEPTS}_${RNNINPUT}_${RNNHIDDEN}_1/Exp-$4-Debate:GRU_${1}_${NCONCEPTS}_${RNNINPUT}_${RNNHIDDEN}_1 \
       --quantize=$QUANTIZE \
       --start_epoch 0 \
       --stop_epoch $(expr $ENDEPOCH - 1) \
       --split_epoch $(expr $SPLITEPOCH - 1)