#!/bin/bash
sbatch <<EOT
#!/bin/bash
# Example of running python script in a batch mode 
#SBATCH -c 1                       # Number of CPU Cores 
#SBATCH -p gpus                    # Partition (queue) 
#SBATCH --gres gpu:1               # gpu:n, where n = number of GPUs 
#SBATCH --mem 32G                  # memory pool for all cores 
#SBATCH --output=$1-$5.%N.%j.log   # Standard output and error loga

echo $1, $2, $3, $4, $5

# Source Virtual environment (conda)
. anaconda3/etc/profile.d/conda.sh
conda activate stylEx

if [ "$1" = "AFHQ" ]; then
    ./AFHQtraining.sh $2 $3 $4 $5
elif [ "$1" = "SHAPES" ]; then
    ./SHAPEStraining.sh $2 $3 $4 $5
elif [ "$1" = "FFHQ" ]; then
    ./FFHQtraining.sh $2 $3 $4 $5
else
    ./MNISTtraining.sh $2 $3 $4 $5
fi
EOT