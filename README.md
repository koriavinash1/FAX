# FAX: Free Argumentative Exchanges

## Environment 
To reproduce all the results please first setup an environment with environment.yml file following the command below:

```
conda env create -f environment.yml
```

```
conda activate FAX
```

## Datasets
Download datasets in an appropriate directory, ideally in '../FAX'



## Example execution with random model
```
export WANDB_USER=<username>
python main.py --config_path configs/mnist/vanilla.yml --case random
```

## Results

To get all the results please consider running `scripts/exps.sh', please be aware that running exps.sh script will run 100+ experiments on all the datasets for 3 different seeds collecting logs with wandb.