# baseline models
python main.py /configs/ffhq/ffhq_vanilla.yml 1
python main.py /configs/ffhq/ffhq_vanilla.yml 2
python main.py /configs/ffhq/ffhq_vanilla.yml 3

python main.py /configs/ffhq/ffhq_resnet.yml 1
python main.py /configs/ffhq/ffhq_resnet.yml 2
python main.py /configs/ffhq/ffhq_resnet.yml 3

python main.py /configs/ffhq/ffhq_densenet.yml 1
python main.py /configs/ffhq/ffhq_densenet.yml 2
python main.py /configs/ffhq/ffhq_densenet.yml 3


# high irrelavence and repetability penalty
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.25 0.25 0.0 0.0

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.25 0.25 0.0 0.0

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.25 0.25 0.0 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.25 0.25 0.0 0.0


# encourage persuasion monotonicity
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.25 0.25 0.25 0.0 
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.25 0.25 0.25 0.0 
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.25 0.25 0.25 0.0 

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.25 0.25 0.25 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.25 0.25 0.25 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.25 0.25 0.25 0.0

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.25 0.25 0.25 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.25 0.25 0.25 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.25 0.25 0.25 0.0


# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.25 0.25 0.25 0.25

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.25 0.25 0.25 0.25

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.25 0.25 0.25 0.25
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.25 0.25 0.25 0.25



# =========================

# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.25 0.25 0.25 0.25 18

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.25 0.25 0.25 0.25 18

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.25 0.25 0.25 0.25 18
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.25 0.25 0.25 0.25 18

# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.25 0.25 0.25 0.25 12

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.25 0.25 0.25 0.25 12

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.25 0.25 0.25 0.25 12
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.25 0.25 0.25 0.25 12




# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0. 0. 0. 0. 18

python main.py /configs/ffhq/ffhq_resnet.yml 1 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_resnet.yml 2 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_resnet.yml 3 0. 0. 0. 0. 18

python main.py /configs/ffhq/ffhq_densenet.yml 1 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_densenet.yml 2 0. 0. 0. 0. 18
python main.py /configs/ffhq/ffhq_densenet.yml 3 0. 0. 0. 0. 18

encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0. 0. 0. 0. 12

python main.py /configs/ffhq/ffhq_resnet.yml 1 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_resnet.yml 2 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_resnet.yml 3 0. 0. 0. 0. 12

python main.py /configs/ffhq/ffhq_densenet.yml 1 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_densenet.yml 2 0. 0. 0. 0. 12
python main.py /configs/ffhq/ffhq_densenet.yml 3 0. 0. 0. 0. 12





# high irrelavence and repetability penalty
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.75 0.75 0.0 0.0

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.75 0.75 0.0 0.0

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.75 0.75 0.0 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.75 0.75 0.0 0.0


# encourage persuasion monotonicity
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.75 0.75 0.75 0.0 
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.75 0.75 0.75 0.0 
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.75 0.75 0.75 0.0 

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.75 0.75 0.75 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.75 0.75 0.75 0.0
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.75 0.75 0.75 0.0

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.75 0.75 0.75 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.75 0.75 0.75 0.0
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.75 0.75 0.75 0.0


# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.75 0.75 0.75 0.75

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.75 0.75 0.75 0.75

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.75 0.75 0.75 0.75
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.75 0.75 0.75 0.75



# =========================

# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.75 0.75 0.75 0.75 18

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.75 0.75 0.75 0.75 18

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.75 0.75 0.75 0.75 18
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.75 0.75 0.75 0.75 18

# encourage persuasion monotonicity and st
python main.py /configs/ffhq/ffhq_vanilla.yml 1 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_vanilla.yml 2 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_vanilla.yml 3 0.75 0.75 0.75 0.75 12

python main.py /configs/ffhq/ffhq_resnet.yml 1 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_resnet.yml 2 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_resnet.yml 3 0.75 0.75 0.75 0.75 12

python main.py /configs/ffhq/ffhq_densenet.yml 1 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_densenet.yml 2 0.75 0.75 0.75 0.75 12
python main.py /configs/ffhq/ffhq_densenet.yml 3 0.75 0.75 0.75 0.75 12

