import os
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from src.debate import Debate
from src.callbacks import PlotCbk, ModelCheckpoint, LearningRateScheduler, EarlyStopping, PropertiesLogger
from src.trainer import Trainer
from src.utils import seed_everything
from src.Classifier.dataset import get
from src.sharedAF import SharedAF 
from src.privateAF import PrivateGAF
import argparse

import yaml
import sys



class Arguments():
    def __init__(self, path):
        self.load(path)

    def load(self, path):
        self.full_dict = yaml.load(open(path), Loader=yaml.FullLoader)

        for rootkey, value in self.full_dict.items():
            if rootkey == 'defaults':
                for path in value:
                    self.load(path)
            else:
                for key, value in value.items():
                    self.full_dict[rootkey][key] = value
                    setattr(self, key, value)



if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--config_path', type=str, help='path to .yaml files')
    opt.add_argument('--run', default = 1, type=int, help='run idx')
    opt.add_argument('--narguments', default = 4, type=int, help='number of arguments during training')
    opt.add_argument('--case', default = 'fair', type=str, help='fair, biased, or random model')
    opt.add_argument('--lambda2', default=0.0, type=float, help='path to .yaml files')
    opt.add_argument('--lambda3', default=0.0, type=float, help='path to .yaml files')
    opt.add_argument('--lambda4', default=0.0, type=float, help='path to .yaml files')
    opt.add_argument('--lambda5', default=0.0, type=float, help='path to .yaml files')

    opt = opt.parse_args()
    config_path = opt.config_path
    print (config_path)
    args = Arguments(config_path)

    run = opt.run

    np.random.seed(run)
    seed = np.random.randint(2000, 5000)

    args.case = opt.case
    args.full_dict['base']['case'] = opt.case

    # update dicts before saving----------------
    args.random_seed = seed; args.run = run
    args.full_dict['base']['run'] = run
    args.full_dict['base']['random_seed'] = seed

    lambda2 = opt.lambda2
    args.lambda2 = lambda2
    args.full_dict['debate']['lambda2'] = lambda2

    lambda3 = opt.lambda2
    args.lambda3 = lambda3
    args.full_dict['debate']['lambda3'] = lambda3
    
    lambda4 = opt.lambda2
    args.lambda4 = lambda4
    args.full_dict['debate']['lambda4'] = lambda4

    lambda5 = opt.lambda2
    args.lambda5 = lambda5
    args.full_dict['debate']['lambda5'] = lambda5

    narguments = opt.narguments
    args.narguments = narguments
    args.full_dict['debate']['narguments'] = narguments


    logger = logging.getLogger(f'VXs-Training from: {config_path}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%m-%d %H:%M")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ensure reproducibility
    seed_everything(args.random_seed)

    # setup logging
    group = f'Debate-n-{args.narguments}-q-{args.quantize}-l2-{args.lambda2}-l3-{args.lambda3}-l4-{args.lambda4}-l5-{args.lambda5}'
    sampling_type = 'Euclidian' if not (args.cosine or args.gumble) else ('Gumble' if args.gumble else 'Cosine')
    run_name = f'Run={args.run}'

    logs_dir = os.path.join(args.log_dir, group, sampling_type, run_name)
    os.makedirs(logs_dir, exist_ok=True)

    # update_paths
    args.log_dir = logs_dir
    args.ckpt_dir = logs_dir


    # setup wandb
    if args.use_wandb:
        import wandb 
        wandb.login()
        username = os.environ['WANDB_USER']
        run = wandb.init(project="FAX" if not args.use_cb_multiplicity else 'FAX_CBM' + '-' + args.name, 
                        group=group,
                        job_type=args.feature_extractor +'-'+ sampling_type,
                        name=run_name,
                        config={"group": group, 
                                'sampling': sampling_type,
                                'run_name': run_name, 
                                **args.full_dict}, 
                        dir=logs_dir,
                        entity = username, 
                        reinit=True)


    # augmentation setup
    kwargs = {'num_workers': args.num_workers,
               'input_size': args.img_size}
    if args.is_train:  
        train_loader, val_loader = get(args.batch_size, args.data_root, train=True, val=True, **kwargs)

        args.num_class = args.n_class
        args.num_channels = train_loader.dataset.num_channels
        mapping = train_loader.dataset.mapping
    else:
        train_loader, val_loader = get(args.batch_size, args.data_root, val=True, **kwargs)

        args.num_class = args.n_class
        args.num_channels = val_loader.dataset.num_channels
        mapping = val_loader.dataset.mapping


    # Create Private pseudo-GAF
    private_GAF = [PrivateGAF(args, iagent) for iagent in range(args.nagents)]

    # Create Shated BAF
    shared_BAF = SharedAF(class_mapping = mapping, 
                            nplayers    = args.nagents,
                            similarity  = args.termination, 
                            threshold   = args.termination_threshold,
                            lambda2     = args.lambda2,
                            lambda3     = args.lambda3,
                            lambda4     = args.lambda4,
                            lambda5     = args.lambda5)



    # build Debate model
    model = Debate(private_GAF, shared_BAF, args)
    
 


    if args.use_gpu: model.cuda()

    logger.info('Number of model parameters: {:,}'.format(
                sum([p.data.nelement() for p in model.parameters()])))

    watch = ['acc', 'loss', 'reward', \
            'faithfulness', 'correctness', 'reliability', \
            'consensus', 'contribution_rate', 'repetability',\
            'irrelevance', 'perplexity', 'persuasion_strength', \
            'persuasion_monotonicity', 'persuasion_rate', 'arr', 'frr' ]
    trainer = Trainer(model, 
                        watch=watch, 
                        val_watch=watch, 
                        logger=logger)


    if args.is_train:
        logger.info("Train on {} samples, validate on {} samples".format(len(train_loader.dataset), len(val_loader.dataset)))
        start_epoch = 0
        if args.resume:
            start_epoch = model.load_model(args.ckpt_dir, 
                                            best=args.best)


        # best model selection method is based on acc in callback.py
        # Need to fix that before changing monitor_val
        trainer.train(train_loader, val_loader,
                      start_epoch=start_epoch,
                      epochs=args.epochs,
                      callbacks=[
                          PlotCbk(plot_dir    = args.log_dir, 
                                    model     = model, 
                                    num_imgs  = args.plot_num_imgs, 
                                    plot_freq = args.plot_freq,
                                    quantize  = args.quantize,
                                    threshold = args.plot_threshold, 
                                    use_gpu   = args.use_gpu,
                                    nagents   = args.nagents,
                                    logger    = wandb if args.use_wandb else None),
                          PropertiesLogger(log_dir  = args.log_dir,
                                            nagents = args.nagents,
                                            watch   = watch,
                                            logger  = wandb if args.use_wandb else None),
                          ModelCheckpoint(model = model, 
                                          ckpt_dir = args.ckpt_dir,
                                          monitor_val = 'val_initial_acc_agent-0'),
                          LearningRateScheduler(model.quantized_optimizer, 
                                                factor = 0.1, 
                                                patience = 3, 
                                                mode = 'max', 
                                                monitor_val = 'val_cqloss')
                      ])
