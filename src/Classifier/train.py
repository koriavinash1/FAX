import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
import model
import numpy as np
from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--channel', type=int, default=3, help='number of input channels')
parser.add_argument('--biased', action='store_true', help='train biased model')

parser.add_argument('--data_root', type=str, default='pytorch-data/', help='dataset root dir')
parser.add_argument('--nclasses', type=int, default=3, help='number of classes')
parser.add_argument('--run_no', type=int, default=1, help='random run')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--input_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--model', default='densenet121', help='model to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
parser.add_argument('--binary', type=bool, default=False)
args = parser.parse_args()


if args.data_root.lower().__contains__('afhq'):
    group = 'AFHQ'
elif args.data_root.lower().__contains__('ffhq'):
    group = 'FFHQ'
elif args.data_root.lower().__contains__('mnist'):
    group = 'MNIST'
elif args.data_root.lower().__contains__('shape'):
    group = 'SHAPE'
else:
    raise ValueError('unknown dataset')




# select gpu
args.ngpu = 1

# logger
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
train_loader, test_loader = dataset.get(data_root=args.data_root,
                                            batch_size=args.batch_size, 
                                            num_workers=1, 
                                            input_size=args.input_size)
if args.model.lower() == 'densenet121':
    model = model.DenseNet121(num_channel=args.channel, classCount=args.nclasses)
elif args.model.lower() == 'densenet161':
    model = model.DenseNet161(num_channel=args.channel, classCount=args.nclasses)
elif args.model.lower() == 'densenet169':
    model = model.DenseNet169(num_channel=args.channel, classCount=args.nclasses)
elif args.model.lower() == 'densenet201':
    model = model.DenseNet201(num_channel=args.channel, classCount=args.nclasses)
elif args.model.lower() == 'resnet18':
    model = model.ResNet18(num_channel=args.channel, classCount=args.nclasses)
elif args.model.lower() == 'vanilla':
    model = model.vanilla(classCount=args.nclasses)

# model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    model.cuda()

#=========================================================================
import wandb
wandb.login()
username = os.environ['WANDB_USER']
print (username)
run = wandb.init(project="FAX-Classifiers", 
                group=group,
                job_type=args.model,
                name=f'Run-{args.run_no}',
                # config=vars(args), 
                dir=args.logdir,
                entity = username, 
                reinit=True,
                settings=wandb.Settings(start_method='thread'))


#===========================================================================



# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()

os.makedirs(args.logdir, exist_ok=True)

try:
    # ready to go
    for epoch in range(args.epochs):
        model.train()
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for batch_idx, (data, target, _) in enumerate(train_loader):
            if args.biased:
                batch_idxs = np.arange(target.shape[0])
                np.random.shuffle(batch_idxs)
                target[batch_idxs[:int(0.5*target.shape[0])]] = 0

                
            indx_target = target.clone()

            if args.binary:
                indx_target = torch.argmax(target, 1)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            if args.nclasses > 1:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.binary_cross_entropy(torch.sigmoid(output), target.view(-1, 1))
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = correct * 1.0 / len(data)
                wandb.log({'Train-Acc': acc})

                print(args.logdir,'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.item(), acc))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        print(args.data_root, "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        
        torch.save(model.state_dict(), os.path.join(args.logdir, 'latest.pth'))

        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target, _ in test_loader:
                indx_target = target.clone()

                if args.binary:
                    indx_target = torch.argmax(target, 1)


                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                if args.nclasses > 1:
                    test_loss += F.cross_entropy(output, target).item()
                else:
                    test_loss += F.binary_cross_entropy(torch.sigmoid(output), target.view(-1, 1)).item()

                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader) # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            wandb.log({'Val-Acc': acc})
            print(args.data_root, '\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best.pth')
                torch.save(model.state_dict(), new_file)
                best_acc = acc
                old_file = new_file
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print(args.data_root, "Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


