
from tqdm import tqdm
from src.utils import AverageMeter
import torch
from src.callbacks import PlotCbk
from pprint import pformat


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training.
    """
    def __init__(self, model, watch=[], val_watch=[], logger=None):
        self.model = model
        self.stop_training = False
        self.watch = watch
        self.val_watch = val_watch
        self.logger = logger
        if 'loss' not in watch:
            watch.insert(0, 'loss')
        if 'loss' not in val_watch:
            val_watch.insert(0, 'loss')

    def train(self, train_loader, val_loader, start_epoch=0, epochs=200, callbacks=[]):
        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                return
            epoch_log = self.train_one_epoch(epoch, train_loader, callbacks=callbacks)
            val_log = self.validate(epoch, val_loader, callbacks=callbacks)

            msg = ' '.join(['{}: {:.3f}'.format(name, avg) for name, avg in epoch_log.items()])
            self.logger.info(pformat(msg, depth=1))
            msg = ' '.join(['{}: {:.3f}'.format(name, avg) for name, avg in val_log.items()])
            self.logger.info(pformat(msg, depth=1))
            epoch_log.update(val_log)

            for cbk in callbacks:
                cbk.on_epoch_end(epoch, epoch_log)

    def train_one_epoch(self, epoch, train_loader, callbacks=[]):
        """
        Train the model for 1 epoch of the training set.
        """
        epoch_log = {}
        self.model.train()
        self.model.discretizer.train()
        self.model.discretizer.training = True
        for i, (x, y, lts) in enumerate(tqdm(train_loader, unit='batch', desc='Epoch {:>3}'.format(epoch))):
            metric = self.model.forward(x, y, lts, is_training=True, epoch=epoch+1)
            for name in self.watch:
                for key in metric.keys():
                    if key.__contains__(name):
                        if (i == 0): epoch_log[key] = AverageMeter()

                        if isinstance(metric[key], float):  measure = metric[key]
                        else: measure = metric[key].item()
                        epoch_log[key].update(measure, x.size()[0])
                        
        return {name: meter.avg for name, meter in epoch_log.items()}

    @torch.no_grad()
    def validate(self, epoch, val_loader , callbacks=[]):
        """
        Evaluate the model on the validation set.
        """
        val_log = {}
        self.model.eval()
        self.model.discretizer.eval()
        self.model.discretizer.training = False
        for i, (x, y, lts) in enumerate(tqdm(val_loader, unit='batch', desc='Epoch {:>3}'.format(epoch))):
            metric = self.model.forward(x, y, lts, is_training=False, epoch=epoch+1)
            for name in self.watch:
                for key in metric.keys():
                    if key.__contains__(name):
                        if (i == 0): val_log[key] = AverageMeter()
                        if isinstance(metric[key], float):  measure = metric[key]
                        else: measure = metric[key].item()
                        val_log[key].update(measure, x.size()[0])
                        

        for cbk in callbacks:
            if not isinstance(cbk, PlotCbk): continue
            cbk.on_batch_end(epoch, 0, logs=metric)
        #-----------------------------------------------
        return {'val_'+name: meter.avg for name, meter in val_log.items()}



    @torch.no_grad()
    def test(self, test_loader, best=True):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # load the best checkpoint
        # self.load_checkpoint(best=best)

        accs = {}
        self.model.eval()
        for i, (x, y, lts) in enumerate(tqdm(test_loader, unit='batch')):
            metric = self.model.forward(x, y, lts, is_training=False)
            for name in self.watch:
                for key in metric.keys():
                    if key.__contains__(name):
                        if (i == 0): val_log[name] = AverageMeter()
                        val_log[name].update(metric[key].item(), x.size()[0])
                        break

            # for cbk in callbacks:
            #     cbk.on_batch_end(epoch, i, logs=metric)

        test_results = {'test_'+name: meter.avg for name, meter in val_log.items()}
        self.logger.info('Test Acc: ({:.2f}%)'.format(test_results))

    @torch.no_grad()
    def plot(self, data_loader, PlotCallback, name):
        self.model.eval()
        x, y, lts = next(iter(data_loader))
        metric = self.model.forward(x, y, lts, is_training=False)
        PlotCallback.on_batch_end(-1,0, name=name, logs=metric)
