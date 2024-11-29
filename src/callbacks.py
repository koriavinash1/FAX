import warnings

import numpy as np

import pickle
import os
import torch
import shutil

# from src.tflogger import TFLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2, copy
from scipy import stats

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

font = {'size'   : 20}
matplotlib.rc('font', **font)

class Callback(object):
    '''Abstract base class used to build new callbacks.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''
    def __init__(self, model):
        self.model = model

    def on_train_beg(self):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, epoch, batch, name='', logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

normalize = lambda x: np.uint8(255*(x - x.min())/(x.max() - x.min())).transpose(1, 2, 0)
normalize_ft = lambda x: np.uint8(255*(x - x.min())/(x.max() - x.min()))

class PlotCbk(Callback):
    def __init__(self, plot_dir, 
                        model, 
                        num_imgs, 
                        plot_freq, 
                        quantize, 
                        threshold, 
                        use_gpu, 
                        nagents,
                        logger=None):
        self.model     = model
        self.nagents   = nagents
        self.logger    = logger
        self.use_gpu   = use_gpu
        self.num_imgs  = num_imgs
        self.quantize  = quantize
        self.threshold = threshold
        self.plot_freq = plot_freq
        self.plot_dir  = os.path.join(plot_dir, self.model.name+'/')
        
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)


    def _perform_postprocessing(self, img, threshold=80):
        """
            connected component analysis with appropreate threshold

            img       : test_image for thresholding
            threshold : area threshold for selecting max area 
                    components
        """

        c,n = label(img)
        nums = np.array([np.sum(c==i) for i in range(1, n+1)])
        selected_components = np.array([threshold<num for num in nums])
        selected_components[np.argmax(nums)] = True
        mask = np.zeros_like(img)
        for i,select in enumerate(selected_components):
            if select:
                mask[c==(i+1)]=1
        return mask



    def _convert_to_blob_(self, argument, size):
        argument = cv2.resize(normalize_ft(argument), tuple(size), \
                        interpolation = cv2.INTER_CUBIC)
        argument = (argument - argument.min())/(argument.max() - argument.min() + 1e-5)
        argument = 255.0*(argument > 0.7)
        contours, hierarchy = cv2.findContours(np.uint8(argument), 
                                                    cv2.RETR_LIST, 
                                                    cv2.CHAIN_APPROX_NONE)

        return_argument = np.zeros_like(argument)
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            perimeter = cv2.arcLength(hull, True)
            # if perimeter < 0.5*np.sum(size):
            cv2.drawContours(return_argument, [hull], -1, 255, -1)


        kernel = np.ones((5, 5), np.uint8)
        return_argument = cv2.dilate(return_argument, kernel, iterations=2)
        return np.array(return_argument)/255.0 
    

    def get_visual_arguments(self, x, z, 
                                sampled_idx, 
                                arg_idx = []):
        size = np.asarray(x.shape[1:])
        return_arguments = [] #nargs x bs
        argument_idx = []

        z_pertub = []; batch_arg = []
        for idx in arg_idx:
            z_ = copy.deepcopy(z)
            if self.quantize == 'channel':
                z_[sampled_idx == sampled_idx[idx]] = 0 
                batch_arg.append(sampled_idx[idx])
            else:
                z_ = z_.reshape(z_.shape[0], -1)
                z_[:, sampled_idx == sampled_idx[idx]] = 0
                z_ = z_.reshape(z.shape)
                batch_arg.append(sampled_idx[idx])

            z_pertub.append(z_)

        z_pertub = np.array(z_pertub)

        F = z[None, ...] - z_pertub 
        
        Bmask = 1.0*(np.mean(F*(F >= np.percentile(F, self.threshold, 1)[:, None, ...]), 1))
        Bmask = np.array([self._convert_to_blob_(Fd, size) for Fd in Bmask])

        return Bmask[..., None], np.array(batch_arg)



    def plot_img(self, ibatch, x, y, jpred, dpred, zs_idx, arguments, claims):
        nagents = len(arguments)
        narguments = len(arguments[0])

        nrows = nagents; ncols = narguments + nagents
        plt.clf()
        fig = plt.figure()
        fig.set_figheight(nrows*3)
        fig.set_figwidth(ncols*3)
        fig.set_dpi(100)

        title = '$Y={}$, $\mathcal{{J}}(x)={}$, $\Gamma(x)={}$, \n'.format(y, jpred, dpred)

        for aidx in range(nagents):
            title += '$\Sigma^{}(x)={}$, '.format(aidx+1, claims[aidx])
                                    
            for argidx in range(ncols):
                if argidx == 0:
                    ax = plt.subplot2grid(shape     = (nrows, ncols), 
                                            loc     = (0, 0), 
                                            colspan = nagents,
                                            rowspan = nagents)
                    ax.imshow(normalize(x),cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title(title)
                    
                elif argidx < nagents:
                    continue

                else:
                    ax = plt.subplot2grid(shape=(nrows, ncols), 
                                            loc=(aidx, argidx))

                    input_img = normalize(x)
                    mask = arguments[aidx][argidx - nagents]
                    blend_mask = np.zeros_like(input_img)
                    blend_mask[mask[:, :, 0] == 0] = (0, 0, 0)
                    blend_mask[mask[:, :, 0] > 0] = (255, 255, 255)
                    plotarg_ = cv2.addWeighted(input_img, 0.3, blend_mask, 0.7, 0.0)
                    
                    ax.imshow(plotarg_)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title('$\mathcal{{A}}^{{}}_{{}} = {}$'.format(aidx +1, 
                                                            argidx - nagents + 1,
                                                            zs_idx[aidx][argidx - nagents]))


                    # ===============================================

            plt.tight_layout()
            # save as png
            if not (self.logger is None): 
                self.logger.log({f'val-plot-{ibatch}': plt})
            else:
                path = os.path.join(os.path.dirname(self.plot_dir), 'pngs/')
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, 'ibatch_{}.png'.format(ibatch))
                plt.savefig(path, bbox_inches='tight')


        
    def plot(self, imgs, zs, zs_idx, arguments, dpreds, preds, jpreds, Ys, epoch, batch_ind, name):
        if self.use_gpu:
            imgs   = imgs.detach().cpu().numpy()
            jpreds = jpreds.cpu().numpy()
            dpreds = dpreds.cpu().numpy()
            Ys     = Ys.cpu().numpy()       
            zs     = zs.detach().cpu().numpy()    
            preds  = [preds_.cpu().numpy() for preds_ in preds]
            zs_idx = [zs_idx_.cpu().numpy() for zs_idx_ in zs_idx]
            arguments = [arguments_.cpu().numpy() for arguments_ in arguments]

        
        

        for i in range(imgs.shape[0]):
            if i > self.num_imgs: continue
            visual_arguments = []; argument_idxs = []
            for ia in range(len(zs_idx)):
                visual_argument, argument_idx = self.get_visual_arguments(imgs[i], zs[i], 
                                                                            zs_idx[ia][i], 
                                                                            arguments[ia][:, i])
                visual_arguments.append(visual_argument)
                argument_idxs.append(argument_idx)
                
            claims   = [claim[i] for claim in preds]
            self.plot_img(batch_ind + i, imgs[i], Ys[i], jpreds[i], dpreds[i], argument_idxs, visual_arguments, claims)


    def on_batch_end(self, epoch, batch_ind, name='',logs={}):

        imgs   = logs['x']
        jpreds = logs['jpred']
        dpreds = logs['dpred']
        Ys     = logs['y']

        zs_idxs = []
        claims = []; arguments = []
        for ai in range(self.nagents):
            zs_idxs.append(logs[f'z_idx_agent-{ai}'])
            claims.append(logs[f'initial_claims_agent-{ai}'])
            arguments.append(logs[f'argument_trace_agent-{ai}'])

        self.plot(imgs, logs['z'], zs_idxs, arguments, dpreds, claims, jpreds, Ys, epoch, batch_ind, name)


class PropertiesLogger(Callback):
    def __init__(self, log_dir, watch, nagents,logger=None):
        self.logs_dir = log_dir
        self.logger   = logger
        self.nagents  = nagents
        self.watch    = watch
        self.csv_logs = []
        os.makedirs(self.logs_dir, exist_ok=True)

    def csv_logger(self, logs):
        csv_path = os.path.join(self.logs_dir, f'logs_.csv')
        
        csv_logs = {}

        for key in logs.keys():
            for watch_key in self.watch:
                if key.lower().__contains__(watch_key.lower()):
                    csv_logs[key] = logs[key]


        # for ai in range(self.nagents):
        #     csv_logs[f'Train-P{ai}_inital_acc'] = logs[f'initial_acc_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_final_acc']  = logs[f'final_acc_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_rlloss']     = logs[f'rlloss_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_reward']     = logs[f'reward_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_repetability']      = logs[f'repetability_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_irrelevance']       = logs[f'irrelevance_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_persuasion_monotonicity'] = logs[f'persuasion_monotonicity_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_persuasion_strength']     = logs[f'persuasion_strength_agent-{ai}']
        #     csv_logs[f'Train-P{ai}_persuasion_rate']         = logs[f'persuasion_rate_agent-{ai}']

        
        # csv_logs['Train_faithfulness']   = logs['faithfulness']
        # csv_logs['Train_correctness']    = logs['correctness']
        # csv_logs['Train_reliability']    = logs['reliability']
        # csv_logs['Train_consensus']      = logs['consensus']
        # csv_logs['Train_cqloss']         = logs['cqloss']
        # csv_logs['Train_perplexity']     = logs['perplexity']
        # csv_logs['Train_contribution_rate'] = logs['contribution_rate']

        # # ======================================
        # for ai in range(self.nagents):
        #     csv_logs[f'Valid-P{ai}_inital_acc'] = logs[f'val_initial_acc_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_final_acc']  = logs[f'val_final_acc_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_rlloss']     = logs[f'val_rlloss_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_reward']     = logs[f'val_reward_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_repetability']      = logs[f'val_repetability_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_irrelevance']       = logs[f'val_irrelevance_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_persuasion_monotonicity'] = logs[f'val_persuasion_monotonicity_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_persuasion_strength']     = logs[f'val_persuasion_strength_agent-{ai}']
        #     csv_logs[f'Valid-P{ai}_persuasion_rate']         = logs[f'val_persuasion_rate_agent-{ai}']

        
        # csv_logs['Valid_faithfulness']   = logs['val_faithfulness']
        # csv_logs['Valid_correctness']    = logs['val_correctness']
        # csv_logs['Valid_reliability']    = logs['val_reliability']
        # csv_logs['Valid_consensus']      = logs['val_consensus']
        # csv_logs['Valid_cqloss']         = logs['val_cqloss']
        # csv_logs['Valid_perplexity']     = logs['val_perplexity']
        # csv_logs['Valid_contribution_rate'] = logs['val_contribution_rate']


        if not (self.logger is None): 
            self.logger.log(csv_logs)
        else:
            self.csv_logs.append(csv_logs)
            df = pd.DataFrame(self.csv_logs)
            df.to_csv(csv_path, index=False)

    def on_epoch_end(self, epoch, logs, name=''):
        self.csv_logger(logs)


class TensorBoard(Callback):
    def __init__(self, model, log_dir):
        self.model = model
        self.logger = TFLogger(log_dir)

    def to_np(self, x):
        return x.data.cpu().numpy()

    def on_epoch_end(self, epoch, logs, name=''):
        for tag in ['loss', 'acc']:
            self.logger.scalar_summary(tag, logs[tag], epoch)

        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, self.to_np(value), epoch)
            self.logger.histo_summary(tag+'/grad', self.to_np(value.grad), epoch)


class ModelCheckpoint(Callback):
    def __init__(self, model, ckpt_dir, monitor_val):
        self.model = model
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.monitor_val = monitor_val
        suff = ''

        filename = self.model.name + suff + '_ckpt'
        self.ckpt_path = os.path.join(ckpt_dir, filename)
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs={}):
        state = {'epoch': epoch}
        
        for k, v in logs.items():
            state[k] = v

        state.update(self.model.get_state_dict())
        torch.save(state, self.ckpt_path)

        if logs[self.monitor_val] >= self.best_val_acc:
            shutil.copyfile(self.ckpt_path, self.ckpt_path + '_best')
            print ("Model saved: epoch: {}, acc:{}, last-max-acc: {}, path: {}".format(epoch, logs[self.monitor_val], self.best_val_acc, self.ckpt_path))
            self.best_val_acc = logs[self.monitor_val]


class LearningRateScheduler(Callback):
    def __init__(self, optimizer, factor, patience, mode, monitor_val,verbose=True):
        
        self.scheduler = None
        if not (optimizer is None):
            self.scheduler = ReduceLROnPlateau(optimizer,
                                            factor=factor,
                                            patience=patience,
                                            mode = mode,
                                            verbose=verbose)
        print("LR schedular included....")
        self.monitor_val = monitor_val
        self.best_val_loss = 1000.0
        self.counter = 0


    def on_epoch_end(self, epoch, logs):

        if not (self.scheduler is None):
            self.scheduler.step(logs[self.monitor_val])

        

class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    '''
    def __init__(self, model, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        '''
        @param monitor: str. Quantity to monitor.
        @param patience: number of epochs with no improvement after which training will be stopped.
        @param verbose: verbosity mode, 0 or 1.
        @param mode: one of {auto, min, max}. Decides if the monitored quantity improves. If set to `max`, increase of the quantity indicates improvement, and vice versa. If set to 'auto', behaves like 'max' if `monitor` contains substring 'acc'. Otherwise, behaves like 'min'.
        '''
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.model = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))

                self.model.stop_training = True
            self.wait += 1
