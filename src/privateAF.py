from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
from src.modules import PlayerNet

torch.autograd.set_detect_anomaly(True)

class PrivateGAF(nn.Module):
    def __init__(self, args, iagent):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(PrivateGAF, self).__init__()
        self.use_gpu = args.use_gpu
        self.nfeatures = args.nfeatures
        self.nplayers = args.nagents
        self.M = args.M

        self.use_cbm = args.use_cb_multiplicity

        if self.use_cbm:
            # current knowledge base
            self.current_KB = nn.Sequential(nn.Linear(args.embedding_dim, args.embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.embedding_dim, args.rnn_hidden))
            if args.use_gpu:
                self.current_KB.cuda()
        self.ram_net = PlayerNet(args)

        self.name = 'Exp-{}-Agent:{}'.format(args.name, iagent)

        self.__init_optimizer(args.player_lr)
        self.classifier = self.ram_net.classifier

        self.update_parameters = not (args.random_player and (iagent != 0))


    def init_argument(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        arg_0 = torch.rand(batch_size, self.nfeatures)
        arg_0 = Variable(arg_0).type(dtype)
        arg_0 = 1.0*F.one_hot(torch.argmax(arg_0, 1), num_classes=arg_0.shape[-1])
        return arg_0


    def init_rnn_hidden(self, batch_size, zi):
        h0 = self.ram_net.rnn.init_hidden(batch_size)
        if self.use_cbm:
            # zi: B x h x s x s
            z = F.adaptive_avg_pool2d(zi , (1, 1)).squeeze()
            h = self.current_KB(z)
            h0 = (h, h0[1])
        return h0


    def __init_optimizer(self, lr=1e-3, weight_decay = 1e-5):
        print(f"Agent:{self.name}, LearningRate: {lr}")
        self.optimizer = torch.optim.Adam (self.parameters(), 
                            lr=lr)


    def forwardStep(self, *args):
        return self.ram_net.step(*args)


    def stepArgument(self, current,
                            opponent = None,
                            y = None, 
                            claim = None,
                            arg_history = None, 
                            h_t = None,
                            htrain = True):

        with torch.no_grad():
            initial_strength = self.classifier(h_t[0]) # logits

        aux_info = self.ram_net.step(current, opponent,
                                    arg_history, 
                                    claim, h_t, 
                                    htrain=htrain)
        with torch.no_grad():
            step_claim = self.classifier(aux_info[0][0]) # logits

        argument_strength = F.softmax(step_claim.clone().detach()) - F.softmax(initial_strength)
        
        return aux_info, step_claim, torch.cat([argument_strength[i][y[i]].unsqueeze(0) for i in range(y.shape[0])], 0)
    

    @torch.no_grad()
    def get_base_scores(self, z, sidxs, y, claim, hm1):
        scores = []
        relative_score = self.classifier(hm1[0]).clone() 
        batch_size, _, nfeatures1, nfeatures2      = z.shape
        
        nfeatures = nfeatures1*nfeatures2

        for _if_ in range(nfeatures):
            argidx          = torch.zeros(batch_size, nfeatures, dtype=z.dtype, device=z.device) # current player argument
            history         = torch.zeros(batch_size, nfeatures, dtype=z.dtype, device=z.device) # current player argument
            argidxop        = torch.zeros(self.nplayers -1, batch_size, nfeatures, dtype=z.dtype, device=z.device) # empty opponent argument
            argidx[:, _if_] = 1.0

            (h0, *_), _, _  = self.stepArgument((z, sidxs, argidx),
                                                ([z]*(self.nplayers -1), sidxs, argidxop), # repeating as opponents arguments to prevent information drop
                                                y,
                                                claim,
                                                history, 
                                                hm1,
                                                htrain=False)

            bscore  = F.softmax(self.classifier(h0[0])) - F.softmax(relative_score) 
            scores.append(torch.cat([bscore[i][y[i]].unsqueeze(0) for i in range(y.shape[0])], dim = 0).unsqueeze(-1))
        return torch.cat(scores, dim = 1) # B x nfeatures


    def get_combined_context(self, z, sidxs, y, claim, hm1):
        batch_size, _, nfeatures1, nfeatures2  = z.shape
        
        nfeatures = nfeatures1*nfeatures2

        for _if_ in range(nfeatures):
            argidx          = torch.zeros(batch_size, nfeatures, dtype=z.dtype, device=z.device) # current player argument
            history         = torch.zeros(batch_size, nfeatures, dtype=z.dtype, device=z.device) # current player argument
            argidxop        = torch.zeros(self.nplayers -1, batch_size, nfeatures, dtype=z.dtype, device=z.device) # empty opponent argument
            argidx[:, _if_] = 1.0

            (hm1, *_), _, _  = self.stepArgument((z, sidxs, argidx),
                                                ([z]*(self.nplayers -1), sidxs, argidxop), # repeating as opponents arguments to prevent information drop
                                                y,
                                                claim,
                                                history, 
                                                hm1,
                                                htrain = True)

        return hm1


    def optStep(self, loss):
        if self.update_parameters:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

