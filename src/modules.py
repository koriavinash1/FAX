import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

from einops import rearrange

from src.hpenalty import hessian_penalty
from torch.distributions import Categorical
from src.utils import get_euclidian_distance, get_cosine_distance


class RevLSTMCell(nn.Module):
    """ Defining Network Completely along with gradients to Variables """
    def __init__(self, input_size, hidden_size):
        super(RevLSTMCell, self).__init__()
        self.f_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.i_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uc_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.u_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.r_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uh_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size
        self.init_weights()

        
    def forward(self, x, state):
        c, h = state
        #import pdb; pdb.set_trace()
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        new_c = (f*c + i*c_)/2
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        new_h = (r*h + u*h_)/2
        return new_h, (new_c, new_h)
    
    def reconstruct(self, x, state):
        new_c, new_h = state
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        h = (2*new_h - u*h_)/(r+1e-64)
        
        
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        c = (2*new_c - i*c_)/(f+1e-64)
        
        return h, (c, h)
    
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.ndimension() == 2:
                nn.init.xavier_uniform(parameter, gain=0.01)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.hidden_size).zero_()),
                Variable(weight.new(bsz, self.hidden_size).zero_()))


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - rnn_hidden: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, rnn_hidden). The glimpse
      representation returned by the glimpse network for the current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, rnn_hidden). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, rnn_hidden). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, 
                    rnn_hidden, 
                    use_gpu, 
                    rnn_type='RNN'):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.rnn_hidden = rnn_hidden
        self.use_gpu = use_gpu
        self.rnn_type = rnn_type
        if rnn_type=='RNN':
            self.rnn = nn.RNNCell(input_size, rnn_hidden, bias=True, nonlinearity='relu')
        if rnn_type=='LSTM':
            self.rnn = nn.LSTMCell(input_size, rnn_hidden, bias=True)
        if rnn_type=='GRU':
            self.rnn = nn.GRUCell(input_size, rnn_hidden, bias=True)
        if rnn_type=='REV':
            self.rnn = RevLSTMCell(input_size, rnn_hidden)

        if use_gpu:
            self.cuda()


    def forward(self, g_t, h_t_prev):
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            h = self.rnn(g_t, h_t_prev[0])
            h_t = (h, 0)
        if self.rnn_type == 'LSTM' or self.rnn_type == 'REV':
            h_t = self.rnn(g_t, h_t_prev)
        return h_t

    def init_hidden(self, batch_size, use_gpu=False):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            h = torch.zeros(batch_size, self.rnn_hidden)
            h = Variable(h).type(dtype)
            h_t = (h, 0)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'REV':
            h = torch.zeros(batch_size, self.rnn_hidden)
            h = Variable(h).type(dtype)
            c = torch.zeros(batch_size, self.rnn_hidden)
            c = Variable(c).type(dtype)
            h_t = (h, c)
        return h_t



class PlayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(PlayerClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)
        # self.fc1 = nn.Linear(input_size, output_size)
        # self.fc2 = nn.Linear(2*output_size, output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        #==================
        h_t = F.relu(self.fc(h_t))

        return h_t



class PolicyNet(nn.Module):
    def __init__(self, concept_size, 
                        input_size, 
                        output_size, 
                        hidden_size, 
                        nclasses, 
                        use_gpu,
                        temperature=1):
        """
        @param input_size: total number of sampled symbols from a codebook.
        @param concept_size: dimension of an individual sampled symbol.
        @param hidden_size: hidden unit size of core recurrent/GRU network.
        @param output_size: output dimension of core recurrent/GRU network.
        @param std: standard deviation of the normal distribution.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size//2)
        self.fc2 = nn.Linear(input_size, output_size//2)
        self.fc3 = nn.Linear (hidden_size, output_size//2)
        self.fc4 = nn.Linear(nclasses, output_size//2)

        self.combine = nn.Linear(2*output_size, output_size)

        self.temperature = temperature
        if use_gpu:
            self.cuda()

    def forward(self, current, opponent, y, h):  
        
        arg1 = (current[0].flatten(-2, -1)*current[2][:, None, ...]).mean(-1)
        arg1_info = F.relu(self.fc1(arg1))
        decision_info = F.relu(self.fc4(y))

        arg2 = torch.zeros_like(arg1)
        if not (opponent == None):
            nopponents = len(opponent[0])
            for oi in range(nopponents):
                arg2 += (opponent[0][oi].flatten(-2, -1)*opponent[2][oi][:, None, ...]).mean(-1)
            arg2 /=nopponents

        arg2_info = F.relu(self.fc2(arg2))
        history = F.relu(self.fc3(h))

        if np.random.uniform() > 0.5:
            combination = torch.cat((arg1_info, 
                                            arg2_info,
                                            decision_info, 
                                            history), dim=1)
        else:
            combination = torch.cat((arg2_info, 
                                            arg1_info,
                                            decision_info, 
                                            history), dim=1)
        logits = self.combine(combination)

        return F.softmax(logits/self.temperature, -1)



class ModulatorNet(nn.Module):
    def __init__(self, input_size, output_size, use_gpu):
        """
        @param concept_size: dimension of an individual sampled symbol.
        @param output_size: output size of the fc layer, hidden vector dimension.
        """
        super(ModulatorNet, self).__init__()
        self.fc = nn.Linear (input_size, output_size)

        self.final = nn.Linear (output_size, output_size)

        if use_gpu:
            self.cuda()

    def forward(self, z, arg):

        arg =  (z*arg[:, None, ...]).mean(-1)
        arg_info = F.relu(self.fc(arg))

        return F.relu(self.final(arg_info))


class BaselineNet(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    @param input_size: input size of the fc layer.
    @param output_size: output size of the fc layer.
    @param h_t: the hidden state vector of the core network
                for the current time step `t`.

    Returns
    -------
    @param b_t: a 2D vector of shape (B, 1). The baseline
                for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(BaselineNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(F.relu(h_t))
        return b_t


class QClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(QClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        h_t = F.relu(h_t)
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t




class PlayerNet(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(PlayerNet, self).__init__()
        self.narguments = args.narguments

        self.quantize = args.quantize
        self.rnn = core_network(args.rnn_input_size, 
                                    args.rnn_hidden, 
                                    args.use_gpu)
        self.modulator_net = ModulatorNet(input_size = args.embedding_dim, 
                                        output_size = args.rnn_input_size,
                                        use_gpu = args.use_gpu)
        self.policy_net = PolicyNet(concept_size = args.embedding_dim, 
                                    input_size = args.embedding_dim, 
                                    output_size = args.embedding_dim if args.quantize == 'channels' else args.nfeatures, 
                                    hidden_size = args.rnn_hidden,
                                    nclasses = args.n_class,
                                    use_gpu = args.use_gpu,
                                    temperature=args.softmax_temperature)

        self.nagents = args.nagents
        self.classifier = PlayerClassifier(args.embedding_dim, 
                                            args.rnn_hidden, 
                                            args.n_class)
        self.baseline_net = BaselineNet(args.rnn_hidden, 1)


        if args.use_gpu:
            self.cuda()


    def distribtuion_conditioning(self, arg_history_onehot, current_logits):
        # distribution conditioning
        # arg_history_onehot: B, Nactions

        argument_prob_ = current_logits.clone().detach()
        B, Nactions = arg_history_onehot.shape 


        # arg_history_onehot = 0.998*arg_history_onehot
        n = self.nagents*(1+self.narguments) # max occurance of a particular argument
        # inverse probability of occurance
        conditioning_prob = n - arg_history_onehot # B, Nactions
        conditioning_prob /= n

        assert conditioning_prob.shape == argument_prob_.shape

        argument_prob_ *= conditioning_prob
        
        if argument_prob_.max() > 0:
            argument_prob_ = (argument_prob_ - argument_prob_.min(1, keepdim=True)[0])/(0.001 + argument_prob_.max(1, keepdim=True)[0] -  argument_prob_.min(1, keepdim=True)[0])
            argument_prob_ /= torch.sum(argument_prob_, 1, keepdim=True)

        argument_prob_ = torch.nan_to_num(argument_prob_, nan=1.0/argument_prob_.shape[1])
        return argument_prob_


    def step(self, current, opponent = None,
                    arg_history = None, 
                    claim = None,
                    h_t = None,
                    htrain = True):
        """
        @param current: triple of agent specific (features, symbols, one-hot-argument)
        @param opponent: triple consisting list of opponent specific ([features], [symbols], [one-hot-argument])
        @param y: predicted class label
        @param arg_history: onehot argument index history
        @param h_t: hidden state vector...
        """

        z = current[0].flatten(-2, -1)
        z_idxs = current[1]

        argument_prob = self.policy_net(current, opponent, claim, h_t[0])

        argument_prob_conditioned = self.distribtuion_conditioning(arg_history,
                                                                    argument_prob)


        try:   
            argument_dist = Categorical(probs = argument_prob_conditioned)
        except:
            print(argument_prob_conditioned)
            import pdb;pdb.set_trace()
        arg_current = argument_dist.sample()

        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        # log_pi = torch.log(torch.clip(0.0001 + argument_prob, 0.0, 1.0))
        # log_pi = log_pi.sum(dim=1)
        log_pi = argument_dist.log_prob(arg_current)


        arg_current_one_hot = 1.0*torch.zeros_like(argument_prob)
        for i, _ in enumerate(z.clone()):
            arg_current_one_hot[i, z_idxs[i] == z_idxs[i][arg_current[i]]] = 1


        # import pdb;pdb.set_trace()
        z_current = self.modulator_net(z, arg_current_one_hot)
        if not htrain:
            with torch.no_grad():
                h_t = self.rnn(z_current, h_t)
        else:
            h_t = self.rnn(z_current, h_t)
        b_t = self.baseline_net(h_t[0]).squeeze()

        return h_t, arg_current_one_hot, b_t, log_pi, argument_prob


# loss functions
# directly taken from https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
class ContrastiveLossELI5(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = torch.log(numerator / denominator)
            # loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss



class SimCLR_Loss(nn.Module):
    def __init__(self, max_batch_size=64, temperature=0.5):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(max_batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = z_i.shape[0]


        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss