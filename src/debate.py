from argparse import Action
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import copy
import torch.nn as nn

import os, json
from src.clsmodel import afhq, mnist, ffhq, shapes
from src.modules import QClassifier
from src.vqcbm import VectorQuantizer as VectorQuantizerCBM
from src.vq import VectorQuantizer


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model
    

class Debate(nn.Module):
    def __init__(self, private_GAF, shared_BAF, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(Debate, self).__init__()

        self.nagents = args.nagents
        assert self.nagents < (args.n_class + 1), 'number of agents should be lower than number of classes.'


        self.M = args.M
        self.narguments = args.narguments

        self.agents = private_GAF
        self.shared_BAF = shared_BAF

        self.use_gpu = args.use_gpu
        self.rl_weightage = args.rl_weightage

        self.name = 'VX:{}_{}_{}_{}_{}'.format(
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.num_embeddings,
                                        args.rnn_hidden,
                                        args.rnn_input_size)

        self.use_mean_dc = args.use_mean_dc
        self.quantize = args.quantize


        # load feature extractor ====
        if args.data_root.lower().__contains__('mnist'):
            self.judge = mnist(pretrained=True, 
                                feature_extractor=args.feature_extractor,
                                case = args.case)
        
        elif args.data_root.lower().__contains__('shapes'):
            self.judge = shapes(pretrained=True, 
                                feature_extractor=args.feature_extractor,
                                case = args.case)
        
        elif args.data_root.lower().__contains__('ffhq'):
            self.judge = ffhq(pretrained=True, 
                                feature_extractor=args.feature_extractor,
                                case = args.case)
        
        elif args.data_root.lower().__contains__('afhq'):
            self.judge = afhq(pretrained=True, 
                                feature_extractor=args.feature_extractor,
                                case = args.case)
        else:
            raise ValueError('Unknown dataset found')



        self.judge.eval()
        if self.use_gpu: self.judge = self.judge.cuda()

        print ("Judge Netwrok loaded...")
        print(vars(args))

        # save config copy ====
        json.dump(vars(args), 
                open(os.path.join(args.ckpt_dir, 
                        'config.json'), 'w'), 
                indent=4)



        # grad setting...
        set_requires_grad(self.judge, False)

        # common model def.============================
        if args.use_cb_multiplicity:
            self.discretizer = VectorQuantizerCBM(args)
        else:
            self.discretizer = VectorQuantizer(args)


        self.discretizer.train()
        self.quantized_classifier = QClassifier(args.embedding_dim, 
                                                    args.n_class)


        self.hprojection = nn.Sequential(nn.Linear(args.rnn_hidden, args.rnn_hidden),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.rnn_hidden, args.embedding_dim))


        # ==============================================
        self.quantized_optimizer = torch.optim.Adam (list(self.discretizer.parameters()) + \
                                                    list(self.hprojection.parameters()) + \
                                                    list(self.quantized_classifier.parameters()), 
                                                    lr=args.debate_lr)


    def fext(self, x, y):
        z_conti = self.judge.features(x)
        zq, symbol_idx, loss, perplexity = self.discretizer(z_conti, y)
        
        return z_conti, zq, symbol_idx, loss, perplexity 

    @torch.no_grad()
    def get_debate_argument_set(self, x, topk_predictions):
        return self.discretizer.debate_arguments(self.judge.features(x), topk_predictions)


    def get_initial_conditions(self, z, symbol_idxs, agent_claims, y):

        batch_size = z[0].shape[0]
        hs_m1      = [ agent.init_rnn_hidden(batch_size, z[ai]) \
                                for ai, agent in enumerate(self.agents) ]


        # compute base scores
        # base scores are agents confidence for all the features wrt the ground-truth class
        # Np x B x nfeature: [0, 1]^{Np x B x nfeatures}
        base_scores = [ agent.get_base_scores(z[ai], \
                                                symbol_idxs[ai], 
                                                y, agent_claims[0], 
                                                hs_m1[ai]) \
                                for ai, agent in enumerate(self.agents) ]

        self.shared_BAF.argument_base_scores = base_scores

        # compute hs0
        # context information with all the aggregated arguments
        hs = [ agent.get_combined_context(z[ai], \
                                        symbol_idxs[ai], 
                                        y, agent_claims[0], 
                                        hs_m1[ai]) \
                                for ai, agent in enumerate(self.agents) ]
        
        return hs
        

    def step(self, z, symbol_idxs, agent_claims, y, topk_predictions, test = False):
        batch_size = z[0].shape[0]
        
        
        # init lists for location, hidden vector, baseline and log_pi
        # dimensions:    (narguments + 1, nagents, *)

        args_idx_t  = [[ agent.init_argument(batch_size) for agent in self.agents ] for _ in range(self.narguments + 1)]
        arg_dists_t = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        logpis_t    = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        bs_t        = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
       

        # context initialization
        hs = self.get_initial_conditions(z, symbol_idxs, agent_claims, y)


        # update classifier and rnn parameters
        initial_claims = []
        for i, agent in enumerate(self.agents):
            
            log_prob_agent = F.log_softmax(agent.classifier(hs[i][0]), dim = 1)
            if not test:
                agent.optStep(F.nll_loss(log_prob_agent, topk_predictions[:, i]))

            initial_claims.append(torch.argmax(log_prob_agent.clone().detach(), 1))


        # ==================
        # debate process to update policy network
        # we use the trained hidden state vectors to perform this operation

        hs = [(h_[0].detach().clone(), h_[1]) for h_ in hs]
        hinitals = [(h_[0].detach().clone(), h_[1]) for h_ in hs]
        argument_index_history = torch.zeros_like(args_idx_t[0][0])

        for t in range(1, self.narguments + 1):
            args_idx_t_const = [_arg_.clone() for _arg_ in args_idx_t[t-1]]    

            for ai, agent in enumerate(self.agents):
                # argument history for conditioning sampling, current agent
                argument_index_history += args_idx_t_const[ai]

                other_agents = list(range(len(self.agents)))
                other_agents.pop(ai)
                args_t_opponent = []
                sampled_idxs_opponent = []
                sampled_features_opponent = []
                

                for oi in other_agents:
                    args_t_opponent.append(args_idx_t_const[oi])
                    sampled_idxs_opponent.append(symbol_idxs[oi])    
                    sampled_features_opponent.append(z[oi].clone().detach())

                    # argument history for conditioning sampling, opponent agent
                    # we need to condition based on opponents argument in case of 
                    # single codebook usage

                    if isinstance(self.discretizer, VectorQuantizer):
                        argument_index_history += args_idx_t_const[oi]

                
                hstm1_ai = (hs[ai][0].clone().detach(), hs[ai][1])
                (hs[ai], arg_idx_t, b_t, log_pi, dist), claims, argument_strength = agent.stepArgument(
                                                                            (z[ai], symbol_idxs[ai], args_idx_t_const[ai]),
                                                                            (sampled_features_opponent, sampled_idxs_opponent, args_t_opponent),
                                                                            y, agent_claims[0],
                                                                            argument_index_history,
                                                                            hs[ai],
                                                                            htrain = True)

                # compute percieved claims
                cross_claims = []
                for aj, agentj in enumerate(self.agents):
                    if ai == aj:
                        cross_claims.append(torch.zeros_like(claims)) 
                        continue
                    
                    with torch.no_grad():
                        sc  = agentj.classifier(hs[ai][0])
                        scp = agentj.classifier(hstm1_ai[0])
                        cross_claims.append(F.softmax(sc - scp, dim = -1))

                    

                self.shared_BAF.update(ai, (z[ai], arg_idx_t, dist), claims, argument_strength, cross_claims)

                agent_claims[ai]     = claims
                args_idx_t[t][ai]    = arg_idx_t
                args_idx_t_const[ai] = arg_idx_t
                arg_dists_t[t][ai]   = dist
                logpis_t[t][ai]      = log_pi
                bs_t[t][ai]          = b_t

            self.shared_BAF.update_reward_trace()


        # remove first time stamp 
        bs_t        = bs_t[1:]
        logpis_t    = logpis_t[1:] 
        args_idx_t  = args_idx_t[1:]
        arg_dists_t = arg_dists_t[1:]

        return args_idx_t, arg_dists_t, bs_t, logpis_t, hs, hinitals, initial_claims


    def reformat(self, x, ai, dist=False):
        """
        @param x: data. [narguments, [nagents, (batch_size, ...)]]
        returns x (batch_size, narguments, ...)
        """
        agents_data = []
        for t in range(self.narguments):
            agents_data.append(x[t][ai].unsqueeze(0))
        agent_data = torch.cat(agents_data, dim=0)
        return torch.transpose(agent_data, 0, 1)


    def get_avgz(self, z_orig):
        # if self.quantize == 'channels':
        #     z = F.adaptive_avg_pool2d(z_orig , (1, 1)).squeeze()
        # else:
        #     z = z_orig.mean(1).view(z_orig.shape[0], -1)
        return z_orig


    @torch.no_grad()
    def get_debate_outcome(self, ht):
        dlogits = 0
        # TODO: try softmax average/ convex combination 
        for iht in ht:
            dlogits += iht[0]

        # predict debate outcome given all arguments
        # here we use the heuristic of <log_prob(agent_classifier(<agent_hidden_state>))>
        # =========================================
        dlogits *= 1.0/self.nagents

        hzq = self.hprojection(dlogits)
        dlog_prob = self.quantized_classifier(hzq)

        dpred = torch.argmax(dlog_prob, 1) 
        return dpred


    def forward(self, x, y, lts, is_training=True, epoch=1):
        """
        @param x: image. (batch, channel, height, width)
        @param y: word indices. (batch, seq_len)
        """
        if self.use_gpu:
            x = x.cuda(); y = y.cuda()
        x = Variable(x); y = Variable(y)

        if not is_training:
            return self.forward_test(x, y, epoch)


        # =====================================
        # Judge prediction
        with torch.no_grad():
            jpred_probs = self.judge(x).detach()
            jpred = torch.argmax(jpred_probs, 1)

        
        # quantized distillation
        self.quantized_optimizer.zero_grad()
        z_orig, zq, symbol_idxs, qloss, perplexity = self.fext(x, jpred)
        z = F.adaptive_avg_pool2d(zq , (1, 1)).squeeze()
        

        cqlog_probs = self.quantized_classifier(z)
        cq_loss = F.nll_loss(cqlog_probs, jpred)
        cq_loss_ = cq_loss + qloss

        
        cqlog_probs = cqlog_probs.clone().detach()
        topk_predictions = torch.topk(cqlog_probs, self.nagents)[1]


        # shared BAF reset
        self.shared_BAF.reset()
        self.shared_BAF.class_base_score = torch.max(torch.softmax(cqlog_probs, -1), -1)[0]
        
        # =========================================

        zs_orig, symbol_idxss = self.get_debate_argument_set(x, topk_predictions)
        zs = [self.get_avgz(zs_) for zs_ in zs_orig]
        args_idx_t, arg_dists_t, b_ts, log_pis, h_t, h_0, initial_claims = self.step(zs, 
                                                                symbol_idxss, 
                                                                [jpred_probs for _ in range(self.nagents)],
                                                                jpred,
                                                                topk_predictions)



        # =========================================
        # total distillation loss
        for iagent in range(self.nagents):
            hzq = self.hprojection(h_t[iagent][0])
            
            with torch.no_grad():
                self.quantized_classifier.eval()
                cqhprob = self.quantized_classifier(hzq)

            pihloss = F.nll_loss(cqhprob, topk_predictions[:, iagent])
            cq_loss_ += pihloss

        cq_loss_.backward(retain_graph=True)
        self.quantized_optimizer.step()
        self.quantized_classifier.train()

        # all elements for zerosum-component:
        arguments = []; log_prob_agents = []; 
        final_claims = []

        for ai, agent in enumerate(self.agents):
            argument  = self.reformat(args_idx_t, ai)
            arguments.append(argument)

            log_prob_agent = F.log_softmax(agent.classifier(h_0[ai][0]), dim =-1)
            log_prob_agents.append(log_prob_agent)

            with torch.no_grad(): 
                final_claims.append(torch.argmax(log_prob_agent, 1))
          


        # update initial claims =========
        self.shared_BAF.initial_claims   = torch.cat([c.unsqueeze(-1) for c in initial_claims], dim = -1)


        dpred = self.get_debate_outcome(h_t)

        # =========================================
        logs = {}


        # individual agent optimizer
        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            log_pi = self.reformat(log_pis, ai)
            args_dist = self.reformat(arg_dists_t, ai, True)


            # Classifier Loss ---> distillation loss
            # loss_classifier = 0
            loss_classifier = F.nll_loss(log_prob_agents[ai], 
                                            topk_predictions[:, ai])


            # Baseline Loss
            # reward:          (batch, num_glimpses)
            reward = self.shared_BAF.get_reward(ai, dpred, jpred) 
            loss_baseline = F.mse_loss(baselines, reward)


            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            # loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            loss = (loss_reinforce + loss_baseline) + loss_classifier


            agent.optStep(loss)


            # Logs record
            logs[f'rlloss_agent-{ai}'] = loss
            logs[f'final_claims_agent-{ai}'] = final_claims[ai]
            logs[f'initial_claims_agent-{ai}'] = initial_claims[ai]
            logs[f'z_idx_agent-{ai}'] = symbol_idxss[ai]
            logs[f'argument_trace_agent-{ai}'] = torch.cat(self.shared_BAF.argument_index_trace[ai], 0)
            logs[f'reward_agent-{ai}'] = torch.mean(reward)

        logs.update(self.shared_BAF.metrics(dpred, y, jpred, topk_predictions[:, 0]))

        
        logs['x'] = x
        logs['y'] = y
        logs['z'] = z_orig
        logs['jpred'] = jpred
        logs['dpred'] = dpred
        logs['cqloss'] = cq_loss
        logs['perplexity'] = perplexity
        return logs


    @torch.no_grad()
    def forward_test(self, x_orig, y, epoch):

        # duplicate M times
        x = x_orig.clone().repeat(self.M, 1, 1, 1)
        
        # Main forwarding step
        # locs:             (batch*M, 2)*num_glimpses
        # baselines:        (batch*M, num_glimpses)
        # log_pi:           (batch*M, num_glimpses)
        # log_probas:       (batch*M, num_class)


        # ======================================= prelim........
        # Judge prediction
        jpred_probs_M = self.judge(x).detach()
        jpred_M = torch.argmax(jpred_probs_M, 1)
        jpred_probs = jpred_probs_M.contiguous().view((self.M, x_orig.shape[0]) + jpred_probs_M.shape[1:])
        jpred_probs = torch.mean(jpred_probs, 0)
        jpred = torch.argmax(jpred_probs, 1)
        

        # quantized predictions
        z_orig_M, zq_M, symbol_idxs_M, qloss, perplexity = self.fext(x, jpred_M)
        z_orig = z_orig_M.contiguous().view((self.M, x_orig.shape[0]) + z_orig_M.shape[1:])
        z_orig = z_orig[0]

        zq = zq_M.contiguous().view((self.M, x_orig.shape[0]) + zq_M.shape[1:])
        zq = zq[0]
        z_M = F.adaptive_avg_pool2d(zq_M , (1, 1)).squeeze()
        
        cqlog_probs_M = self.quantized_classifier(z_M)
        cqlog_probs = cqlog_probs_M.contiguous().view((self.M, x_orig.shape[0]) + cqlog_probs_M.shape[1:])
        cqlog_probs = torch.mean(cqlog_probs, 0)

        cq_loss = F.nll_loss(cqlog_probs_M, jpred_M)
        cq_loss = cq_loss + qloss


        topk_predictions_M = torch.topk(cqlog_probs_M, self.nagents)[1]
        topk_predictions   = torch.topk(cqlog_probs, self.nagents)[1]

        # shared BAF reset
        self.shared_BAF.reset()
        self.shared_BAF.class_base_score = torch.max(torch.softmax(cqlog_probs_M, -1), -1)[0]

        # =========================================
        zs_M, symbol_idxss_M = self.get_debate_argument_set(x, topk_predictions_M)
        zs = [self.get_avgz(zs_) for zs_ in zs_M]
        args_idx_t_M, arg_dists_t_M, b_ts_M, log_pis_M, h_t_M, h_0_M, initial_claims_M = self.step(zs, 
                                                                        symbol_idxss_M, 
                                                                        [jpred_probs_M for _ in range(self.nagents)],
                                                                        jpred_M,
                                                                        topk_predictions_M,
                                                                        test=True)


        
        # =========================================
        # total distillation loss
        for iagent in range(self.nagents):
            hzq = self.hprojection(h_t_M[iagent][0])
            cqhprob = self.quantized_classifier(hzq)
            pihloss = F.nll_loss(cqhprob, topk_predictions_M[:, iagent])
            cq_loss += pihloss


        # all elements for zerosum-component:

        arguments = []; log_prob_agents = []; 
        initial_claims = []; final_claims = []

        for ai, agent in enumerate(self.agents):
            argument_ = self.reformat(args_idx_t_M, ai)
            _argument_ = argument_.contiguous().view((self.M, x_orig.shape[0]) + argument_.shape[1:])
            _argument_ = torch.clip(torch.sum(_argument_, dim = 0), 0, 1)

            initial_claims_ = initial_claims_M[ai].contiguous().view((self.M, x_orig.shape[0]) + initial_claims_M[ai].shape[1:])
            initial_claims.append(initial_claims_[0])

            log_prob_agentf = agent.classifier(h_0_M[ai][0])
            log_prob_agentf = log_prob_agentf.contiguous().view((self.M, x_orig.shape[0]) + log_prob_agentf.shape[1:])
            log_prob_agentf = torch.mean(log_prob_agentf, dim=0)
            log_prob_agents.append(log_prob_agentf)

            final_claims.append(torch.argmax(log_prob_agentf, 1))
            arguments.append(_argument_)  

        # update initial claims =========
        self.shared_BAF.initial_claims   = torch.cat([c.unsqueeze(-1) for c in initial_claims], dim = -1)


        dpred = self.get_debate_outcome(h_t_M)

        # =========================================
        logs = {}

        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts_M, ai)
            log_pi = self.reformat(log_pis_M, ai)
            args_dist = self.reformat(arg_dists_t_M, ai, True)


            # Average     
            baselines = baselines.contiguous().view((self.M, x_orig.shape[0]) + baselines.shape[1:])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view((self.M, x_orig.shape[0]) + log_pi.shape[1:])
            log_pi = torch.mean(log_pi, dim=0)


            # classifier loss -> distillation
            # loss_classifier = 0
            loss_classifier = F.nll_loss(log_prob_agents[ai], topk_predictions[:, ai])


            # Prediction Loss & Reward
            # preds:    (batch)
            # reward:   (batch)
            reward_M = self.shared_BAF.get_reward(ai, dpred, jpred_M) 
            reward = reward_M.contiguous().view((self.M, x_orig.shape[0]) + reward_M.shape[1:])
            reward = torch.mean(reward, dim=0)
            loss_baseline = F.mse_loss(baselines, reward)


            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            # loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            loss = (loss_reinforce + loss_baseline) + 0.005*loss_classifier


            # Logs record
            logs[f'rlloss_agent-{ai}'] = loss
            logs[f'final_claims_agent-{ai}'] = final_claims[ai]
            logs[f'initial_claims_agent-{ai}'] = initial_claims[ai]
        
            symbol_idxss = symbol_idxss_M[ai].contiguous().view((self.M, x_orig.shape[0]) + symbol_idxss_M[ai].shape[1:])
            symbol_idxss = symbol_idxss[0]
            logs[f'z_idx_agent-{ai}'] = symbol_idxss
        
            # zs = zs_M[ai].contiguous().view((self.M, x_orig.shape[0]) + zs_M[ai].shape[1:])
            # zs = zs[0]
            # logs[f'z_agent-{ai}'] = zs

            argument_trace = torch.cat(self.shared_BAF.argument_index_trace[ai], 0).transpose(0, 1)
            argument_trace = argument_trace.contiguous().view((self.M, x_orig.shape[0]) + argument_trace.shape[1:])
            argument_trace = argument_trace[0].transpose(0, 1)
            logs[f'argument_trace_agent-{ai}'] = argument_trace
            logs[f'reward_agent-{ai}'] = torch.mean(reward)


        logs.update(self.shared_BAF.metrics(dpred, y, jpred, topk_predictions[:, 0]))

        logs['x'] = x
        logs['y'] = y
        logs['z'] = z_orig
        logs['jpred'] = jpred
        logs['dpred'] = dpred
        logs['cqloss'] = cq_loss
        logs['perplexity'] = perplexity
        return logs



    def load_model(self, ckpt_dir,  best=False):
        suff = '' #supportive' if not contrastive else 'contrastive'
        if best:
            print ('model name',self.name)
            path = os.path.join(ckpt_dir, self.name + suff + '_ckpt_best')
        else:
            path = os.path.join(ckpt_dir, self.name + suff + '_ckpt')
        
        print ('Model loaded from: {}'.format(path))
        ckpt = torch.load(path)
        for ai, agent in enumerate(self.agents):
            agent.load_state_dict(ckpt[ai]['model_state_dict'])
            agent.optimizer.load_state_dict(ckpt[ai]['optim_state_dict'])


        self.quantized_classifier.load_state_dict(ckpt['qclassifier_state_dict'])
        self.discretizer.load_state_dict(ckpt['discretizer'])
        self.quantized_optimizer.load_state_dict(ckpt['qclassifier_optim_state_dict'])
        return ckpt['epoch']


    def get_state_dict(self):
        state = {}
        for ai, agent in enumerate(self.agents):
            state[ai] = {} 
            state[ai]['model_state_dict'] = agent.state_dict()
            state[ai]['optim_state_dict'] = agent.optimizer.state_dict()

        state['qclassifier_state_dict'] = self.quantized_classifier.state_dict()
        state['discretizer'] = self.discretizer.state_dict()
        state['qclassifier_optim_state_dict'] = self.quantized_optimizer.state_dict()
        return state
