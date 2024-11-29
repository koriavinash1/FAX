import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 


def cosine(x, y):
    return F.cosine_similarity(x, y, dim = -1)

def nmi(x, y):
    pass

zipcat = lambda x,y: torch.cat([x[i][y[i]].unsqueeze(0) for i in range(y.shape[0])], dim =0)

class SharedAF(object):
    def __init__(self,
                    class_mapping: dict, 
                    nplayers: int   = 2,
                    similarity: str = 'cosine', 
                    af_semantics: str  ='dfquad',
                    threshold: float = 0.0,
                    lambda2: float = 0.0,
                    lambda3: float = 0.0,
                    lambda4: float = 0.0,
                    lambda5: float = 0.0):

        self.class_mapping = class_mapping
        self.threshold = threshold
        self.nplayers = nplayers
        self.similarity = similarity
        self.af_semantics = af_semantics

        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5

        assert self.similarity in ['cosine', 'NMI'], "Unkown similarity found, allowed types: ['cosine', 'NMI']."

        if self.similarity == 'cosine':
            self.similarity_criterion = cosine
        else:
            self.similarity_criterion = nmi


        self.reset()


    def init_AF(self, y):
        self.G = nx.DiGraph()

        self.G.add_node(f'${self.class_mapping[y]}$')
        pass 


    def check_termination(self, curernt_arguments):
        raise NotImplementedError()

    
    def update(self, agent, current_state, current_claim, argument_strength, cross_strengths):
        
        self.argument_history[agent].append(current_state)
        z = current_state[0].flatten(-2, -1)
        # argument_feature = (current_state[0]*current_state[1]).unsqueeze(0)
        argument_feature = (z*current_state[1][:, None, ...]).mean(-1).unsqueeze(0)
        self.argument_feature_trace[agent].append(argument_feature)
        self.argument_dist_trace[agent].append(current_state[2].unsqueeze(0))
        self.argument_index_trace[agent].append(torch.argmax(current_state[1], 1).unsqueeze(0))
        self.claim_trace[agent].append(torch.argmax(current_claim, 1).unsqueeze(0))
        self.strength_trace[agent].append(argument_strength.unsqueeze(0))

        for i, cs in enumerate(cross_strengths):
            self.cross_strength_trace[agent][i].append(cs)

        # ====================================
        if len(self.argument_feature_trace[agent]) > 1:
            similarity_score = self.similarity_criterion(torch.cat(self.argument_feature_trace[agent][:-1], 0),
                                                                        argument_feature)
        else:
            similarity_score = 0.0

        self.intra_similarity_score_trace[agent].append(similarity_score) # n, b


        if len(self.argument_feature_trace[agent]) > 2:
            other_agents = list(range(self.nplayers))
            other_agents.pop(agent)
            other_arguments = []
            for oi in other_agents:
                other_arguments.extend(self.argument_feature_trace[oi][:-1])
            similarity_score = self.similarity_criterion(torch.cat(other_arguments, 0), 
                                                                        argument_feature)
        else:
            similarity_score = 0.0

        current_arg_lengths = len(self.argument_history[agent][:-1])
        self.inter_similarity_score_trace[agent].append(similarity_score)  # (nagents -1)*n, b

        pass 
    
    
    def get_state_t_reward(self, iagent):
        
        reward = 0

        if self.af_semantics == 'dfquad':
            # given argumentation is biased towards supporter's prediction 
            # v+ > v- by design
            
            index = self.argument_index_trace[iagent][-1].squeeze()
            base_score = torch.cat([self.argument_base_scores[iagent][i][index[i]].unsqueeze(0) for i in range(index.shape[0])], dim=0)

            st = torch.abs(self.strength_trace[iagent][-1]).squeeze()

            reward = torch.zeros_like(base_score)

            reward[base_score < 0]  = (base_score + (1 - base_score)*st)[base_score < 0]
            reward[base_score >= 0] = (base_score - base_score*st)[base_score >= 0]
            
            reward = reward.squeeze(0)

        return reward
         


    def update_reward_trace(self):
        # ====================================
        # compute reward trace

        for iagent in range(self.nplayers):
            self.reward_trace[iagent].append(self.get_state_t_reward(iagent).unsqueeze(0))
        
        pass


    def get_reward(self, iagent, dpred, jpred):
        reward = torch.cat(self.reward_trace[iagent], 0).transpose(0, 1) # B x debate length x 1
        # reward[self.claim_trace[iagent][-1].squeeze(0) != dpred] *= -1 

        # IR = self.irrelevance_fn(iagent)
        # RP = self.repetability_fn(iagent)
        # PM = self.persuasion_monotonicity_fn(iagent, jpred, dpred)
        # PS = self.persuasion_strength_fn(iagent, jpred, dpred)
        
        # reward = reward - self.lambda2 *IR - self.lambda3*RP + self.lambda4*PM + self.lambda5*PS
        return reward


    def reset(self):
        self.argument_history = {ai: [] for ai in range(self.nplayers)}
        self.argument_index_trace = {ai: [] for ai in range(self.nplayers)}
        self.argument_feature_trace = {ai: [] for ai in range(self.nplayers)}
        self.claim_trace = {ai: [] for ai in range(self.nplayers)}
        self.strength_trace = {ai: [] for ai in range(self.nplayers)}

        self.cross_strength_trace = {ai: {aj: [] for aj in range(self.nplayers)} for ai in range(self.nplayers)}
        self.reward_trace = {ai: [] for ai in range(self.nplayers)}
        self.argument_dist_trace = {ai: [] for ai in range(self.nplayers)}
        self.intra_similarity_score_trace = {ai: [] for ai in range(self.nplayers)}
        self.inter_similarity_score_trace = {ai: [] for ai in range(self.nplayers)}

        # self.intra_similarity_idx_trace = {ai: [] for ai in range(self.nplayers)}
        # self.inter_similarity_idx_trace = {ai: [] for ai in range(self.nplayers)}
    

    def accuracy(self, x, y):
        correct = (x == y).float()
        acc = 100 * (correct.sum() / len(y))
        return acc 

    def for_resolution_representation(self, y):
        # check in the stance of opponent is equal to gt
        frr = 0
        nagents = len(self.claim_trace)
        narguments = len(self.claim_trace[0])
        y = y.unsqueeze(0).repeat(narguments, 1)

        for_final_stance = self.claim_trace[0][0]
        

        for iagent in range(1, nagents):
            against_final_stance = self.claim_trace[iagent][-1] 
            resolution = (for_final_stance == against_final_stance)

            con_stance = self.claim_trace[iagent]
            # con_stance: B x narguments
            if torch.sum(1.0*resolution) > 0:
                frr += (1.0*(con_stance == y)).sum(0)[resolution].mean()
            else:
                frr += 1
        frr /= (nagents - 1)
        return frr

    def against_resolution_representation(self, y):
        # check in the stance of pro player not equal to gt
        arr = 0
        nagents = len(self.claim_trace)
        narguments = len(self.claim_trace[0])
        y = y.unsqueeze(0).repeat(narguments, 1)
        for_stance = self.claim_trace[0]
        

        for_final_stance = self.claim_trace[0][-1]
        

        for iagent in range(1, nagents):
            against_final_stance = self.claim_trace[iagent][0] 
            resolution = (for_final_stance == against_final_stance)

            if torch.sum(1.0*resolution) > 0:
                arr += (1.0*(for_stance != y)).sum(0)[resolution].mean()
            else:
                arr += 1

        arr /= (nagents - 1)
        return arr


    def irrelevance_fn(self, iagent):
        if not isinstance(self.inter_similarity_score_trace[iagent][-1], float):
            return torch.mean(self.inter_similarity_score_trace[iagent][-1].sum(0))
        return 0.0


    def repetability_fn(self, iagent):
        if not isinstance(self.intra_similarity_score_trace[iagent][-1], float):
            return torch.mean(self.intra_similarity_score_trace[iagent][-1].sum(0))
        return 0.0


    def persuasion_monotonicity_fn(self, iagent, jpred, dpred=None):
        # dpred only used in case of rewards
        iagent_persuasion_monotonicity = 0.0
        own_strength = self.strength_trace[iagent][-1]
        for jagent in range(0, self.nplayers):
            if iagent == jagent: continue

            percieved_st = zipcat(self.cross_strength_trace[iagent][jagent][-1], jpred)

            st = 1.0*((own_strength >= 0.0)*(percieved_st >= 0.0)) +\
                 1.0*((own_strength < 0.0)*(percieved_st < 0.0))

            st = st.squeeze()
            # if not (dpred is None):
            #     st[self.initial_claims[:, iagent] != dpred] *= -1

            iagent_persuasion_monotonicity += torch.mean(st)
    
        iagent_persuasion_monotonicity /= (self.nplayers - 1)
        return iagent_persuasion_monotonicity


    def persuasion_strength_fn(self, iagent, jpred, dpred=None):
        iagent_persuasion_strength = 0.0
        for jagent in range(0, self.nplayers):
            if iagent == jagent: continue
            
            st = zipcat(self.cross_strength_trace[iagent][jagent][-1], jpred)
            st = st.squeeze()
            if not (dpred is None):
                st[self.initial_claims[:, iagent] != dpred] *= -1

            iagent_persuasion_strength += torch.mean(st)
    
        iagent_persuasion_strength /= (self.nplayers - 1)
        return iagent_persuasion_strength



    @torch.no_grad()
    def metrics(self, dpred, ygt, jpred, cqpred):
        # correctness: measure the correctness of considered classifier
        #              accuracy of classifier wrt gt label
        # faithfulness: measure how faithful the debate is to the considered classifier
        #              accuracy of debate outcome wrt classifier prediction
        # reliability: measure reliability of the debate framework
        #              accuracy of debate outcome wrt gt label
        # consensus rate: measures how quickly consensus was reached
        # persuasion_strength : change final argument strength wrt initial argument strength 
        # repetability : intra argument trace similarity
        # irrelevance : inter argument trace similarity

        for agent in range(self.nplayers):
            self.argument_dist_trace[agent] = torch.cat(self.argument_dist_trace[agent], 0)
            self.strength_trace[agent] = torch.cat(self.strength_trace[agent], 0)
            self.claim_trace[agent] = torch.cat(self.claim_trace[agent], 0)


        trace_length = self.claim_trace[0].shape[0]

        # =====================================================================
        # P2: average consensus
        agent_0_final_claims = self.claim_trace[0][trace_length - 1]
        consensus = 0
        for iagent in range(1, self.nplayers):
            consensus += self.accuracy(agent_0_final_claims,
                                self.claim_trace[iagent][trace_length - 1])
        consensus /= (self.nplayers -1)


        # =====================================================================
        # P3: consensus rate
        # earlier consensus implies higher cumulative sum
        agent_0_claim_trace = self.claim_trace[0].detach().cpu().numpy()
        contribution_rate = 0
        for iagent in range(1, self.nplayers):
            current_agent_trace = self.claim_trace[iagent].detach().cpu().numpy()
            # NOTE: np.where will only consider resolved AXICS
            matched_idx = np.where(current_agent_trace == agent_0_claim_trace)[0]
            if len(matched_idx): contribution_rate += np.mean(matched_idx)

        contribution_rate /= max((self.nplayers -1), 1)

        # =====================================================================
        # intial player accuracies
        player_acc = {}
        for iagent in range(self.nplayers):
            player_acc[f'initial_acc_agent-{iagent}'] = self.accuracy(self.initial_claims[:, iagent], ygt) 
            player_acc[f'final_acc_agent-{iagent}']   = self.accuracy(self.claim_trace[iagent][trace_length - 1], ygt)


        # =====================================================================
        # P4: compute persuasion_rate
        # counts total number of successful persuasion at the end of debate
        persuasion_rate = {}
        for iagent in range(0, self.nplayers):
            ref_claims = self.claim_trace[iagent][0].detach().cpu().numpy() # initial stance
            iagent_persuasion_rate = 0

            for jagent in range(0, self.nplayers):
                if iagent == jagent: continue

                current_agent_claim = self.claim_trace[jagent][trace_length - 1].detach().cpu().numpy() # final stance
                iagent_persuasion_rate += np.mean(current_agent_claim == ref_claims)

            iagent_persuasion_rate /= max((self.nplayers -1), 1)   
            persuasion_rate[f'persuasion_rate_agent-{iagent}'] = iagent_persuasion_rate


        # =====================================================================
        # P5: compute persuasion_strength
        # average strength of all attacker agents: range(1, n)
        persuasion_strength = {}
        for iagent in range(0, self.nplayers):
            persuasion_strength[f'persuasion_strength_agent-{iagent}'] = self.persuasion_strength_fn(iagent, jpred)



        # =====================================================================
        # P6: compute persuasion_monotonicity
        # average strength of all attacker agents: range(1, n)
        persuasion_monotonicity = {}
        for iagent in range(0, self.nplayers):
            persuasion_monotonicity[f'persuasion_monotonicity_agent-{iagent}'] = self.persuasion_monotonicity_fn(iagent, jpred)

        # =====================================================================
        # P7 & P8: repetability and irrelevance
        repetability = {}; irrelevance = {}
        for iagent in range(self.nplayers):
            repetability[f'repetability_agent-{iagent}'] = self.repetability_fn(iagent)
            irrelevance[f'irrelevance_agent-{iagent}'] = self.irrelevance_fn(iagent)



        # =====================================================================
        results = {'correctness'     : self.accuracy(cqpred, ygt),
                    'faithfulness'   : self.accuracy(cqpred, jpred), 
                    'reliability'    : self.accuracy(dpred, ygt),
                    'data_accuracy'  : self.accuracy(jpred, ygt),
                    'consensus'      : consensus,
                    'contribution_rate' : contribution_rate,
                    'arr'            : self.for_resolution_representation(jpred),
                    'frr'            : self.against_resolution_representation(jpred)
                    }

        results.update(player_acc)
        results.update(repetability)
        results.update(irrelevance)
        results.update(persuasion_monotonicity)
        results.update(persuasion_strength)
        results.update(persuasion_rate)
        
        return results
