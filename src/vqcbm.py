import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

# import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange
from src.hpenalty import hessian_penalty
from src.utils import unique_sampling_fn, get_euclidian_distance, sorting_idxs, get_cosine_distance, get_cb_variance
from torch.distributions import Categorical


class BaseVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, 
                        embedding_dim: int,
                        nclasses: int,
                        quantize: str = 'spatial',
                        codebook_dim: int = 8,
                        commitment_cost: int = 0.25,
                        usage_threshold: int = 1.0e-9,
                        cosine: bool = False,
                        gumble: bool = False,
                        temperature: float = 1.0,
                        kld_scale: float = 1.0):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.codebook_dim = codebook_dim
        self._cosine = cosine
        self._nclasses = nclasses

        self.quantize = quantize


        requires_projection = codebook_dim != embedding_dim
        self.project_in = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, codebook_dim)) if requires_projection else nn.Identity()
        self.project_out = nn.Sequential(nn.Linear(codebook_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim)) if requires_projection else nn.Identity()

        self.norm_in  = nn.LayerNorm(codebook_dim)
        self.norm_out  = nn.LayerNorm(embedding_dim)



        self._embeddings = nn.ModuleList([])
        for _ in range(self._nclasses):
            self._embeddings.append(nn.Embedding(self._num_embeddings, codebook_dim))

        self._get_distance = get_euclidian_distance
        self.loss_fn = F.mse_loss
        

        if self._cosine:
            sphere = Hypersphere(dim=codebook_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            for i in range(self._nclasses):
                self._embeddings[i].weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance
            self.loss_fn = lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())


        self.data_mean = [0]*nclasses
        self.data_std = [0]*nclasses

        self.register_buffer('_usage', torch.ones(self._nclasses, self._num_embeddings), persistent=False)


        self.kld_scale = kld_scale

        # ======================
        # Gumble parameters
        self.gumble = gumble
        if self.gumble:
            # assert (self._num_embeddings % self._nclasses) > 0, 'number of cb embeddings should be multiple of nclasses'
            self.temperature = temperature
            self.gumble_proj = nn.Sequential(nn.Linear(self.codebook_dim, self._num_embeddings))


    def update_usage(self, min_enc, y):
        self._usage[y, min_enc] = self._usage[y, min_enc] + 1  # if code is used add 1 to usage
        self._usage[y] /= 2 # decay all codes usage

    def reset_usage(self, y):
        self._usage[y].zero_() #  reset usage between epochs

    def random_restart(self, y):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage[y] < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage[y] > self._usage_threshold).squeeze(1)
        N = self.data_std.shape[1]

        if len(dead_codes) > 0:
            eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embeddings[0].weight.device)
            rand_codes = eps*self.data_std[y].unsqueeze(0).repeat(len(dead_codes), 1) +\
                                self.data_mean[y].unsqueeze(0).repeat(len(dead_codes), 1)

            with torch.no_grad():
                self._embeddings[y].weight[dead_codes] = rand_codes

            self._embeddings[y].weight.requires_grad = True


    def get_class_encodings(self, encodings, y):
        quantized = torch.matmul(encodings, self._embeddings[y].weight)
        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        return quantized


    def get_encodings(self, encodings, y):
        quantized = torch.zeros_like(encodings)
        for uy in torch.unique(y):
            class_quantized = self.get_class_encodings(encodings[y == uy], uy)
            quantized[y == uy] = class_quantized
        return quantized


    def vq_sample(self, features, y,
                        hard = False, 
                        idxs=None):
        input_shape = features.shape
        
        def _min_encoding_(distances):
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            # import pdb;pdb.set_trace()
            encoding_indices = encoding_indices.view(-1)
            # encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
            return encoding_indices

        # Flatten features
        features = features.view(-1, self.codebook_dim)


        # Update prior with previosuly wrt principle components
        # This will ensure that codebook indicies that are not in idxs wont be sampled
        # hacky way need to find a better way of implementing this
        key_codebook = self._embeddings[y].weight.clone()
        if not (idxs is None):
            idxs = torch.unique(idxs).reshape(-1, 1)
            for i in range(self._num_embeddings):
                if not (i in idxs): 
                    key_codebook[i, :] = 2*torch.max(features) 


        # Quantize and unflatten
        distances = self._get_distance(features, key_codebook)
        encoding_indices = _min_encoding_(distances)

        encodings = torch.zeros(encoding_indices.shape[0], 
                                self._num_embeddings, 
                                device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embeddings[y].weight)
    

        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)

        quantized = quantized.view(input_shape[0], -1, quantized.shape[-1])
        encoding_indices = encoding_indices.view(input_shape[0], -1, encoding_indices.shape[-1])
        return quantized, encoding_indices, encodings, None


    def gumble_sample(self, features, y,
                        hard=False, 
                        idxs = None):
        
        input_shape = features.shape

        # force hard = True when we are in eval mode, as we must quantize
        logits = self.gumble_proj(features)

        # Update prior with previosuly wrt principle components
        if not (idxs is None):
            mask_idxs = torch.zeros_like(logits)

            for b in range(input_shape[0]):
                mask_idxs[b, :, idxs[b]] = 1
            logits = mask_idxs*logits


        soft_one_hot = F.gumbel_softmax(logits, 
                                        tau=self.temperature, 
                                        dim=1, hard=hard)
        
        quantized = torch.einsum('b t k, k d -> b t d', 
                                        soft_one_hot, 
                                        self._embeddings[y].weight)            



        encoding_indices = soft_one_hot.argmax(dim=-1).view(-1, 1)
        encodings = torch.zeros(encoding_indices.shape[0], 
                                self._num_embeddings, 
                                device=features.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        encoding_indices = encoding_indices.view(input_shape[0], -1, encoding_indices.shape[-1])
        return quantized, encoding_indices, encodings, logits



    def sample(self, features, y,
                        hard = False, 
                        idxs = None,
                        quantized = None,
                        from_train = False):
        input_shape = features.shape
        nheads = input_shape[1]

        if not from_train:
            features = features.flatten(-2, -1)

            if self.quantize == 'spatial':
                features = features.permute(0, 2, 1)
            
            nheads = features.shape[1]
            quantized = torch.zeros_like(features)


            features = self.project_in(features)
            # layer norm on features
            features = self.norm_in(features)



        encoding_indices = torch.zeros(input_shape[0], nheads, 1, dtype=torch.long,).to(features.device)
        encodings = torch.zeros(input_shape[0], nheads, self._num_embeddings).to(features.device)

        for uy in torch.unique(y):
            if self.gumble:
                class_quantized, class_encoding_indices, class_encodings, logits = self.gumble_sample(features[y == uy], uy,
                                                                                            hard=hard, 
                                                                                            idxs=idxs)
            else:
                class_quantized, class_encoding_indices, class_encodings, logits = self.vq_sample(features[y == uy], uy,
                                                                                            hard=hard, 
                                                                                            idxs=idxs)
            
            quantized[y == uy] = class_quantized
            encoding_indices[y == uy] = class_encoding_indices
            encodings[y == uy] = class_encodings.view(-1, nheads, class_encodings.shape[-1])

        
        encoding_indices = encoding_indices.view(-1, 1)
        return quantized, encoding_indices, encodings, logits




    def compute_baseloss(self, quantized, inputs, logits=None, loss_type=0):
        if self.gumble:
            # + kl divergence to the prior loss
            if (logits is None):
                loss = 0
            else:
                # print (logits.min(), logits.max(), logits.mean(), '===========')
                qy = F.softmax(logits, dim=-1)
                loss = self.kld_scale * torch.sum(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=-1).mean()

        else:
             # Loss
            e_latent_loss = self.loss_fn(quantized.detach(), inputs)
            q_latent_loss = self.loss_fn(quantized, inputs.detach())


            if loss_type == 0:
                loss = q_latent_loss + self._commitment_cost * e_latent_loss
            elif loss_type == 1:
                loss = q_latent_loss
            else:
                loss = self._commitment_cost * e_latent_loss

        return loss 


    def forward(self, *args):
        return NotImplementedError()



class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, args):

        super(VectorQuantizer, self).__init__(num_embeddings = args.num_embeddings, 
                                                    embedding_dim = args.embedding_dim if args.quantize == 'spatial' else args.nfeatures,
                                                    codebook_dim = args.codebook_dim,
                                                    commitment_cost = args.beta,
                                                    usage_threshold = args.usage_threshold,
                                                    quantize = args.quantize,
                                                    nclasses = args.n_class,
                                                    cosine = args.cosine,
                                                    gumble = args.gumble,
                                                    temperature = args.temperature,
                                                    kld_scale = args.kld_scale)
        

        # ======================
        self.register_buffer('_ema_cluster_size', torch.zeros(self._nclasses, self._num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self._nclasses, self._num_embeddings, self.codebook_dim))
        self._ema_w.data.normal_()
        
        self._decay = args.decay
        self._epsilon = args.epsilon


    def forward(self, inputs, 
                        labels,
                        update=False,
                        loss_type=0,
                        idxs=None,
                        reset_usage=False):
        input_shape = inputs.shape
        features_ = inputs.flatten(-2, -1)

        if self.quantize == 'spatial':
            features_ = features_.permute(0, 2, 1)

        quantized = torch.zeros_like(features_)

        features = self.project_in(features_)
        # ====================================
        # feature disentanglement loss
        hp = hessian_penalty(self.project_in, 
                                    z=features_, 
                                    G_z = features)

        # layer norm on features
        features = self.norm_in(features)

        
        
        
        # update data stats
        if self.training:
            for uy in torch.unique(labels):
                class_features = features[labels == uy].detach().clone()
                self.data_mean[uy] = 0.9*self.data_mean[uy] + 0.1*class_features.mean(1).mean(0)
                self.data_std[uy] = 0.9*self.data_std[uy] + 0.1*class_features.std(1).mean(0)



        quantized, encoding_indices, encodings, logits = self.sample(features,
                                                                            labels, 
                                                                            idxs=idxs, 
                                                                            hard = True,
                                                                            quantized = quantized,
                                                                            from_train=True)

        if self.quantize == 'spatial':
            quantized = quantized.permute(0, 2, 1)

        quantized = quantized.view(input_shape)
        # ===================================
        ema_w_tmp = {}; sort_idx = []
        for uy in torch.unique(labels):
            # Tricks to prevent codebook collapse---
            # Restart vectors
            if update:
                if np.random.uniform() > 0.99: self.random_restart(uy)
                if reset_usage: self.reset_usage(uy)
            
                self.update_usage(encoding_indices[labels == uy], uy)


            # Use EMA to update the embedding vectors
            if self.training:
                self._ema_cluster_size[uy] = self._ema_cluster_size[uy] * self._decay + \
                                        (1 - self._decay) * encodings[labels == uy].sum(0).sum(0)
                
                # Laplace smoothing of the cluster size
                n = torch.sum(self._ema_cluster_size[uy].data)
                self._ema_cluster_size[uy] = ((self._ema_cluster_size[uy] + self._epsilon)
                                    / (n + (self._num_embeddings )* self._epsilon) * n)
                
                dw = torch.matmul(encodings[labels == uy].flatten(0, 1).t(), 
                                    features[labels == uy].view(-1, self.codebook_dim))
                
                sort_idx.append(uy)
                ema_w_tmp[uy] = (self._ema_w[uy] * self._decay + (1 - self._decay) * dw).unsqueeze(0)
                self._embeddings[uy].weight = nn.Parameter(torch.mean(ema_w_tmp[uy], 0) / self._ema_cluster_size[uy].unsqueeze(1))

        if self.training and len(ema_w_tmp):
            tmp = []
            for si in range(self._nclasses):
                if si in ema_w_tmp.keys():
                    tmp.append(ema_w_tmp[si])
                else:
                    tmp.append(self._ema_w[si].unsqueeze(0))
            self._ema_w = nn.Parameter(torch.cat(tmp, 0))

        
        # ============================
        loss = self.compute_baseloss(quantized, inputs, logits, loss_type)
        unique_code_ids = torch.unique(encoding_indices)


        # Regularization terms ==========================
        loss += 2*hp
        for _embeddings_ in self._embeddings:
            loss += get_cb_variance(_embeddings_.weight)
        
        # Straight Through Estimator
        if not self.gumble: quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.mean(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        return quantized, encoding_indices, loss, perplexity

    @torch.no_grad()
    def debate_arguments(self, inputs, topk_predictions):
        # returns set of arguments to debate upon
        self.training = False
        quantized_features = []; quantized_indicies = []
        for agent in range(topk_predictions.shape[1]):
            agent_predictions = topk_predictions[:, agent]
            quantized, encoding_indices, loss, perplexity = self.forward(inputs, agent_predictions)

            quantized_features.append(quantized.unsqueeze(0))
            quantized_indicies.append(encoding_indices.unsqueeze(0))
        
        quantized_features = torch.cat(quantized_features, 0)
        quantized_indicies = torch.cat(quantized_indicies, 0)

        self.training = True
        return quantized_features, quantized_indicies