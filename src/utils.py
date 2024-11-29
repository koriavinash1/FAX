import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True




def unique_sampling_fn(distances, nunique=-1):
    # distance: Bxntokensxncbtokens

    B, S, N = distances.shape
    if not (isinstance(nunique, list) or isinstance(nunique, np.ndarray)):
        if (nunique == -1): 
            nunique = min(S, N)
        nunique = [nunique]*B
    
    nunique = np.minimum(nunique, N)
    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        # Create a bin over codebook direction of distance vectors
        # with nunique bins and sample based on that...
        # 
        #  
        sampled_vectors = []
        sampled_distances = []
        for i in range(S):

            if i < nunique[b]:
                for k in range(N):
                    current_idx = sorted_idx[i, k]
                    if not (current_idx in sampled_vectors):
                        sampled_vectors.append(current_idx.unsqueeze(0))
                        sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
                        break
            else:
                current_idx = sorted_idx[i, 0]
                sampled_vectors.append(current_idx.unsqueeze(0))
                sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
 
        
        sampled_vectors = torch.cat(sampled_vectors, 0)
        sampled_distances = torch.cat(sampled_distances, 0)
        sampled_vectors = sampled_vectors[torch.argsort(sampled_distances, descending=False)]
        batch_sampled_vectors.append(sampled_vectors.unsqueeze(0))

    batch_sampled_vectors = torch.cat(batch_sampled_vectors, 0)
    return batch_sampled_vectors.view(-1)

    
def get_euclidian_distance(u, v):
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(u**2, dim=1, keepdim=True) + \
        torch.sum(v**2, dim=1) - 2 * torch.matmul(u, v.t())
    return d



def sorting_idxs(quantized, cidxs):
    B, N, _ = quantized.shape

    batch_sampled_vectors = []
    for b in range(B):
        sampled_idxs = cidxs[b, ...]

        st_pointer = -1
        end_pointer = 0
        unique = torch.zeros_like(sampled_idxs)
        for unique_idx in torch.sort(torch.unique(sampled_idxs)):
            idxs = torch.argwhere(sampled_idxs == unique_idx)
            
            st_pointer += 1
            end_pointer -= len(idxs[1:])

            unique[st_pointer] = idxs[0]
            unique_idx[end_pointer : end_pointer + len(idxs[1:])] = idxs[1:]

            pass 


    return quantized, cidxs


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(torch.abs(geod))/(2.0*np.pi)


def get_cb_variance(cb):
    # cb = cb / (1e-5 + torch.norm(cb, dim=1, keepdim=True))
    cd = get_cosine_distance(cb, cb)
    return 1 - torch.mean(torch.var(cd, 1)) 



def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T))


def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2.0))
    y = int(y - (size / 2.0))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=5, edgecolor=color, fill=False
    )
    return rect


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = 'ram_{}_{}x{}_{}'.format(
        config.num_glimpses, config.patch_size,
        config.patch_size, config.glimpse_scale
    )
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


    
def get_euclidian_distance(u, v):
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(u**2, dim=1, keepdim=True) + \
        torch.sum(v**2, dim=1) - 2 * torch.matmul(u, v.t())
    return d



def sorting_idxs(quantized, cidxs):
    B, N, _ = quantized.shape

    batch_sampled_vectors = []
    for b in range(B):
        sampled_idxs = cidxs[b, ...]

        st_pointer = -1
        end_pointer = 0
        unique = torch.zeros_like(sampled_idxs)
        for unique_idx in torch.sort(torch.unique(sampled_idxs)):
            idxs = torch.argwhere(sampled_idxs == unique_idx)
            
            st_pointer += 1
            end_pointer -= len(idxs[1:])

            unique[st_pointer] = idxs[0]
            unique_idx[end_pointer : end_pointer + len(idxs[1:])] = idxs[1:]

            pass 


    return quantized, cidxs


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, v.t())
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, ed2.t())
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(torch.abs(geod))/(2.0*np.pi)



def get_cb_variance(cb):
    # cb = cb / (1e-5 + torch.norm(cb, dim=1, keepdim=True))
    cd = get_cosine_distance(cb, cb)
    return 1 - torch.mean(torch.var(cd, 1)) 
