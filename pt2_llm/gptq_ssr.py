import math
import time
from exceptiongroup import catch
import torch
import torch.nn as nn
import transformers
from pt2_llm.gptq import error_computing_x_all_accelerate

from pt2_llm import model_utils
import logging
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from collections import defaultdict
import random
logger = logging.getLogger()

DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
GPTQ_SSR: GPTQ blockwise quantization combined with Structural Similarity-based
Reordering (SSR). Before quantizing each block, columns are reordered so that the
most structurally similar ones are grouped together (paper Section 3.3).
'''
            
def cluster_based_perm(W, num_groups=32, pca_dim_ratio=0.05):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    W_norm = torch.nn.functional.normalize(W, dim=0)
    
    W_np = W_norm.T.cpu().numpy()  # shape: (dim, features)

    # orig_dim = W_np.shape[1]
    # pca_dim = int(orig_dim * pca_dim_ratio)
    # if pca_dim < orig_dim:
    #     W_np = PCA(n_components=pca_dim).fit_transform(W_np)

    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto').fit(W_np)
    labels = kmeans.labels_

    from collections import Counter
    label_counts = Counter(labels)

    first_cluster_label = min(label_counts.keys())
    first_cluster_size = label_counts[first_cluster_label]
    print(f"[INFO] Cluster-{first_cluster_label} contains {first_cluster_size} columns.")

    # counts = Counter(kmeans.labels_)
    # print(len(counts.items()))

    perm = torch.argsort(torch.tensor(labels, device=W.device))
    return perm

def cluster_based_perm_balanced(W, num_groups=32, pca_dim_ratio=0.05):
    """
    Cluster columns with KMeans into equal-sized groups; return a permutation for reordering.
    """
    d, n = W.shape
    assert n % num_groups == 0, "number of columns must be divisible by num_groups"
    group_size = n // num_groups

    W_norm = torch.nn.functional.normalize(W, dim=0)
    W_np = W_norm.T.cpu().numpy()  # shape: (n_cols, dim)

    orig_dim = W_np.shape[1]
    pca_dim = int(orig_dim * pca_dim_ratio)
    if pca_dim < orig_dim:
        W_np = PCA(n_components=pca_dim).fit_transform(W_np)

    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto').fit(W_np)
    
    labels = kmeans.labels_

    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    groups = [[] for _ in range(num_groups)]
    overflow = []

    for cluster_id in range(num_groups):
        members = cluster_to_indices[cluster_id]
        if len(members) <= group_size:
            groups[cluster_id].extend(members)
        else:
            groups[cluster_id].extend(members[:group_size])
            overflow.extend(members[group_size:])

    for cluster_id in range(num_groups):
        needed = group_size - len(groups[cluster_id])
        if needed > 0:
            groups[cluster_id].extend(overflow[:needed])
            overflow = overflow[needed:]

    perm = torch.tensor([i for group in groups for i in group], device=W.device)
    return perm

def topk_similar_columns(W_left, blocksize=128):
    """
    Select the blocksize most similar columns from W_left; return their permutation (SSR).
    """
    # print(W_left.device)
    n_cols = W_left.shape[1]
    if blocksize >= n_cols:
        return torch.arange(n_cols, device=W_left.device)
    # L2 normalize each column
    W_norm = torch.nn.functional.normalize(W_left, dim=0)  # [out_dim, n_cols]
    
    mean_vec = W_norm.mean(dim=1, keepdim=True)  # shape: [out_dim, 1]
    similarity = torch.matmul(mean_vec.T, W_norm).squeeze(0)  # [n_cols]
    
    topk_indices = torch.topk(similarity, k=blocksize, largest=True).indices
    # print(topk_indices[0].dtype)
    rest_indices = torch.tensor(
        [i for i in range(W_left.shape[1]) if i not in topk_indices],
        device=W_left.device
    )
    perm = torch.cat([topk_indices, rest_indices], dim=0)
    return perm

def ternary_init(x):
    mean = x.mean(dim=1, keepdim=True)
    x_centered = x - mean
    
    threshold = 0.75 * x_centered.abs().mean(dim=1, keepdim=True)

    ternary_matrix = torch.zeros_like(x, dtype=torch.int)
    ternary_matrix[x_centered > threshold] = 1
    ternary_matrix[x_centered < -threshold] = -1
    
    masked_x = x_centered * ternary_matrix
    numerator = masked_x.sum(dim=1, keepdim=True)
    denominator = ternary_matrix.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
    scale = numerator / denominator

    scale = scale.squeeze(1)  # [4096, 1] -> [4096]
    mean = mean.squeeze(1)    # [4096, 1] -> [4096]
    # print(scale.shape)
    # print(mean.shape)
    # print(ternary_matrix.shape)

    return ternary_matrix

def topk_pattern_aligned_columns(W_left, blocksize=128):
    """
    Select the blocksize columns most similar to the main pattern.
    Return the column permutation (indices within W_left).
    """
    n_cols = W_left.shape[1]
    device = W_left.device

    if blocksize >= n_cols:
        return torch.arange(n_cols, device=device)

    W_ternary = ternary_init(W_left).to(torch.float32)  # shape: [dim, n_cols], values in {-1, 0, 1}

    ref_pattern = W_ternary.mean(dim=1).sign()

    dot = torch.matmul(ref_pattern, W_ternary)  # shape: [n_cols]
    norm_ref = ref_pattern.norm()
    norm_each = W_ternary.norm(dim=0)
    similarity = dot / (norm_ref * norm_each + 1e-6)

    topk_indices = torch.topk(similarity, k=blocksize, largest=True).indices

    mask = torch.ones(n_cols, dtype=torch.bool, device=device)
    mask[topk_indices] = False
    rest_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    perm = torch.cat([topk_indices, rest_indices], dim=0)
    return perm

class GPTQ_SSR:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False, method='arb', order2_group=False, gptaq=False, compact_ternary=False, pattern_size=8, csv_path='ternary_pattern.csv', codebook_path=None, reorder=False
    ):
        self.method = method
        self.order2_group = order2_group
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq
        self.gptaq = gptaq
        self.inp = []
        self.fp_inp1 = []
        if self.gptaq:
            self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
            self.fp_inp = []
        self.compact_ternary = compact_ternary
        self.pattern_size = pattern_size
        self.csv_path = csv_path
        self.codebook_path = codebook_path
        self.reorder = reorder
        # self.inp2 = torch.zeros(self.columns, self.columns, device=self.dev, dtype=torch.float16)

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            
        self.inp.append(inp)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            
        self.H *= self.nsamples / (self.nsamples + tmp)
        if self.gptaq:
            self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        if self.gptaq:
            dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
            self.dXXT += dX.matmul(inp.t()) 
            del self.fp_inp[0]

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01,
                    orders=(1,1,2),
                    num_p=1,
                    disable_mask=False,
                    no_mask_order=1,
                    alpha=0.25,
                    ):
        W = self.layer.weight.data.clone()
        W_ori = self.layer.weight.data.clone()

        W_ori = W_ori.float()
        
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        W_ori = W_ori.float()
        tick = time.time()
        T = torch.zeros_like(W)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        if self.gptaq:
            self.dXXT[:, dead] = 0

        self.inp = torch.concat(self.inp)

        Losses = torch.zeros(self.rows, device=self.dev)
        Q = torch.zeros_like(W)
        T = torch.zeros_like(W)
    
        x_loss_all = 0        
        perm = torch.arange(self.columns, device=self.dev)

        # is_zero_column = (W == 0).all(dim=0)

        # num_zero_columns = is_zero_column.sum().item()
        # num_total_columns = W.shape[1]

        # print(f"Zero columns: {num_zero_columns} / {num_total_columns}")
        
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            st = col_st
            ed = col_ed
            num_left = self.columns - blocki * blocksize
            n_this_block = min(blocksize, num_left)
            if self.reorder:
                remaining = perm[col_st:] 
                W_left = W[:, remaining]
                num_groups = max(1, W_left.shape[1] // blocksize)
                # perm_local = cluster_based_perm(W_left, num_groups=num_groups)
                perm_local = topk_similar_columns(W_left, blocksize=blocksize)
                # perm_local = topk_pattern_aligned_columns(W_left, blocksize=blocksize)
                perm[col_st:] = remaining[perm_local]
            
            W = W[:, perm]
            H_perm = H[perm][:, perm]
            dXXT_perm = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)
            inp_perm = self.inp[:, :, perm]
            
            damp = percdamp * torch.mean(torch.diag(H_perm))
            diag = torch.arange(self.columns, device=self.dev)
            H_perm[diag, diag] += damp
            H_perm = torch.linalg.cholesky(H_perm)
            H_perm = torch.cholesky_inverse(H_perm)
            H_perm = torch.linalg.cholesky(H_perm, upper=True)
            Hinv = H_perm
            
            P = alpha * ((dXXT_perm @ Hinv.T).triu_(diagonal=1)) @ Hinv

            S = torch.matmul(inp_perm[:, :, st:ed].to(torch.float32).transpose(1, 2), inp_perm[:, :, st:ed].to(torch.float32)).mean(dim=0)   # avoid overflow
            # S = torch.matmul(inp_perm[:, :, st:ed].to(torch.float32).transpose(1, 2), inp_perm[:, :, st:ed].to(torch.float32)).sum(dim=0)

            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
            if self.gptaq:
                P1 = P[col_st:col_ed, col_st:col_ed]
            # print(self.layer)
            q_all, ternary = self.braq_quantizer.quantize(W1, S=S, logger=logger)
                
            diff = W1 - q_all        # [oc, n_cols]
            d_vec = torch.diag(Hinv1).view(1, -1)  # [1, n_cols]
            Q1 = q_all
            Q[:, col_st:col_ed] = Q1
            x_loss = error_computing_x_all_accelerate(W1, Q1, S)
            x_loss_all += x_loss
            Losses1 = (diff ** 2) / (d_vec ** 2)
            Err1 = diff / d_vec
            
            W[:, col_st:col_ed] = Q1
            T[:, col_st:col_ed] = ternary

            Losses += torch.sum(Losses1, 1) / 2


            if self.gptaq:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:]) - W1.matmul(P[col_st:col_ed, col_ed:])
            else:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

            W = W[:, invperm]
        Q = Q[:, invperm]
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        mse_loss = torch.norm(W_ori - Q, p='fro')**2 / W_ori.numel()
        
        torch.cuda.synchronize()
        times = time.time() - tick
        logger.debug(f'time {times:.2f}')
    
        # new error
        logger.debug(f"mse loss: {mse_loss.item():.6f}, x loss: {torch.sum(Losses).item():.6f}")
        
        del W1, Q1, W, Err1, Losses1, Hinv1, W_ori, mse_loss, x_loss
        del H, Hinv, self.inp, S
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
            self.fp_inp1 = None
        self.H = None
        self.Losses = None
        if self.gptaq:
            self.dXXT = None
        torch.cuda.empty_cache()
        model_utils.cleanup_memory(verbos=False)
        
