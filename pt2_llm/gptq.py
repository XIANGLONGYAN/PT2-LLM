import math
import time
from exceptiongroup import catch
import torch
import torch.nn as nn
import transformers

from pt2_llm import model_utils
import logging
import csv
import os
import sys
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
GPTQ: blockwise weight quantization with Hessian-based error compensation.
Weights are ternarized block by block (block size = quantization group size),
and the residual error is propagated to the remaining columns.
'''

def ternary_pattern_to_id(pattern):
    base3 = [(x + 1) for x in pattern]
    pattern_id = 0
    for val in base3:
        pattern_id = pattern_id * 3 + val
    return pattern_id

def count_ternary_patterns(T: torch.Tensor, pattern_size: int):
    T_flat = T.flatten().tolist()
    total = len(T_flat)
    counter = Counter()
    for i in range(0, total - pattern_size + 1):
        pattern = T_flat[i:i+pattern_size]
        pattern_id = ternary_pattern_to_id(pattern)
        counter[pattern_id] += 1
    return counter

def append_counts_to_csv(counter, csv_path):
    existing_counts = Counter()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)  # skip header
            except StopIteration:
                header = None
            for row in reader:
                if row:
                    pattern_id, count = int(float(row[0])), int(float(row[1]))
                    existing_counts[pattern_id] += count

    for k, v in counter.items():
        existing_counts[k] += v

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern_id', 'count'])
        for pattern_id in sorted(existing_counts.keys()):
            writer.writerow([pattern_id, existing_counts[pattern_id]])
            
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

class GPTQ:
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
            
        # if self.method in {'arb-x', 'ternary-x', 'ternary-x-new', 'ternary-x-new-plus1', 'ternary-x-new-plus2', 'ternary-x-new-count', 'ternary-x-new-salience', 'ternary-x-gptaq', 'ternary-x-rc'}:
        self.inp.append(inp)
        self.fp_inp1.append(self.fp_inp[0].t().unsqueeze(0))
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        # if self.HXXT == None:
        #     self.HXXT = inp.float() @ self.fp_inp[0].float().t()
        #     self.HHT = inp.float() @ inp.float().t()
        # else:
        #     self.HXXT += inp.float() @ self.fp_inp[0].float().t()
        #     self.HHT += inp.float() @ inp.float().t()
            
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
        self.fp_inp1 = torch.concat(self.fp_inp1)

        if self.reorder:
            # hessian
            perm = torch.argsort(torch.diag(H), descending=True)
            
            # random
            # torch.manual_seed(42)
            # perm = torch.randperm(W.shape[1], device=W.device)
            
            # # K-means
            # num_groups = int(self.columns / blocksize)
            
            # perm = cluster_based_perm(W, num_groups=num_groups)
            
            # print(perm)
            # sys.exit()
            # balanced K-means
            # perm = cluster_based_perm_balanced(W, num_groups=num_groups)
  
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)
            self.inp = self.inp[:, :, perm]

        Losses = torch.zeros(self.rows, device=self.dev)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        if self.gptaq:
            P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
            del self.dXXT
    
        x_loss_all = 0
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed

            S = torch.matmul(self.inp[:, :, st:ed].to(torch.float32).transpose(1, 2), self.inp[:, :, st:ed].to(torch.float32)).mean(dim=0)
            S1 = None
            S2 = None

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN (no GPTQ error compensation)
                w = W[:, col_st:col_ed]
                q = self.braq_quantizer.quantize(w, None, order=no_mask_order, S=S, H_diag=torch.diag(H[st:ed, st:ed]))
                W[:, col_st:col_ed] = q
                
                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    x_error = torch.sum((self.layer(self.inp1) - self.out1) ** 2)
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
                if self.gptaq:
                    P1 = P[col_st:col_ed, col_st:col_ed]
                    
                q_all, ternary = self.braq_quantizer.quantize(W1, S=S, compact_ternary=self.compact_ternary, codebook_path=self.codebook_path, logger=logger, H_diag=torch.diag(H[st:ed, st:ed]), S1=S1, S2=S2)
                
                # for i in range(n_cols):
                #     # shape of w: [oc, 1]
                #     w = W1[:, i]
                #     d = Hinv1[i, i]

                #     q = torch.zeros_like(w)
                #     q = q_all[:, i]


                #     Q1[:, i] = q
                #     Losses1[:, i] = (w - q) ** 2 / d**2
                #     # breakpoint()

                #     err1 = (w - q) / d
                #     Err1[:, i] = err1
                
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

                # if DEBUG:
                #     self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                #     self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                #     # x_error = torch.sum((self.layer(self.inp1) - self.out1) ** 2)   
                                 
        if self.reorder:
            Q = Q[:, invperm]
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        mse_loss = torch.norm(W_ori - Q, p='fro')**2 / W_ori.numel()
        
        # # R = (W - W_ori).to(torch.float32)
        # # x_loss = (R @ S_all * R).sum()   
        # pred = self.layer(inp_all)
        # x_loss = torch.mean((pred - out_all) ** 2)  
        # x_loss = torch.sum((self.layer(self.inp1) - self.out1) ** 2).to(torch.float32) / self.out1.numel()
        
        torch.cuda.synchronize()
        times = time.time() - tick
        logger.debug(f'time {times:.2f}')
        
        # original error
        # logger.info(f'error {torch.sum(Losses).item()}')
        # logger.info(f'x error {x_error.item()}')
        
        # print(self.layer.weight.data.shape)
        # print(self.layer.weight.data)
        # print(W_ori)
        
        # print(torch.norm(W_ori - Q, p='fro')**2)
        # print(W_ori.numel())

        # new error
        logger.debug(f"mse loss: {mse_loss.item():.6f}, x loss: {torch.sum(Losses).item():.6f}")

        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        #     self.layer.weight.data.dtype
        # )
        # x_error = torch.sum((self.layer(self.inp1) - self.out1) ** 2)   
        
        # pred_out = self.layer(self.inp1)
        # # x_error = torch.sum((pred_out - self.out1) ** 2)

        # torch.save({
        #     "target": self.out1.detach().cpu(),
        #     "pred": pred_out.detach().cpu()
        # }, "outputs1.pt")
        # print("save")
        # sys.exit()
        
        
        
        if not self.disable_gptq:
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
        

# ---- activation-aware quantization error (from utils/autosearch_arb.py) ----


def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse


def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, S):
    # Activation-aware quantization error: tr((W - Q)^T (W - Q) S).
    R = (origin_matrix - quantized_matrix).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S)
