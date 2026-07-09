from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
import numpy as np
from collections import Counter
import csv
import os

index = 0
@torch.no_grad()
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

    return mean, scale, ternary_matrix

@torch.no_grad()
def ternary_init_only(x):
    
    mean, scale, ternary_matrix = ternary_init(x)

    w_ternary = ternary_matrix * scale[:, None] + mean[:, None]
    return w_ternary

@torch.no_grad()
def atq(x, S=None, logger=None):
    if torch.all(x == 0):
        return torch.zeros_like(x) 
    
    device = x.device
    mean, scale, ternary_matrix = ternary_init(x)
    
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    
    mean_tensor_all = mean
    scale_tensor_all = scale
    new_ternary = ternary_matrix.clone()

    sum_order = ternary_matrix * scale_tensor_all[:, None] + mean_tensor_all[:, None]

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha = scale_tensor_all.clone()
    
    # loss1_1 = L1loss(x, refine_mean, new_alpha, ternary_matrix)
    # loss2_1 = L2loss(x, refine_mean, new_alpha, ternary_matrix, S)
    # logger.info(f"initial L1 loss: {loss1_1.item():.6f}, L2 loss: {loss2_1.item():.6f}")
    
    for k in range(100):
        prev_ternary = new_ternary.clone()
        
        # # 1. Fix alpha and B, update mean
        # residual = new_matrix - sum_order_alternating
        # mean_tensor_all = torch.mean(residual, dim=1)
        # refine_mean += mean_tensor_all.clone()
        
        # # 2. Fix mean and B, update alpha
        # new_alpha = 1. / (torch.sum(new_ternary * new_ternary, dim=1) + 1e-8) * torch.sum(new_ternary * (new_matrix - refine_mean[:, None]), dim=1)
        
        new_alpha, refine_mean = solve_closed_form_alpha_mu(new_ternary, new_matrix)
        # loss1_2 = L1loss(x, refine_mean, new_alpha, new_ternary)
        # loss2_2 = L2loss(x, refine_mean, new_alpha, new_ternary, S)
        # logger.info(f"after arb L1 loss: {loss1_2.item():.6f}, L2 loss: {loss2_2.item():.6f}")
        # 3. Fix mean and alpha, update T
        new_ternary = update_ternary(new_matrix, new_alpha, refine_mean)
        
        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_ternary + refine_mean[:, None]) 
        
        if torch.equal(new_ternary, prev_ternary):
            # print(f"Early stopping at iteration {k} due to convergence.")
            break
        # loss1_2 = L1loss(x, refine_mean, new_alpha, new_ternary)
        # loss2_2 = L2loss(x, refine_mean, new_alpha, new_ternary, S)
        # logger.info(f"after arb L1 loss: {loss1_2.item():.6f}, L2 loss: {loss2_2.item():.6f}")
        
    # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    # sys.exit()
    # loss1_2 = L1loss(x, refine_mean, new_alpha, new_ternary)
    # loss2_2 = L2loss(x, refine_mean, new_alpha, new_ternary, S)
    # logger.info(f"after arb L1 loss: {loss1_2.item():.6f}, L2 loss: {loss2_2.item():.6f}")


    T = new_ternary.to(torch.float32)

    T = T.to(dtype=torch.float64)
    S = S.to(dtype=torch.float64)
    x = x.to(dtype=torch.float64)
    one = torch.ones((S.shape[0], 1), device=device, dtype=torch.float64)
    a = torch.sum((T @ S) * T, dim=1) 
    
    b = T @ S @ one  # shape: (M, 1)
    b = b.squeeze(-1)  # shape: (M,)
    
    d = (one.T @ S @ one).squeeze() 
    
    y1 = torch.sum((x @ S) * T, dim=1) 
    
    y2 = x @ S @ one  # shape: (N, 1)
    y2 = y2.squeeze(-1)  # shape: (N,)
    
    bc = b * b
    
    ad = a * d
    denom = ad - bc
    
    num_alpha = d * y1 - b * y2
    
    num_mu = a * y2 - b * y1

    new_alpha_try = num_alpha / denom  # shape: (M,)
    refine_mean_try = num_mu / denom        # shape: (M,)

    
    if torch.isnan(new_alpha_try).any() or torch.isnan(refine_mean_try).any():
        # print("NaN detected!")
        # print("new_alpha:", new_alpha)
        # print("refine_mean:", refine_mean)
        # print("a:", a)
        # print("b:", b)
        # print("bc (not directly used):", bc)
        # print("d:", d)
        # print("y1:", y1)
        # print("y2:", y2)
        # print("denom:", denom)
        # print("num_alpha:", num_alpha)
        # print("num_mu:", num_mu)
        # print("T:", T)
        # print("S:", S)
        # print("x:", x)
        # sys.exit("Exiting due to NaN in new_alpha or refine_mean.")
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * T + refine_mean[:, None])
        
    else:
        new_alpha = new_alpha_try
        refine_mean = refine_mean_try
        # T = update_ternary(new_matrix, new_alpha, refine_mean)
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * T + refine_mean[:, None])
        
    S = S.to(dtype=torch.float32)
    # loss1_3 = L1loss(x, refine_mean, new_alpha, T)
    # loss2_3 = L2loss(x, refine_mean, new_alpha, T, S)
    # logger.info(f"after arb-x L1 loss: {loss1_3.item():.6f}, L2 loss: {loss2_3.item():.6f}")
    
    # print("kkkkkkkkkkkkkk")
    
    sum_order_alternating = sum_order_alternating.to(dtype=torch.float32)
    return sum_order_alternating

@torch.no_grad()
def atq_aga(x, S=None, logger=None):
    if torch.all(x == 0):
        return torch.zeros_like(x) 
    
    device = x.device
    mean, scale, ternary_matrix = ternary_init(x)
    
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    
    mean_tensor_all = mean
    scale_tensor_all = scale
    new_ternary = ternary_matrix.clone()

    sum_order = ternary_matrix * scale_tensor_all[:, None] + mean_tensor_all[:, None]

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha = scale_tensor_all.clone()
    
    # loss1_1 = L1loss(x, refine_mean, new_alpha, ternary_matrix)
    # loss2_1 = L2loss(x, refine_mean, new_alpha, ternary_matrix, S)
    # logger.info(f"initial L1 loss: {loss1_1.item():.6f}, L2 loss: {loss2_1.item():.6f}")
    
    # for k in range(100):
    #     prev_ternary = new_ternary.clone()
        
    #     # # 1. Fix alpha and B, update mean
    #     # residual = new_matrix - sum_order_alternating
    #     # mean_tensor_all = torch.mean(residual, dim=1)
    #     # refine_mean += mean_tensor_all.clone()
        
    #     # # 2. Fix mean and B, update alpha
    #     # new_alpha = 1. / (torch.sum(new_ternary * new_ternary, dim=1) + 1e-8) * torch.sum(new_ternary * (new_matrix - refine_mean[:, None]), dim=1)
        
    #     new_alpha, refine_mean = solve_closed_form_alpha_mu(new_ternary, new_matrix)
        
    #     # 3. Fix mean and alpha, update T
    #     new_ternary = update_ternary(new_matrix, new_alpha, refine_mean)
        
    #     # Final refine results
    #     sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_ternary + refine_mean[:, None]) 
        
    #     if torch.equal(new_ternary, prev_ternary):
    #         # print(f"Early stopping at iteration {k} due to convergence.")
    #         break

    # loss1_2 = L1loss(x, refine_mean, new_alpha, new_ternary)
    # loss2_2 = L2loss(x, refine_mean, new_alpha, new_ternary, S)
    # logger.info(f"after arb L1 loss: {loss1_2.item():.6f}, L2 loss: {loss2_2.item():.6f}")


    T = new_ternary.to(torch.float32)

    T = T.to(dtype=torch.float64)
    S = S.to(dtype=torch.float64)
    x = x.to(dtype=torch.float64)
    one = torch.ones((S.shape[0], 1), device=device, dtype=torch.float64)
    a = torch.sum((T @ S) * T, dim=1) 
    
    b = T @ S @ one  # shape: (M, 1)
    b = b.squeeze(-1)  # shape: (M,)
    
    d = (one.T @ S @ one).squeeze() 
    
    y1 = torch.sum((x @ S) * T, dim=1) 
    
    y2 = x @ S @ one  # shape: (N, 1)
    y2 = y2.squeeze(-1)  # shape: (N,)
    
    bc = b * b
    
    ad = a * d
    denom = ad - bc
    
    num_alpha = d * y1 - b * y2
    
    num_mu = a * y2 - b * y1

    new_alpha_try = num_alpha / denom  # shape: (M,)
    refine_mean_try = num_mu / denom        # shape: (M,)

    
    if torch.isnan(new_alpha_try).any() or torch.isnan(refine_mean_try).any():
        # print("NaN detected!")
        # print("new_alpha:", new_alpha)
        # print("refine_mean:", refine_mean)
        # print("a:", a)
        # print("b:", b)
        # print("bc (not directly used):", bc)
        # print("d:", d)
        # print("y1:", y1)
        # print("y2:", y2)
        # print("denom:", denom)
        # print("num_alpha:", num_alpha)
        # print("num_mu:", num_mu)
        # print("T:", T)
        # print("S:", S)
        # print("x:", x)
        # sys.exit("Exiting due to NaN in new_alpha or refine_mean.")
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * T + refine_mean[:, None])
        
    else:
        new_alpha = new_alpha_try
        refine_mean = refine_mean_try
        # T = update_ternary(new_matrix, new_alpha, refine_mean)
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * T + refine_mean[:, None])
        
    # S = S.to(dtype=torch.float32)
    # loss1_3 = L1loss(x, refine_mean, new_alpha, T)
    # loss2_3 = L2loss(x, refine_mean, new_alpha, T, S)
    # logger.info(f"after arb-x L1 loss: {loss1_3.item():.6f}, L2 loss: {loss2_3.item():.6f}")
    
    sum_order_alternating = sum_order_alternating.to(dtype=torch.float32)
    return sum_order_alternating

@torch.no_grad()
def update_ternary(x, alpha, mean):
    """
    Snap each element to the nearest of {mean-alpha, mean, mean+alpha}.
    x: [B, D] weight matrix
    alpha: [B, 1] scale
    mean: [B, 1] offset

    Returns:
        new_ternary: [B, D] with values in {-1, 0, +1}
    """
    alpha = alpha  # [B, 1]
    mean = mean    # [B, 1]
    
    d_neg1 = torch.abs(x - (mean[:, None] - alpha[:, None]))
    d_0    = torch.abs(x - mean[:, None])
    d_pos1 = torch.abs(x - (mean[:, None] + alpha[:, None]))

    distances = torch.stack([d_neg1, d_0, d_pos1], dim=0)   # [3, B, D]
    indices = torch.argmin(distances, dim=0)               # [B, D]

    ternary = (indices - 1).to(torch.float32)
    return ternary

@torch.no_grad()
def atq_itf(x, order=1, iter=15):

    mean, scale, ternary_matrix = ternary_init(x)
    
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    
    mean_tensor_all = mean
    scale_tensor_all = scale
    new_ternary = ternary_matrix.clone()

    sum_order = ternary_matrix * scale_tensor_all[:, None] + mean_tensor_all[:, None]

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()

    for k in range(100):
        prev_ternary = new_ternary.clone()
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        mean_tensor_all = torch.mean(residual, dim=1)
        # print(mean_tensor_all.shape)
        # print(refine_mean.shape)
        refine_mean += mean_tensor_all.clone()
        
        
        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_ternary * new_ternary, dim=1) + 1e-8) * torch.sum(new_ternary * (new_matrix - refine_mean[:, None]), dim=1)
        
        # 3. Fix mean and alpha, update T
        new_ternary = update_ternary(new_matrix, new_alpha, refine_mean)
        
        # threshold = 0.75 * x_centered.abs().mean(dim=1, keepdim=True)

        # new_ternary = torch.zeros_like(new_matrix, dtype=torch.int)
        # new_ternary[x_centered >  threshold] =  1
        # new_ternary[x_centered < -threshold] = -1

                    
        if torch.equal(new_ternary, prev_ternary):
            # print(f"Early stopping at iteration {k} due to convergence.")
            break
        
        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_ternary + refine_mean[:, None]) 


    return sum_order_alternating

@torch.no_grad()
def solve_closed_form_alpha_mu(B: torch.Tensor, x: torch.Tensor):
    """
    B: (n, d) - ternary matrix
    x: (n, d) - input matrix
    """
    n, d = x.shape

    sum_B = B.sum(dim=1)                  # (n,)
    sum_x = x.sum(dim=1)                  # (n,)
    sum_Bx = (B * x).sum(dim=1)           # (n,)
    sum_B2 = (B * B).sum(dim=1)           # (n,)

    # denom = sum_B2 - (sum_B ** 2) / d + 1e-8
    denom = sum_B2 - (sum_B ** 2) / d
    alpha = (sum_Bx - (sum_B * sum_x) / d) / denom
    mu = (sum_x - alpha * sum_B) / d
    # print(denom)

    return alpha, mu

class TernaryQuantizer(nn.Module):
    def __init__(self, weight, method="arb", groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.mean = 0

    def quantize(self, w, mask=None, order=1, groupi=0, S=None, compact_ternary=False, codebook_path=None, logger=None, H_diag=None, S1=None, S2=None):
        if mask is None:
            mask = torch.ones_like(w, dtype=torch.bool)
        if self.method == "ternary-init":       # asymmetric ternary init only
            w = ternary_init_only(w)
        elif self.method == "atq-itf":             # Iterative Ternary Fitting only
            w = atq_itf(w)
        elif self.method == "atq-aga":             # Activation-aware Grid Alignment only
            w = atq_aga(w, S=S, logger=logger)
        elif self.method == "atq":                 # full ATQ = ITF + AGA
            w = atq(w, S=S, logger=logger)

        T = torch.zeros_like(w, dtype=torch.float32)
        return w, T
