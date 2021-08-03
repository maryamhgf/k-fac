
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from multiprocessing import Pool


def get_lambda(alpha=1.0, alpha2=None):
    '''Return lambda'''
    if alpha > 0.:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
    else:
        lam = 1.
    return lam


def to_one_hot(inp,num_classes,device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


def cost_matrix(width):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)
    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i,m_j]= abs(i1-i2)**2 + abs(j1-j2)**2
    C = C/(width-1)**2
    C = torch.tensor(C).cuda()
    return C

cost_matrix_dict = {'2':cost_matrix(2).unsqueeze(0), '4':cost_matrix(4).unsqueeze(0), '8':cost_matrix(8).unsqueeze(0), '16':cost_matrix(16).unsqueeze(0)}

  

def mask_transport(mask, grad_pool, eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]
    
    z = (mask>0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1-plan_win) * plan

        cost += plan_lose
    return plan_win
    

def mixup_process(out, target_reweighted, hidden=0, args=None, grad=None, noise=None, adv_mask1=0, adv_mask2=0, mp=None):
    '''various mixup process'''
    if args is not None:
        mixup_alpha = args.mixup_alpha
        in_batch = args.in_batch
        mean = args.mean
        std = args.std
        box = args.box
        graph = args.graph
        beta = args.beta
        gamma = args.gamma
        eta = args.eta
        neigh_size = args.neigh_size
        n_labels = args.n_labels
        transport = args.transport
        t_eps = args.t_eps
        t_size = args.t_size
    
    block_num = 2**np.random.randint(1, 5)
    indices = np.random.permutation(out.size(0))
    
    lam = get_lambda(mixup_alpha)
    
    if hidden:
        # Manifold Mixup
        out = out*lam + out[indices]*(1-lam)
        ratio = torch.ones(out.shape[0], device='cuda') * lam
    else:
        if box:
            # CutMix
            out, ratio = mixup_box(out, out[indices], alpha=lam, device='cuda')
        elif graph:
            # PuzzleMix
            if block_num > 1:
                out, ratio = mixup_graph(out, grad, indices, block_num=block_num,
                                 alpha=lam, beta=beta, gamma=gamma, eta=eta, neigh_size=neigh_size, n_labels=n_labels,
                                 mean=mean, std=std, transport=transport, t_eps=t_eps, t_size=t_size, 
                                 noise=noise, adv_mask1=adv_mask1, adv_mask2=adv_mask2, mp=mp, device='cuda')
            else: 
                ratio = torch.ones(out.shape[0], device='cuda')
        else:
            # Input Mixup
            out = out*lam + out[indices]*(1-lam)
            ratio = torch.ones(out.shape[0], device='cuda') * lam

    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * ratio.unsqueeze(-1) + target_shuffled_onehot * (1 - ratio.unsqueeze(-1))
    
    return out, target_reweighted

def transport_image(img, plan, block_num, block_size):
    '''apply transport plan to images'''
    batch_size = img.shape[0]
    input_patch = img.reshape([batch_size, 3, block_num, block_size, block_num * block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)

    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
    input_transport = input_transport.reshape([batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, 3, block_num * block_size, block_num * block_size])
    
    return input_transport
 