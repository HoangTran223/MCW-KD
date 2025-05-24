import torch
import torch.nn as nn
import math


def pairwise_euclidean_distance(x, y):
    return torch.cdist(x, y, p=2)  # Computes pairwise Euclidean distance


def pairwise_cosin_distance(a, b, eps=1e-8):
    # a = a.float()
    # b = b.float()
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=torch.bfloat16))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=torch.bfloat16))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_attention_distance(x, y, eps=1e-8):
    # x = x.float()
    # y = y.float()
    
    d = x.shape[1]
   
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)

    dist_mt = 1.0 - attention_weights
    return dist_mt

class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha=0.1, stopThr=1e-9, OT_max_iter=100, epsilon=1e-9, ot_dist_type='attention'):
        super(ETP, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.stopThr = stopThr
        self.OT_max_iter = OT_max_iter
        self.epsilon = epsilon
        self.ot_dist_type = ot_dist_type

    def forward(self, M):
        
        device = M.device
        dtype = M.dtype
        # Sinkhorn's algorithm

        # Initialize a and b, also in bf16
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device).to(dtype=dtype)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device).to(dtype=dtype)
        # a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        # b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device).to(dtype=dtype)

        # K matrix
        K = torch.exp(-M * self.sinkhorn_alpha).to(dtype=dtype)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)
        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp