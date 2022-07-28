import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from r3d import generate_model
from r2plus1d import r3d_18, r2plus1d_18
from torch.autograd import Function

def sinkhorn(scores, eps=0.05, niters=3):
    """SK cluster, from SWAV"""
    with torch.no_grad():
        M = torch.max(scores/eps)
        Q = scores/eps - M
        Q = torch.exp(Q).transpose(0, 1)
        Q = shoot_infs(Q)
        Q = Q / torch.sum(Q)
        K, B = Q.shape
        u, r, c = torch.zeros(K).to(scores.device), torch.ones(K).to(scores.device)/K, \
            torch.ones(B).to(scores.device)/B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q = Q * u.unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).transpose(0, 1)

def shoot_infs(inp_tensor):
    """SK cluster, from SWAV"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

class LocalfeatureIntegrationTransformer(nn.Module):
    """Map a set of local features to a fixed number of SuperFeatures """

    def __init__(self, T, input_dim, dim):
        """
        T: number of iterations
        N: number of SuperFeatures
        input_dim: dimension of input local features
        dim: dimension of SuperFeatures
        """
        super().__init__()
        self.T = T
        self.input_dim = input_dim
        self.dim = dim
        # qkv
        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(input_dim, dim, bias=False)
        self.project_v = nn.Linear(input_dim, dim, bias=False)
        # layer norms
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_templates = nn.LayerNorm(dim)
        # for the normalization
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5
        # mlp
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_dim = dim//2
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, dim) )


    def forward(self, x, templates):
        """
        input:
            x has shape BxCxHxW
        output:
            template (output SuperFeatures): tensor of shape BxCxNx1
            attn (attention over local features at the last iteration): tensor of shape BxNxHxW
        """
        # reshape inputs from BxCxHxW to Bx(H*W)xC
        B, C, T, H, W = x.size()
        x = x.reshape(B, C, T*H*W).permute(0, 2, 1)

        # k and v projection
        x = self.norm_inputs(x)
        k = self.project_k(x)
        v = self.project_v(x)

        # template initialization
        templates = torch.repeat_interleave(templates.unsqueeze(0), B, dim=0)
        attn = None

        # main iteration loop
        for _ in range(self.T):
            templates_prev = templates

            # q projection
            templates = self.norm_templates(templates)
            q = self.project_q(templates)

            # attention
            q = q * self.scale  # Normalization.
            attn_logits =  torch.einsum('bnd,bld->bln', q, k)
            attn = self.softmax(attn_logits)
            attn = attn + 1e-8 # to avoid zero when with the L1 norm below
            attn = attn / attn.sum(dim=-2, keepdim=True)

            # update template
            templates = torch.einsum('bld,bln->bnd', v, attn) + templates_prev

            # mlp
            templates = self.mlp(self.norm_mlp(templates)) + templates

        return templates

class GreatModel(nn.Module):
    def __init__(self, concept=100):
        super(GreatModel, self).__init__()
        self.concept = concept
        # self.backbone = generate_model(18)
        self.backbone = r2plus1d_18()
        self.fvv = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fvs = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fvd = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fsv = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fdv = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fv = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.fs = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.fd = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.gv = nn.Sequential(nn.Linear(2*self.concept, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.gs = nn.Sequential(nn.Linear(self.concept, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.gd = nn.Sequential(nn.Linear(self.concept, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.prototype = nn.Linear(512, 2*self.concept, bias=False)
        self.prototype_s = nn.Linear(512, 2*self.concept, bias=False)
        self.prototype_d = nn.Linear(512, 2*self.concept, bias=False)
        self.lit = LocalfeatureIntegrationTransformer(1, 512, 512)
        self.register_buffer("queue", torch.randn(1024, 6, 2*self.concept))
        self.avgpool = nn.AdaptiveAvgPool3d(1)
    
    def MI_estimation(self, feat):
        rgb_1, rgb_2, frame_1, frame_2, res_1, res_2 = torch.unbind(feat, dim=1)
        
        rgb_v1 = self.fvv(rgb_1)
        rgb_v2 = self.fvv(rgb_2)
        rgb_s1 = self.fvs(rgb_1)
        rgb_s2 = self.fvs(rgb_2)
        rgb_d1 = self.fvd(rgb_1)
        rgb_d2 = self.fvd(rgb_2)
        frame_v1 = self.fsv(frame_1)
        frame_v2 = self.fsv(frame_2)
        res_v1 = self.fdv(res_1)
        res_v2 = self.fdv(res_2)       
        return rgb_v1, rgb_v2, rgb_s1, rgb_s2, rgb_d1, rgb_d2, frame_v1, frame_v2, res_v1, res_v2 

    def mlp_head(self, feat):
        rgb_1, rgb_2, frame_1, frame_2, res_1, res_2 = torch.unbind(feat, dim=1)
        rgb_v1 = self.fv(rgb_1)
        rgb_v2 = self.fv(rgb_2)
        frame_v1 = self.fs(frame_1)
        frame_v2 = self.fs(frame_2)
        res_v1 = self.fd(res_1)
        res_v2 = self.fd(res_2)  
        feat = torch.stack([rgb_v1, rgb_v2, frame_v1, frame_v2, res_v1, res_v2], dim=1)
        return feat

    def decouple_prob(self, feat, maps, use_queue):
        '''
        feat b x n x 512
        maps b x n x 512 x t x h x w
        prob b x n x 400
        attribute b x n x 200
        '''
        logit = self.mlp_head(feat)
        logit = F.normalize(logit, dim=-1)
        # prob = self.prototype(logit)
        prob = torch.cat([self.prototype(logit[:, :2].contiguous()), self.prototype_s(logit[:, 2: 4].contiguous()), self.prototype_d(logit[:, 4:].contiguous())], dim=1)
        queue = torch.cat([prob.detach(), self.queue], dim=0) if use_queue else prob.detach()
        static = prob[:, :, :self.concept].contiguous()
        dynamic = prob[:, :, self.concept:].contiguous()
        loss_static, entropy_static = self.calc_entropy(static, queue[:, :, :self.concept], [0, 1, 2, 3], [4, 5])
        loss_dynamic, entropy_dynamic = self.calc_entropy(dynamic, queue[:, :, self.concept: 2*self.concept], [0, 1, 4, 5], [2, 3])
        rec_v = self.self_reconstruct(prob[:, :2].contiguous().view(-1, 2*self.concept), feat[:, :2].contiguous().detach(), self.gv)
        rec_s = self.self_reconstruct(prob[:, 2: 4, :self.concept].contiguous().view(-1, self.concept), feat[:, 2: 4].contiguous().detach(), self.gs)
        rec_d = self.self_reconstruct(prob[:, 4:, self.concept:].contiguous().view(-1, self.concept), feat[:, 4:].contiguous().detach(), self.gd)
        local_v = self.local_match(maps[:, :2].contiguous(), self.prototype.weight, prob[:, :2].detach().contiguous())
        local_s = self.local_match(maps[:, 2: 4].contiguous(), self.prototype_s.weight[:self.concept].contiguous(), prob[:, 2: 4, :self.concept].detach().contiguous())
        local_d = self.local_match(maps[:, 4:].contiguous(), self.prototype_d.weight[self.concept:].contiguous(), prob[:, 4:, self.concept:].detach().contiguous())
        return prob.detach(), loss_static, loss_dynamic, entropy_static, entropy_dynamic, rec_v, rec_s, rec_d, local_v, local_s, local_d
    
    def local_match(self, maps, templates, prob):
        '''
        maps b x n x 512 x t x h x w
        templates b x n x 400 x 512
        prob b x n x 400
        '''
        b, n = maps.shape[:2]
        k = prob.shape[-1]
        templates = self.lit(maps.view(b*n, *maps.shape[2:]), templates)
        templates = templates.view(b, n, *templates.shape[1:])
        templates = F.normalize(templates, dim=-1)
        loss = 0
        n_pairs = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                query = templates[:, i].contiguous()
                key = templates[:, j].contiguous()
                prob_q = prob[:, i]
                prob_k = prob[:, j]
                margin_q = torch.topk(prob_q, k=10, dim=1)[0][:, -1]
                margin_k = torch.topk(prob_k, k=10, dim=1)[0][:, -1]
                index = (prob_q>margin_q.unsqueeze(-1)) * (prob_k>margin_k.unsqueeze(-1))
                index = index.float()
                delta = torch.sqrt(2 - 2*torch.einsum('bnc,pnc->bpn', query, key.detach()) + 1e-5)
                posi_loss = delta[torch.arange(b), torch.arange(b)].pow(2) * index
                nega_mask = torch.ones(b, b, k).to(maps.device)
                nega_mask[torch.arange(b), torch.arange(b)] = 0
                nega_loss = torch.relu(1.4-delta).pow(2) * (nega_mask*index.unsqueeze(1))
                loss += (torch.sum(posi_loss) + torch.sum(nega_loss)/(b-1))
                # delta = torch.sqrt(2 - 2*torch.einsum('bnc,bmc->bnm', query, key.detach()) + 1e-5)
                # posi_loss = delta[:, torch.arange(k), torch.arange(k)].pow(2) * index
                # nega_mask = torch.ones(b, k, k).to(maps.device)
                # nega_mask[:, torch.arange(k), torch.arange(k)] = 0
                # nega_index = index.unsqueeze(-1) * (prob_k>margin_k.unsqueeze(-1)).unsqueeze(1)
                # nega_loss = torch.relu(1.4-delta).pow(2) * (nega_mask*nega_index)
                # loss += (torch.sum(posi_loss) + torch.sum(nega_loss)/8)
                n_pairs += torch.sum(index)
        loss /= (n_pairs+1)
        return loss

    def self_reconstruct(self, prob, feat, layer):
        feat = F.normalize(feat, dim=-1)
        reconstruct = layer(prob)
        reconstruct = F.normalize(reconstruct, dim=-1)
        loss = -torch.mean(torch.einsum('bnc,bnc->bn', feat, reconstruct.view(*feat.shape)))
        return loss

    def calc_entropy(self, prob, queue, pos_order, neg_order):
        '''
        prob b x n x 200
        queue k x n x 200
        query b x 200
        key b x 200
        '''
        loss = 0
        for i in pos_order:
            query = queue[:, i].contiguous()
            query = sinkhorn(query)[:prob.shape[0]]
            subloss = 0
            for k in pos_order:
                if i == k:
                    continue
                key = prob[:, k] / 0.1
                subloss -= torch.mean(torch.sum(query * F.log_softmax(key, dim=1), dim=1))
            loss += subloss / (len(pos_order)-1)
        loss /= len(pos_order)

        entropy = 0
        for i in neg_order:
            query = prob[:, i] / 0.1
            query = F.softmax(query, dim=1)
            entropy += torch.mean(torch.sum(query * torch.log(query), dim=1))
        if len(neg_order) > 0:
            entropy /= len(neg_order)
        return loss, entropy

    def forward(self, seq, use_queue=False):
        b, n, c, t, h, w = seq.shape
        seq = seq.view(b*n, c, t, h, w)
        feat, maps = self.backbone(seq)
        feat = feat.view(b, n, 512)
        maps = maps.view(b, n, 512, 2, 8, 8)
        # rgb = seq[:, :2].contiguous().view(b*2, c, t, h, w)
        # frame = seq[:, 2: 4].contiguous().view(b*2, c, t, h, w)
        # res = seq[:, -2:].contiguous().view(b*2, c, t, h, w)
        # rgb = self.backbone(rgb)
        # frame = self.backbone_s(frame)
        # res = self.backbone_d(res)
        # feat = torch.cat([rgb.view(b, 2, 512), frame.view(b, 2, 512), res.view(b, 2, 512)], dim=1)
        
        return self.decouple_prob(feat, maps, use_queue), self.MI_estimation(feat)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
