import numpy as np
from pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

def block_decider(name):
    if name == 'startblock':
        return StartBlock
    if name == 'upsample':
        return Upsampling 
    if name == 'downsample':
        return Downsampling

class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, p, x, o):
        x = self.mlp(p)  # [B*N, d_out]
        return p, x, o
    
class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nsample = nsample
        self.stride = stride
        self.mre = MRE(d_in, d_out)

    def forward(self, p, x, o):
        # MRE
        x = self.mre(
            p, x, o
        )  # p[5000, 3] x[5000, 32] o[1000, 2000, 3000, 4000, 5000] ;return x [5000, 64]

        # fps
        count = o[0].item() * self.stride // (self.stride + 1)
        n_o = [count * (i + 1) for i in range(o.shape[0])]
        # n_o = torch.cuda.IntTensor(n_o)
        n_o = torch.tensor(n_o, dtype=torch.int32, device='cuda')
        idx = pointops.furthestsampling(p, o, n_o)  # (m)

        n_p = p[idx.long(), :]  # (m, 3)  [2500, 3] FPS后的点
        n_x = x[idx.long(), :]  # (m, c)  [2500, 64] FPS后的点对应特征

        mask = torch.ones(p.size(0), dtype=torch.bool)
        mask[idx.long()] = False

        return n_p, n_x, n_o # [2500,3] [2500,64] [5] [2500]


class RFE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.BiMLP = nn.Sequential(
            nn.Conv1d(10, 2*d_out, 1),
            nn.BatchNorm1d(2*d_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*d_out, d_out, 1),
        )
        
        self.score_fn = nn.Sequential(
            nn.Linear(d_out * 2, d_out * 2, bias=False), nn.Softmax(dim=-2)
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(d_out * 2, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )

    def forward(self, p, x):
        # position embedding
        extended_coords = p[:, 0, :].unsqueeze(1).expand(-1, 16, -1)
        neighbors = p
        dist = torch.sqrt(
            torch.sum((extended_coords - neighbors) ** 2, dim=2, keepdim=True)
        )
        concat = torch.cat(
            [extended_coords, neighbors, extended_coords - neighbors, dist], dim=-1
        )  # [m, 16, 10]
        p_c = self.BiMLP(concat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # 对p和x进行融合自注意力加强
        p_x = torch.cat([p_c, x], dim=-1)  # [m, 16, 32]
        scores = self.score_fn(p_x)
        features = torch.sum(scores * p_x, dim=1, keepdim=True)
        features = self.mlp_out(features.squeeze())
        return features


class MRE(nn.Module):
    def __init__(self, d_in, d_out,):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp0 = nn.Sequential(
            nn.Linear(d_in, d_out // 2),
            nn.BatchNorm1d(d_out // 2),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(d_out , d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )
        self.mlp01 = nn.Sequential(
            nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )
        self.Rfe_1 = RFE(d_out // 2, d_out // 2)
        self.Rfe_2 = RFE(d_out // 2, d_out // 2)

    def forward(self, p, x, o):
        x_start = x
        x = self.mlp0(x)
        xr, _ = pointops.queryandgroup(
            16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True
        )
        x = self.Rfe_1(xr[:, :, :3], xr[:, :, 3:])  # x:[5000, 16] xr:[5000, 16, 3+16]
        x_middle = x
        xr, _ = pointops.queryandgroup(
            16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True
        )
        x = self.Rfe_2(xr[:, :, :3], xr[:, :, 3:])
        x = torch.cat([x_middle, x], dim=1)
        x = self.mlp01(x_start) + self.mlp1(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_fusion, d_out, nsample):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.nsample = nsample
        self.d_out = d_out

        self.CrossPT_func = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse,
            attn_mlp_hidden_mult=1,
            num_neighbors=16
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse + d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
        
        self.dense_mlp = nn.Sequential(
            nn.Linear(d_in_dense, d_in_sparse),
            nn.BatchNorm1d(d_in_sparse),
            nn.ReLU(inplace=True)
        )

    def forward(self,p1,x1,o1, p2,x2,o2, batch_size=5):
        '''
            pxo1: dense 
            pxo2: sparse  
        '''
        num_points = p1.shape[0]//batch_size

        x1_dense = self.dense_mlp(x1)

        x2_interpolated = pointops.interpolation(p2, p1, x2, o2, o1, k=8)

        x1_enhance = self.CrossPT_func( x1_dense.view(batch_size,num_points,-1),
                                        x2_interpolated.view(batch_size,num_points,-1),
                                        x2_interpolated.view(batch_size,num_points,-1),
                                        p1.view(batch_size,num_points,-1)).view(batch_size*num_points,-1)
        
        x = self.mlp(torch.cat([x1_enhance, x1], dim=-1))
        return p1, x, o1 


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, attn_mlp_hidden_mult=4,num_neighbors = None):
        super().__init__()
        self.num_neighbors = num_neighbors
        dim_cf_q = dim // 2
        self.to_q = nn.Linear(dim, dim, bias=False)
        # self.to_kv = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.attn_bimlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x_e, x_r, x_d, pos):
        '''
        x_e:(B, N, C1)
        x_r:(B, N, C2)
        x_d:(B, N, C2)
        '''
        n = x_e.shape[1]  # x_e: (B, N, C)
        # get queries, keys, values
        q = self.to_q(x_e)
        # k, v = self.to_kv(x_d).chunk(2, dim=-1)
        k = self.to_k(x_r)
        v = self.to_v(x_d)

        # qk_rel = q[:, :, None, :] - k[:, None, :, :]  # (B, N, N, C)

        # v = repeat(v, 'b j d -> b i j d', i = n)  # (B, N, N, C)
        # determine k nearest neighbors for each point, if specified
        if self.num_neighbors is not None and self.num_neighbors < n:
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
            rel_dist = rel_pos.norm(dim = -1)
            dist, indices = rel_dist.topk(self.num_neighbors, largest = False)

            v = batched_index_select(v, indices, dim = 1)
            k = batched_index_select(k, indices, dim = 1)
            qk_rel = q[:,:,None] - k
            # qk_rel = batched_index_select(qk_rel, indices, dim = 2)
            x_e = batched_index_select(x_e, indices, dim = 1)

        v = v + x_e
        # use attention mlp
        B, N, neigh_num, C = qk_rel.shape
        # print(f'qk_rel: {qk_rel.shape}, x_e: {x_e.shape}')

        sim = self.attn_bimlp((qk_rel + x_e).view(B*N*neigh_num, C)).view(B, N, neigh_num, C)
        # sim = sim.view(B, N, neigh_num, C)
        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = torch.einsum('bmnf,bmnf->bmf', attn, v)
        return agg


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

