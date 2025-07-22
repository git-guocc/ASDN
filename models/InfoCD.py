
'''
==============================================================

    0-------------------------------0
    |       Loss Functions          |
    0-------------------------------0

==============================================================

    Compute chamfer distance loss L1/L2

==============================================================
'''

import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

MIN_NORM = 1e-15


chamfer_dist = chamfer_3DDist()

def calc_cd_like_InfoV2(p1, p2):


    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
    distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    return (torch.sum(distances1) + torch.sum(distances2)) / (2*p1.shape[0])







