import math
import torch
import pytorch3d.loss
import pytorch3d.structures
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch_cluster import fps
from scipy.stats import entropy
import numpy as np

def entropy_from_histogram(hist):
    """计算熵的辅助函数，输入为直方图"""
    hist = hist[hist > 0]  # 去除零值
    p = hist / torch.sum(hist)  # 归一化
    return -torch.sum(p * torch.log(p))

def get_entropy_B(point_clouds, voxel_size=0.01):
    entropies = []
    
    # 计算每个点云的熵值
    for point_cloud in point_clouds:
        voxel_size = 0.001 * 100000 / point_cloud.shape[0]  # 动态调整体素大小
        min_bound = torch.min(point_cloud, dim=0).values  # 计算点云的最小边界
        max_bound = torch.max(point_cloud, dim=0).values  # 计算点云的最大边界
        voxel_counts = torch.ceil((max_bound - min_bound) / voxel_size).long()  # 计算每个维度的体素数量
        
        # 使用哈希表方式进行体素化计算
        indices = torch.floor((point_cloud - min_bound) / voxel_size).long()
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True, dim=0)
        voxel_densities = torch.bincount(inverse_indices)
        
        # 计算直方图
        hist = torch.histc(voxel_densities.float(), bins=10, min=0, max=torch.max(voxel_densities).item())
        hist[hist == 0] = 1  # 将直方图中的零值替换为 1
        hist = hist / torch.sum(hist)  # 归一化直方图

        ent = entropy_from_histogram(hist)  # 计算熵
        entropies.append(ent)  # 将熵值添加到列表中

    return torch.tensor(entropies, device=point_clouds.device)  # 返回熵值数组

def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale


def normalize_std(pc, std=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    center = pc.mean(dim=-2, keepdim=True)   # (B, 1, 3)
    pc = pc - center
    scale = pc.view(pc.size(0), -1).std(dim=-1).view(pc.size(0), 1, 1) / std
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)): #pcls.size(0)=B
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices


def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)


def pointwise_p2m_distance_normalized(pcl, verts, faces):
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face


def hausdorff_distance_unit_sphere(gen, ref):
    """
    Args:
        gen:    (B, N, 3)
        ref:    (B, N, 3)
    Returns:
        (B, )
    """
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)

    dists_ab, _, _ = pytorch3d.ops.knn_points(ref, gen, K=1)
    dists_ab = dists_ab[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ab)

    dists_ba, _, _ = pytorch3d.ops.knn_points(gen, ref, K=1)
    dists_ba = dists_ba[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ba)
    
    dists_hausdorff = torch.max(torch.cat([dists_ab, dists_ba], dim=1), dim=1)[0]

    return dists_hausdorff
