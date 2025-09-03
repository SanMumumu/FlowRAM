import torch
import einops
import numpy as np
from openpoints.models.layers import furthest_point_sample as farthest_point_sample, random_sample

def index_points(points, idx):
    # Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    C = points.shape[-1]
    if len(idx.shape) == 2:
        new_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    elif len(idx.shape) == 3:
        B, K, S = idx.shape
        idx = idx.reshape(B, -1)
        new_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        new_points = new_points.reshape(B, K, S, C)
    return new_points

def determine_sampling_points(num_points, ratio=1200/1024):
    """Determine the total number of points to sample based on the requested number."""
    return int(num_points * ratio)


def process_xyz_with_fps_and_nearest(xyz_list, feature_list, pcd, pcd_feat, sqrt_alpha, num_points=4096):
    processed_xyz_list = []
    processed_feature_list = []

    for i, (xyz, features) in enumerate(zip(xyz_list, feature_list)):
        if xyz.shape[0] == 0 or xyz.shape[0] < num_points or sqrt_alpha[i]>0.9:
            # If xyz is empty or the number of points is less than num_points, directly sample the farthest point of pcd
            sampled_indices = farthest_point_sample(pcd[i].contiguous().unsqueeze(0), num_points).long()
            sampled_xyz = index_points(pcd[i].unsqueeze(0), sampled_indices).squeeze(0)
            sampled_features = index_points(pcd_feat[i].unsqueeze(0), sampled_indices).squeeze(0)
        else:
            # If the number of points is greater than num_points, use FPS for sampling
            if xyz.shape[0] > num_points:
                sampled_indices = farthest_point_sample(xyz.unsqueeze(0), num_points).long()
                sampled_xyz = index_points(xyz.unsqueeze(0), sampled_indices).squeeze(0)
                sampled_features = index_points(features.unsqueeze(0), sampled_indices).squeeze(0)
            else:
                sampled_xyz = xyz
                sampled_features = features

        processed_xyz_list.append(sampled_xyz)
        processed_feature_list.append(sampled_features)

    # Convert a list to a tensor and return (batch_size, num_points, C)
    processed_xyz = torch.stack(processed_xyz_list, dim=0)
    processed_feature = torch.stack(processed_feature_list, dim=0)

    return processed_xyz, processed_feature

def filter_and_sample_points(image_features, pcd, sqrt_alpha, noisy_trajectory, num_points, bs, bounds, pcd_bound_masks=None):
    # Flatten image features from different views into a single tensor
    flat_image_features = einops.rearrange(image_features, "(bt ncam) c h w -> bt (h w ncam) c", ncam=4)
    # Flatten point cloud data
    pcd_flat = pcd.permute(0, 3, 4, 1, 2).reshape(bs, -1, 3)
    if pcd_bound_masks is not None:
        pcd_bound_masks_flat = torch.cat([
            p.reshape(bs, -1) for p in pcd_bound_masks
        ], 1) # Shape: [bs, H * W * n_cameras]

    xyz = pcd_flat.permute(0, 2, 1) # (bs, 3, N)
    feature = flat_image_features.permute(0, 2, 1) # (bs, C, N)

    # Filter points outside of bounds
    all_indices = []
    for i in range(bs):
        if pcd_bound_masks is not None:
            indices = pcd_bound_masks_flat[i].nonzero().squeeze(-1)
        else:
            indices = (
                (xyz[i, 0, :] >= bounds[0]) *
                (xyz[i, 1, :] >= bounds[1]) *
                (xyz[i, 2, :] >= bounds[2]) *
                (xyz[i, 0, :] <= bounds[3]) *
                (xyz[i, 1, :] <= bounds[4]) *
                (xyz[i, 2, :] <= bounds[5])
            ).nonzero().squeeze(-1)

        all_indices.append(indices)
    
    indices_len = [len(indices) for indices in all_indices]
    for i in range(bs):
        num_pad = np.max(indices_len) - indices_len[i]
        if np.max(indices_len) == 0:
            # all batch of points are outside of bounds
            all_indices[i] = torch.arange(xyz.shape[2]).to(xyz.device)
        else:
            if indices_len[i] > 0:
                all_indices[i] = torch.cat([
                    all_indices[i], all_indices[i][np.random.randint(indices_len[i], size=num_pad)]
                    ], dim=0)
            else:
                # all of points are outside of bounds
                all_indices[i] = torch.randperm(xyz.shape[2])[:num_pad].to(xyz.device)

    # Gather points and features based on the indices
    xyz_list, feature_list = [], []
    for i in range(bs):
        xyz_list.append(xyz[i,:,:].index_select(dim=-1, index=all_indices[i]).unsqueeze(dim=0))
        feature_list.append(feature[i,:,:].index_select(dim=-1, index=all_indices[i]).unsqueeze(dim=0))
    xyz = torch.cat(xyz_list, dim=0)
    feature = torch.cat(feature_list, dim=0)

    xyz = xyz.permute(0, 2, 1) # (bs, N, 3)
    feature = feature.permute(0, 2, 1)

    distances = torch.norm(xyz - noisy_trajectory[:, :, :3], dim=-1)
    # sqrt_alpha = torch.clamp(sqrt_alpha, min=0.3).squeeze(1)
    mask = (distances < 1.2) # 1.3 
    filtered_pcd = xyz * mask.unsqueeze(-1)
    filtered_feature = feature * mask.unsqueeze(-1)
    filtered_pcd = [pcd[mask[i]] for i, pcd in enumerate(filtered_pcd)]
    filtered_feature = [feature[mask[i]] for i, feature in enumerate(filtered_feature)]
    sampled_xyz, sampled_features = process_xyz_with_fps_and_nearest(filtered_pcd, filtered_feature, xyz, feature, sqrt_alpha, num_points=num_points)

    return sampled_xyz, sampled_features 

