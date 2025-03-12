import torch

from einops import rearrange

def flow_warp2(img, fwd_flow, fwd_occ):
    # Flow: nchw -> nhwc
    fwd_flow = rearrange(fwd_flow, 'n c h w -> n h w c')    
    
    n, c, h, w = img.shape
    img = img * (1 - fwd_occ)
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    pos = torch.stack([i, j], dim=2).unsqueeze(0)
    pos = pos.to(device=fwd_flow.device)
    pos2 = torch.round(pos + fwd_flow).to(torch.int64)
    pos2_index = pos2[:, :, :, 0].clamp(0, h - 1) * w + pos2[:, :, :, 1].clamp(0, w - 1)
    pos2_index = pos2_index.unsqueeze(1).repeat(1, c, 1, 1).reshape(n * c, h * w)
    feature_flat = img.reshape(n * c, h * w)
    new_img = torch.zeros_like(feature_flat)
    new_img.scatter_add_(src=feature_flat, dim=1, index=pos2_index)
    new_img = new_img.reshape(n, c, h, w)
    return new_img

def flow_revserse_map(img, flow):
    # Flow: nchw -> nhwc
    flow = rearrange(flow, 'n c h w -> n h w c')    
    
    n, c, h, w = img.shape
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    pos = torch.stack([i, j], dim=2).unsqueeze(0)
    pos = pos.to(device=flow.device)
    pos2 = torch.round(pos + flow).to(torch.int64)
    pos2 = pos2[:, :, :, 0].clamp(0, h - 1) * w + pos2[:, :, :, 1].clamp(0, w - 1)
    pos2 = pos2.repeat(n*c, 1, 1).flatten(1)
    feature_flat = img.reshape(n * c, h * w)
    return torch.gather(feature_flat, 1, pos2).reshape(n, c, h, w)

def get_intermediate_warp_mask(fwd_flow, fwd_occ, alpha):
    
    fwd_flow = fwd_flow * alpha
    fwd_flow_tmp = rearrange(fwd_flow, 'n c h w -> n h w c')    
    n, h, w, _ = fwd_flow_tmp.shape
    
    one_matrix = torch.ones(n, 1, h, w).to(fwd_occ.device)
    one_matrix = one_matrix * (1 - fwd_occ)  
    one_matrix = one_matrix.flatten(1)
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    pos = torch.stack([i, j], dim=2).unsqueeze(0)
    pos = pos.to(device=fwd_flow.device)
    pos2 = torch.round(pos + fwd_flow_tmp).to(torch.int64)
    pos2_index = pos2[:, :, :, 0].clamp(0, h - 1) * w + pos2[:, :, :, 1].clamp(0, w - 1)
    pos2_index = pos2_index.flatten(1)
    
    cnt_matrix = torch.zeros_like(one_matrix)
    cnt_matrix.scatter_add_(src=one_matrix, dim=1, index=pos2_index)
    bwd_occ = cnt_matrix != 1
    bwd_occ = bwd_occ.reshape(n, 1, h, w)
    bwd_occ = bwd_occ.to(fwd_occ.dtype)
    
    fwd_flow = fwd_flow * ((1 - fwd_occ).reshape(n, 1, h, w))
    fwd_flow = fwd_flow.reshape(n * 2, h * w)
    bwd_flow = torch.zeros_like(fwd_flow)
    pos2_index = pos2_index.reshape(n, 1, h, w).repeat(1, 2, 1, 1).reshape(n * 2, h * w)
    bwd_flow.scatter_add_(src=-fwd_flow, dim=1, index=pos2_index)
    bwd_flow = bwd_flow.reshape(n, 2, h, w)
    
    return bwd_flow, bwd_occ
