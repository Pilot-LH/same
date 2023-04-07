import torch
import numpy as np

# # Visualize the initial feature map (norms)
# init_feat_norms = init_feat[0].norm(dim=0).detach().cpu().numpy()
# plt.figure(figsize=(10, 10))
# plt.title(f'max: {init_feat_norms.max():.2f}, min: {init_feat_norms.min():.2f}, mean: {init_feat_norms.mean():.2f}, std: {init_feat_norms.std():.2f}')
# plt.imshow(init_feat[0].norm(dim=0).detach().cpu().numpy())
# plt.show()

def normalize_coords(coords, h, w):
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)])
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]
    return grid

def coords_grid(b, h, w, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    stacks = [x, y]
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W]
    
    if device is not None:
        grid = grid.to(device)
    
    return grid


# Modify mask alpha to 1.0
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Modify marker_size to 100
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], 
               color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], 
               color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

    
def local_correlation_softmax(feature0, feature1, local_radius, temperature, padding_mode='zeros'):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y

    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = torch.nn.functional.grid_sample(
        feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True
    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / temperature

    corr[~valid] = -1e9

    prob = torch.nn.functional.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return correspondence