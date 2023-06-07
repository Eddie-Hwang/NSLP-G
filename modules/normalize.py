import torch
from einops_exts import check_shape


def preprocess_keypoints(keypoints_tensor, **kwargs):
    def normalize_skeleton(keypoints, nfeats):
        if nfeats == 3:
            return normalize_skeleton_3d(keypoints)
        elif nfeats == 2:
            return normalize_skeleton_2d(keypoints)
        else:
            raise ValueError(f"Invalid number of features: {nfeats}")

    check_shape(keypoints_tensor, "f v c")
    nfeats = keypoints_tensor.shape[-1]

    keypoints_tensor = center_keypoints(keypoints_tensor)
    keypoints_tensor = remove_noisy_frames(keypoints_tensor, **kwargs)
    keypoints_tensor = normalize_skeleton(keypoints_tensor, nfeats)
    keypoints_tensor = normalize(keypoints_tensor)

    return keypoints_tensor


def remove_noisy_frames(X, threshold=200.):
    """
    Remove noisy frames based on the Euclidean distance between consecutive frames.

    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints
        threshold: threshold value for the Euclidean distance, frames with distance above the threshold are removed

    Returns:
        X_clean: torch tensor of shape (num_clean_frames, num_joints, 3) representing the cleaned 3D keypoints
    """
    num_frames, num_joints, _ = X.shape
    X_diff = torch.diff(X, dim=0)  # compute the difference between consecutive frames
    distances = torch.norm(X_diff, dim=-1)  # compute the Euclidean distance
    distances = torch.mean(distances, dim=-1)

    mask = torch.ones(num_frames, dtype=torch.bool)  # initialize a mask to keep all frames
    
    mask[1:] = distances <= threshold  # set to False all frames with distance above the threshold
    X_clean = X[mask]  # apply the mask to the input keypoints
    
    return X_clean


def normalize_skeleton_3d(X, resize_factor=None):
    def distance_3d(x1, y1, z1, x2, y2, z2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 3)  # neck

    if resize_factor is None:
        neck_height = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2],
                                  X[:, 0, 0], X[:, 0, 1], X[:, 0, 2]).float()
        shoulder_length = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 2, 0], X[:, 2, 1], X[:, 2, 2]) + \
                          distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 5, 0], X[:, 5, 1], X[:, 5, 2])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor
        normalized_X[:, i, 2] = (X[:, i, 2] - anchor_pt[:, 2]) / resize_factor

    return normalized_X


def center_keypoints(keypoints, joint_idx=1):
    """
    Center the keypoints around a specific joint.
    
    Args:
        keypoints (torch.Tensor): Tensor of shape (batch_size, nframes, njoints, nfeat) or (nframes, njoints, nfeat) containing the keypoints.
        joint_idx (int): Index of the joint to center the keypoints around.
        
    Returns:
        torch.Tensor: Tensor of the same shape as input with the keypoints centered around the specified joint.
    """
    if len(keypoints.shape) == 4:
        batch_size, nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, :, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
    elif len(keypoints.shape) == 3:
        nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(nframes, 1, nfeats)
    else:
        raise ValueError("Input keypoints tensor must have either 3 or 4 dimensions")

    return centered_keypoints


def normalize(X):
    """
    Normalize 3D keypoints using min-max normalization.

    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints

    Returns:
        X_norm: torch tensor of shape (num_frames, num_joints, 3) representing the normalized 3D keypoints
    """
    T, n, d = X.shape
    X = X.reshape(T*n, d)
    X_min = torch.min(X, dim=0)[0]
    X_max = torch.max(X, dim=0)[0]
    X_norm = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
    X_norm = X_norm.reshape(T, n, d)
    
    return X_norm


def normalize_skeleton_2d(X, resize_factor=None):
    def distance_2d(x1, y1, x2, y2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 2)  # neck

    if resize_factor is None:
        neck_height = distance_2d(X[:, 1, 0], X[:, 1, 1],
                                  X[:, 0, 0], X[:, 0, 1]).float()
        shoulder_length = distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 2, 0], X[:, 2, 1]) + \
                          distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 5, 0], X[:, 5, 1])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor

    return normalized_X


def scaling_keypoints(keypoints, width=800, height=900, is_2d=False):
    """
    Unnormalize keypoints by scaling the coordinates based on the width and height
    of the original image.

    Args:
        keypoints: a PyTorch tensor of shape (batch_size, num_frames, num_joints, coord_dim) or (num_frames, num_joints, coord_dim)
            representing keypoints
        width: width of the original image
        height: height of the original image
        is_2d: bool, if True, assumes keypoints are 2D, otherwise assumes keypoints are 3D

    Returns:
        unnormalized_keypoints: a PyTorch tensor of the same shape as keypoints, with each element
        unnormalized based on the width, height, and depth (if 3D) of the original image
    """
    coord_dim = keypoints.shape[-1]
    if coord_dim == 2:
        is_2d = True
    else:
        is_2d = False

    # Scale the x and y coordinates based on the width and height of the original image
    unnormalized_keypoints = keypoints.clone()
    unnormalized_keypoints[..., 0] *= width
    unnormalized_keypoints[..., 1] *= height

    if not is_2d:
        depth = (width + height) * 0.5
        unnormalized_keypoints[..., 2] *= depth

    return unnormalized_keypoints