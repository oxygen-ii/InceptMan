from monai.transforms import SaveImage
from monai.utils import first, set_determinism
import numpy as np
import onnxruntime
from tqdm import tqdm
import glob
import shutil
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import torch

def surface_recall(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    class_thresholds: list[float],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    spacing: int | float | None = None,
    use_subvoxels: bool = False,
) -> torch.Tensor:

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(
            f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y)."
        )

    batch_size, n_class = y_pred.shape[:2]

    if n_class != len(class_thresholds):
        raise ValueError(
            f"number of classes ({n_class}) does not match number of class thresholds ({len(class_thresholds)})."
        )

    if any(~np.isfinite(class_thresholds)):
        raise ValueError("All class thresholds need to be finite.")

    if any(np.array(class_thresholds) < 0):
        raise ValueError("All class thresholds need to be >= 0.")

    nsd = torch.empty((batch_size, n_class), device=y_pred.device, dtype=torch.float)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas = get_edge_surface_distance(  # type: ignore
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            use_subvoxels=use_subvoxels,
            symmetric=True,
            class_index=c,
        )
        boundary_correct: int | torch.Tensor | float
        boundary_complete: int | torch.Tensor | float
        if not use_subvoxels:
            #boundary_complete = len(distances_pred_gt) 
            boundary_complete = len(distances_gt_pred)
            boundary_correct = torch.sum(distances_pred_gt <= class_thresholds[c]) + torch.sum(
                distances_gt_pred <= class_thresholds[c]
            )
        else:
            areas_pred, areas_gt = areas  # type: ignore
            areas_gt, areas_pred = areas_gt[edges_gt], areas_pred[edges_pred]
            boundary_complete = areas_gt.sum() + areas_pred.sum()
            gt_true = areas_gt[distances_gt_pred <= class_thresholds[c]].sum() if len(areas_gt) > 0 else 0.0
            pred_true = areas_pred[distances_pred_gt <= class_thresholds[c]].sum() if len(areas_pred) > 0 else 0.0
            boundary_correct = gt_true + pred_true
        if boundary_complete == 0:
            # the class is neither present in the prediction, nor in the reference segmentation
            nsd[b, c] = torch.tensor(np.nan)
        else:
            nsd[b, c] = boundary_correct / (2 * boundary_complete)

    return nsd

def asymmetric_hausdorff_distance_voxel(source_voxel, target_voxel):
    """
    Compute Asymmetric Hausdorff Distance (one-way) between two binary voxel grids using KD-Tree.

    :param source_voxel: (D, H, W) binary numpy array representing the source voxel grid
    :param target_voxel: (D, H, W) binary numpy array representing the target voxel grid
    :return: One-way Hausdorff Distance (scalar)
    """
    # Extract coordinates of occupied voxels (value == 1)
    source_points = np.argwhere(source_voxel > 0)
    target_points = np.argwhere(target_voxel > 0)

    # If either source or target is empty, return a large distance
    if len(source_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Build KD-Tree for fast nearest-neighbor lookup
    tree = cKDTree(target_points)

    # Find the nearest neighbor in the target for each source point
    distances, _ = tree.query(source_points)

    # Return the max of these minimum distances (one-way Hausdorff distance)
    return np.percentile(distances, 95)
    #return np.max(distances)

def extract_skull_surface(data):
    depth, height, width = data.shape
    out = np.zeros_like(data)
    for x in range(depth):
        for y in range(height):
            # หาว่าใน slice data[x, y, :] มีตำแหน่งใดเป็น skull = 1 บ้าง
            sk = np.where(data[x, y, :] == 1)[0]
            if len(sk) > 0:
                # เอาจุดแรกกับจุดสุดท้ายเป็น "ขอบ" ของ skull
                out[x, y, sk[0]]  = 1
                out[x, y, sk[-1]] = 1
    for x in range(depth):
        for z in range(width):
            sk = np.where(data[x, :, z] == 1)[0]
            if len(sk) > 0:
                out[x, sk[0],  z] = 1
                out[x, sk[-1], z] = 1
    for y in range(height):
        for z in range(width):
            sk = np.where(data[:, y, z] == 1)[0]
            if len(sk) > 0:
                out[sk[0],  y, z] = 1
                out[sk[-1], y, z] = 1
    return out