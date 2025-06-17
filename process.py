import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
import dbscan
from joblib import Parallel, delayed
import os
import nibabel as nib
import glob
import time
from scipy.ndimage import label, find_objects
from collections import Counter
import ants
import torchio as tio
import shutil
from pathlib import Path
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

#############################################################################################################
def morphological_erosion_dilation_3d(data, iterations=1):
    struct = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(data, structure=struct, iterations=iterations)
    return eroded.astype(np.uint8)

def fill_vol(volume, i, x):
    indices = np.array(volume == i, dtype=int)
    indices = remove_small_islands_3d(morphological_erosion_dilation_3d(indices, iterations=1), 100)
    x += indices*i
    return x

def remove_small_islands_3d(array, min_size):
    # Label connected components (3D islands)
    labeled_array, num_features = label(array)
    
    # Get the size of each labeled region (island)
    island_sizes = np.bincount(labeled_array.ravel())
    
    # Create a mask for regions larger than or equal to the min_size
    large_island_mask = np.zeros_like(array, dtype=bool)
    
    # Iterate over each labeled region and keep only large islands
    for island_num, size in enumerate(island_sizes):
        if size >= min_size and island_num != 0:  # Ignore background (label 0)
            large_island_mask[labeled_array == island_num] = True
    
    # Apply the mask to remove small islands (set small regions to 0)
    cleaned_array = np.where(large_island_mask, array, 0)
    
    return cleaned_array
#############################################################################################################
def revert_indices(indices, shape):
    # Create an array filled with zeros of the specified shape
    restored_array = np.zeros(shape, dtype=float)
    # Set elements at the given indices to 1.0
    for index in indices:
        restored_array[tuple(index)] = 1.0
    return restored_array

def cal_ratio(indices, defect, healthy):
    d = np.sum(np.logical_and(indices, defect) == 1)
    h = np.sum(np.logical_and(indices, healthy) == 1)
    if d+h ==0:
        return 0
    return d / (d + h)

def find_edges(arr, threshold=0.7):
    arr = np.asarray(arr)
    above = arr >= threshold                     # Boolean mask
    edges = []
    for i in range(len(arr) - 1):
        if above[i] != above[i + 1]:             # Crossing detected
            start = i + 1                        # first index after crossing
            end   = i + 2                        # the next index
            if end < len(arr):                   # stay inside bounds
                edges.append([start, end])
    return edges
#############################################################################################################
def plane_cut_segment_from_planes(volume, plane_position, plane_normal, revert=False):
    # Get the shape of the volume
    shape = volume.shape
    # Create a grid of coordinates and calculate distances to planes
    x, y, z = np.indices(shape)
    #print(0)
    points = np.stack((x, y, z), axis=-1)
    #print(1)
    # Vectorized computation of distances
    distances_plane = point_to_plane_distance(points, plane_position, plane_normal)
    #print(2)
    if revert == False:
        mask = (distances_plane >= 0)
    else:
        mask = (distances_plane <= 0) 
    #print(3)
    # Apply the mask to the volume
    cut_segment = np.where(mask, volume, 0)
    return cut_segment

def point_to_plane_distance(point, plane_position, plane_normal):
    # Calculate the signed distance from a point to a plane (plane form)
    return np.dot(point - plane_position, plane_normal)

def label_data(labels, data):
    # Find unique labels
    unique_labels = np.unique(labels)
    # Initialize the result list
    result = []
    # Group data by labels
    for label in unique_labels:
        group = data[labels == label]
        result.append(group)
    return np.array(result, dtype=object)

def find_closest_index(arrays, x, threshold=5):
    min_distance = float('inf')
    closest_index = -1
    x = np.array(x)
    for idx, array in enumerate(arrays):
        # Calculate the distance between each element in the array and x
        distances = np.linalg.norm(array - x, axis=1)
        # Find the minimum distance in this array
        min_dist_in_array = np.min(distances)
        if min_dist_in_array < min_distance:
            min_distance = min_dist_in_array
            closest_index = idx
        if min_distance < threshold:
            return closest_index
    return closest_index
#############################################################################################################
def step1_regis(moving_file, fixed_file):
    moving = ants.image_read(moving_file).astype('uint8')
    fixed = ants.image_read(fixed_file).astype('uint8')
    outs = ants.registration(fixed, moving, type_of_transform = 'AffineFast')
    warped_img = outs['warpedmovout']
    warped_filename = "*/test_regis.nii.gz"
    ants.image_write(warped_img.astype('uint8'), warped_filename)
    
    nifti_mandible = nib.load(warped_filename)
    volume = nifti_mandible.get_fdata()
    volume = np.round(volume, decimals=2)
    volume_denoise = np.zeros_like(volume)
    for i in range(14):
        fill_vol(volume, i+1, volume_denoise)
    volume_denoise = np.array(volume_denoise, dtype=np.int32)
    return volume_denoise

def step2_selection(template_volume, defect_file):
    nifti_defect = nib.load(defect_file)
    defect_volume = nifti_defect.get_fdata()
    defect_volume = np.round(defect_volume, decimals=2)
    indices_healthy = np.array(defect_volume == 1.0, dtype=int)
    indices_2 = np.argwhere(defect_volume == 2.0)

    # Perform DBSCAN clustering
    labels, core_samples_mask = dbscan.DBSCAN(indices_2, eps=5, min_samples=100)
    
    # Number of clusters in labels(indices_1), ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    most_common_element = Counter(labels).most_common(1)[0][0]
    indices_2 = indices_2[labels == most_common_element]
    labels = labels[labels == most_common_element]
    indices_defect = revert_indices(indices_2, defect_volume.shape)

    template_volume = np.round(template_volume, decimals=2)
    x = []
    for i in range(14):
        x.append(cal_ratio(np.array(template_volume == i+1, dtype=int), indices_defect, indices_healthy))
    return find_edges(x)

def step3_find_hyperplane(region, template_volume):
    template_volume = np.round(template_volume, decimals=2)

    edge_1 = np.argwhere(np.array(template_volume == region[0][0], dtype=int))
    edge_2 = np.argwhere(np.array(template_volume == region[0][1], dtype=int))
    if len(region) > 1:
        edge_4 = np.argwhere(np.array(template_volume == region[1][0], dtype=int))
        edge_3 = np.argwhere(np.array(template_volume == region[1][1], dtype=int))

    positions_SVC_01 = np.vstack((edge_1[1::25], edge_2[1::25]))
    labels_SVC_01 = np.concatenate((np.zeros(len(edge_1[1::25])), np.ones(len(edge_2[1::25]))))

    # Fit the data with an SVM
    svc_01 = SVC(kernel='linear')
    svc_01.fit(positions_SVC_01, labels_SVC_01)

    # Plane coefficients in array coordinates
    A_01, B_01, C_01, D_01 = svc_01.coef_[0][0], svc_01.coef_[0][1], svc_01.coef_[0][2], svc_01.intercept_[0]
    
    w_01 = (-1)*D_01/(A_01**2 + B_01**2 + C_01**2)
    plane1_position = np.array([A_01*w_01, B_01*w_01, C_01*w_01])
    plane1_normal = [A_01, B_01, C_01]

    if len(region) > 1:   
        positions_SVC_02 = np.vstack((edge_3[1::25], edge_4[1::25]))
        labels_SVC_02 = np.concatenate((np.zeros(len(edge_3[1::25])), np.ones(len(edge_4[1::25]))))
    
        # Fit the data with an svm
        svc_02 = SVC(kernel='linear')
        svc_02.fit(positions_SVC_02, labels_SVC_02)
    
        # Plane coefficients in array coordinates
        A_02, B_02, C_02, D_02 = svc_02.coef_[0][0], svc_02.coef_[0][1], svc_02.coef_[0][2], svc_02.intercept_[0]
    
        w_02 = (-1)*D_02/(A_02**2 + B_02**2 + C_02**2)
        plane2_position = np.array([A_02*w_02, B_02*w_02, C_02*w_02])
        plane2_normal = [A_02, B_02, C_02]
        return [[plane1_normal, plane1_position, edge_1[0]], [plane2_normal, plane2_position, edge_3[0]]]
    return [[plane1_normal, plane1_position, edge_1[0]]]

def step4_cutting(plane, mandible_file):
    nifti_mandible = nib.load(mandible_file)
    volume = nifti_mandible.get_fdata()
    volume = remove_small_islands_3d(volume, 1000)
    volume = np.round(volume, decimals=2)
    ###############################################
    cut_segment = plane_cut_segment_from_planes(volume, plane[0][1], plane[0][0])
    indices = np.argwhere(cut_segment == 1.0)
    labels, core_samples_mask = dbscan.DBSCAN(indices, eps=50, min_samples=1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ > 1:
        # Delete FP
        transformed_data = label_data(labels, indices)
        output = find_closest_index(transformed_data, plane[0][2])
        indices = transformed_data[output]
    
    cut_segment_revert = plane_cut_segment_from_planes(volume, plane[0][1], plane[0][0], revert=True)
    indices_revert = np.argwhere(cut_segment_revert == 1.0)
    labels_revert, core_samples_mask_revert = dbscan.DBSCAN(indices_revert, eps=20, min_samples=10)
    labeled_array_revert, num_features_revert = label(cut_segment_revert)
    n_clusters_revert = len(set(labels_revert)) - (1 if -1 in labels_revert else 0)
    if n_clusters_revert > 1:
        transformed_data_revert = label_data(labels_revert, indices_revert)
        output_revert = find_closest_index(transformed_data_revert, plane[0][2])
        indices_revert = transformed_data_revert[int(not(output_revert))]
        indices = np.concatenate([indices, indices_revert])
    restored_array = revert_indices(indices, cut_segment.shape)
    ###############################################
    if len(plane) > 1:
        cut_segment_02 = plane_cut_segment_from_planes(volume, plane[1][1], plane[1][0])
        indices_02 = np.argwhere(cut_segment_02 == 1.0)
        # Perform DBSCAN clustering
        labels_02, core_samples_mask_02 = dbscan.DBSCAN(indices_02, eps=50, min_samples=1)
        n_clusters_02 = len(set(labels_02)) - (1 if -1 in labels_02 else 0)
        if n_clusters_02 > 1:
            # Delete FP
            transformed_data_02 = label_data(labels_02, indices_02)
            output_02 = find_closest_index(transformed_data_02, plane[1][2])
            indices_02 = transformed_data[output_02]
        
        cut_segment_revert_02 = plane_cut_segment_from_planes(volume, plane[1][1], plane[1][0], revert=True)
        indices_revert_02 = np.argwhere(cut_segment_revert_02 == 1.0)
        labels_revert_02, core_samples_mask_revert_02 = dbscan.DBSCAN(indices_revert_02, eps=20, min_samples=10)
        n_clusters_revert_02 = len(set(labels_revert_02)) - (1 if -1 in labels_revert_02 else 0)
        if n_clusters_revert_02 > 1:
            # Transform data
            transformed_data_revert_02 = label_data(labels_revert_02, indices_revert_02)
            output_revert_02 = find_closest_index(transformed_data_revert_02, plane[1][2])
            indices_revert_02 = transformed_data_revert_02[int(not(output_revert_02))]
            indices_02 = np.concatenate([indices_02, indices_revert_02])
        restored_array_02 = revert_indices(indices_02, cut_segment_02.shape)
        healthy_array = np.logical_and(volume, np.logical_not(np.logical_and(restored_array, restored_array_02)))
        healthy_array = remove_small_islands_3d(healthy_array, 2000)
        return healthy_array
    return np.logical_and(volume, np.logical_not(restored_array))

def mandible_cutting(Vhd, V, T):
    #T: template_mandible (.nii.gz)
    #V: binary_volumetric_mandibles (.nii.gz)
    #Vhd: binary_volumetric_healthy_defective_mandible (.nii.gz)
    x = step1_regis(T, V)
    region = step2_selection(x, Vhd)
    p = step3_find_hyperplane(region, x)
    out = step4_cutting(p, V)
    mandible = nib.load(V)
    segment_mandible = nib.Nifti1Image(out, nifti_mandible.affine, nifti_mandible.header)
    return segment_mandible
