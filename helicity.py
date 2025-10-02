import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import warnings

def compute_spline(object_data):
    x = object_data[:, 2].astype(float)
    y = object_data[:, 3].astype(float)
    z = object_data[:, 4].astype(float)
    time_raw = object_data[:, 1].astype(float)

    pos_raw = np.stack([x, y, z], axis=1)
    num_points = len(time_raw)
    u_raw = np.linspace(0, 1, len(time_raw))
    smoothing = len(time_raw) * 0.1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A theoretically impossible result")
        tck, u = splprep(pos_raw.T, u=u_raw, s=smoothing, quiet=1)

    u_fine = np.linspace(0, 1, num_points)
    pos_spline = np.array(splev(u_fine, tck)).T
    total_time = time_raw[-1] - time_raw[0]
    dt = total_time / (num_points - 1)

    return pos_spline, dt

def compute_metrics_single_object(args):
    object_id, object_data, category, min_timepoints = args

    if len(object_data) < min_timepoints:
        return {
            'Object ID': int(object_id),
            'Category': category,
            'Mean Helicity': np.nan,
            'Median Helicity': np.nan,
            'Mean Curvature': np.nan,
            'Median Curvature': np.nan
        }

    try:
        pos, dt = compute_spline(object_data)

        r_dot = np.gradient(pos, dt, axis=0)
        r_ddot = np.gradient(r_dot, dt, axis=0)

        curl_v = np.gradient(np.cross(r_dot[:-1], r_dot[1:]), dt, axis=0)
        helicity_inst = np.einsum('ij,ij->i', r_dot[1:], curl_v) / (np.linalg.norm(r_dot[1:], axis=1)**2 + 1e-8)
        mean_helicity = np.nanmean(helicity_inst)
        median_helicity = np.nanmedian(helicity_inst)

        cross = np.cross(r_dot[:-2], r_ddot[:-2])
        norm_cross = np.linalg.norm(cross, axis=1)
        curvature = norm_cross / (np.linalg.norm(r_dot[:-2], axis=1)**3 + 1e-8)
        mean_curvature = np.nanmean(curvature)
        median_curvature = np.nanmedian(curvature)

        return {
            'Object ID': int(object_id),
            'Category': category,
            'Mean Helicity': mean_helicity,
            'Median Helicity': median_helicity,
            'Mean Curvature': mean_curvature,
            'Median Curvature': median_curvature
        }
    except Exception as e:
        return {
            'Object ID': int(object_id),
            'Category': category,
            'Mean Helicity': np.nan,
            'Median Helicity': np.nan,
            'Mean Curvature': np.nan,
            'Median Curvature': np.nan
        }

def compute_metrics_batch(batch_args):
    results = []
    for args in batch_args:
        result = compute_metrics_single_object(args)
        results.append(result)
    return results

def compute_helicity_analysis(arr_segments, arr_cats, parameters):
    min_timepoints = parameters['moving']
    unique_objects = np.unique(arr_segments[:, 0]).astype(int)

    if arr_cats.size > 0:
        cat_dict = dict(zip(arr_cats[:, 0].astype(int), arr_cats[:, 1].astype(str)))
    else:
        cat_dict = {}

    sorted_indices = np.argsort(arr_segments[:, 0])
    sorted_segments = arr_segments[sorted_indices]

    object_ids = sorted_segments[:, 0].astype(int)
    split_indices = np.where(np.diff(object_ids))[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(object_ids)]))

    object_args = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        obj_id = object_ids[start_idx]
        object_data = sorted_segments[start_idx:end_idx]
        category = cat_dict.get(obj_id, '0')
        object_args.append((obj_id, object_data, category, min_timepoints))

    max_workers = max(1, min(61, mp.cpu_count() - 2))
    batch_size = max(10, len(object_args) // (max_workers * 2))

    batches = []
    for i in range(0, len(object_args), batch_size):
        batch = object_args[i:i + batch_size]
        batches.append(batch)

    results = []

    with mp.Pool(processes=max_workers) as pool:
        batch_results = pool.map(compute_metrics_batch, batches)

        for batch_result in batch_results:
            results.extend(batch_result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    column_order = ['Object ID', 'Category', 'Mean Helicity', 'Median Helicity', 'Mean Curvature', 'Median Curvature']
    results_df = results_df[column_order]
    results_df = results_df.sort_values('Object ID').reset_index(drop=True)

    return results_df

if __name__ == '__main__':
    mp.set_start_method("spawn")
