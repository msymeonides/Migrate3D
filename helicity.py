import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev

def compute_spline(object_data):
    x = object_data[:, 2].astype(float)
    y = object_data[:, 3].astype(float)
    z = object_data[:, 4].astype(float)
    time_raw = object_data[:, 1].astype(float)

    pos_raw = np.stack([x, y, z], axis=1)
    num_points = len(time_raw)
    u_raw = np.linspace(0, 1, len(time_raw))
    smoothing = len(time_raw) * 0.1
    tck, u = splprep(pos_raw.T, u=u_raw, s=smoothing)
    u_fine = np.linspace(0, 1, num_points)
    pos_spline = np.array(splev(u_fine, tck)).T
    total_time = time_raw[-1] - time_raw[0]
    dt = total_time / (num_points - 1)
    return pos_spline, dt

def compute_metrics(object_data, object_id):
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
        'Category': '',
        'Mean Helicity': mean_helicity,
        'Median Helicity': median_helicity,
        'Mean Curvature': mean_curvature,
        'Median Curvature': median_curvature
    }

def compute_helicity_analysis(arr_segments, arr_cats, parameters):
    min_timepoints = parameters['moving']
    results = []
    unique_objects = np.unique(arr_segments[:, 0]).astype(int)

    if arr_cats.size > 0:
        cat_dict = dict(zip(arr_cats[:, 0].astype(int), arr_cats[:, 1].astype(str)))
    else:
        cat_dict = {}

    for obj_id in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == obj_id, :]

        if len(object_data) < min_timepoints:
            metrics = {
                'Object ID': int(obj_id),
                'Category': cat_dict.get(obj_id, '0'),
                'Mean Helicity': np.nan,
                'Median Helicity': np.nan,
                'Mean Curvature': np.nan,
                'Median Curvature': np.nan
            }
            results.append(metrics)
            continue

        metrics = compute_metrics(object_data, obj_id)
        metrics['Category'] = cat_dict.get(obj_id, '0')
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    column_order = ['Object ID', 'Category', 'Mean Helicity', 'Median Helicity', 'Mean Curvature', 'Median Curvature']
    results_df = results_df[column_order]

    metric_columns = ['Mean Helicity', 'Median Helicity', 'Mean Curvature', 'Median Curvature']
    categories = sorted(results_df['Category'].unique())

    mean_data = {}
    median_data = {}
    std_data = {}

    for category in categories:
        category_data = results_df[results_df['Category'] == category]
        mean_data[category] = category_data[metric_columns].mean()
        median_data[category] = category_data[metric_columns].median()
        std_data[category] = category_data[metric_columns].std()

    return results_df
