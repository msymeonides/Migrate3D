from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd


def calculate_euclidean(coords, tau):
    num_rows = len(coords)
    euclid_data = {}

    for lag in range(1, tau + 1):
        if num_rows > lag:
            curr_coords = coords[lag:]
            prev_coords = coords[:-lag]
            distances = np.linalg.norm(curr_coords - prev_coords, axis=1)

            col = np.full(num_rows, np.nan)
            col[lag:] = distances
            euclid_data[f'Euclid {lag}'] = col
        else:
            euclid_data[f'Euclid {lag}'] = np.full(num_rows, np.nan)

    return euclid_data


def calculate_angles(coords, tau):
    num_rows = len(coords)
    angle_data = {}
    angle_medians = []

    for step in range(1, tau + 1):
        angles = np.full(num_rows, np.nan)
        max_t = num_rows - 2 * step

        if max_t > 0:
            t_indices = np.arange(max_t)
            v1 = coords[t_indices + step] - coords[t_indices]
            v2 = coords[t_indices + 2 * step] - coords[t_indices + step]

            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            valid_mask = (norm1 > 0) & (norm2 > 0)

            if np.any(valid_mask):
                dot_products = np.sum(v1[valid_mask] * v2[valid_mask], axis=1)
                normalized_dots = np.clip(dot_products / (norm1[valid_mask] * norm2[valid_mask]), -1.0, 1.0)
                angles[t_indices[valid_mask]] = np.degrees(np.arccos(normalized_dots))

        angle_data[f'Turning Angle {step}'] = angles
        angle_medians.append(np.nanmedian(angles) if np.any(~np.isnan(angles)) else np.nan)

    return angle_data, angle_medians


def calculations(object_data, tau, object_id, parameters):
    num_rows, _ = object_data.shape
    timelapse = parameters['timelapse']
    arrest_limit = parameters['arrest_limit']

    coords = object_data[:, 2:5].astype(float)
    diffs = np.diff(coords, axis=0)
    disp = np.linalg.norm(diffs, axis=1)

    instantaneous_displacement = np.zeros(num_rows)
    instantaneous_displacement[1:] = disp

    path_length = np.zeros(num_rows)
    path_length[1:] = np.cumsum(disp)

    total_displacement = np.linalg.norm(coords - coords[0], axis=1)

    instantaneous_velocity = np.zeros(num_rows)
    instantaneous_velocity[1:] = disp / timelapse

    instantaneous_acceleration = np.zeros(num_rows)
    instantaneous_acceleration[2:] = np.diff(instantaneous_velocity[1:]) / timelapse

    filtered_mask = instantaneous_displacement > arrest_limit
    instantaneous_velocity_filtered = np.where(filtered_mask, instantaneous_velocity, 0)
    acc_diff = np.diff(instantaneous_velocity, prepend=instantaneous_velocity[0])
    instantaneous_acceleration_filtered = np.where(filtered_mask, acc_diff / timelapse, 0)

    data = {
        'Object ID': [object_id] * num_rows,
        'Time': object_data[:, 1],
        'Instantaneous Displacement': instantaneous_displacement,
        'Total Displacement': total_displacement,
        'Path Length': path_length,
        'Instantaneous Velocity': instantaneous_velocity,
        'Instantaneous Acceleration': instantaneous_acceleration,
        'Instantaneous Velocity Filtered': instantaneous_velocity_filtered,
        'Instantaneous Acceleration Filtered': instantaneous_acceleration_filtered,
    }

    euclid_data = calculate_euclidean(coords, tau)
    data.update(euclid_data)

    angle_data, angle_medians = calculate_angles(coords, tau)
    data.update(angle_data)

    angle_steps = list(range(1, tau + 1))
    angle_medians_dict = {step: angle_medians[i] for i, step in enumerate(angle_steps)}

    df_object_calcs = pd.DataFrame(data)

    return df_object_calcs, angle_steps, angle_medians_dict


def process_object_calculations(obj, arr_segments, tau, parameters):
    object_data = arr_segments[arr_segments[:, 0] == obj, :]
    object_id = object_data[0, 0]
    df_calcs, angle_steps, angle_medians_dict = calculations(object_data, tau, object_id, parameters)
    return df_calcs, angle_steps, angle_medians_dict, object_id


def calculations_parallel(arr_segments, unique_objects, tau, parameters, n_workers=None):
    max_processes = max(1, min(61, mp.cpu_count() - 1))
    num_workers = n_workers if n_workers is not None else max_processes

    if len(unique_objects) < num_workers * 2:
        all_calcs = []
        all_angle_medians = {}
        all_angle_steps = None

        for obj in unique_objects:
            object_data = arr_segments[arr_segments[:, 0] == obj, :]
            object_id = object_data[0, 0]
            df_calcs, angle_steps, angle_medians_dict = calculations(object_data, tau, object_id, parameters)
            all_calcs.append(df_calcs)
            all_angle_medians[object_id] = angle_medians_dict
            if all_angle_steps is None:
                all_angle_steps = angle_steps

        return all_calcs, all_angle_steps, all_angle_medians

    try:
        with mp.Pool(processes=num_workers) as pool:
            worker_func = partial(process_object_calculations,
                                  arr_segments=arr_segments, tau=tau, parameters=parameters)
            results = pool.map(worker_func, unique_objects)
    except Exception as e:
        print(f"Multiprocessing failed ({e}), falling back to sequential processing...")
        results = [process_object_calculations(obj, arr_segments, tau, parameters) for obj in unique_objects]

    all_calcs = []
    all_angle_medians = {}
    all_angle_steps = None

    for df_calcs, angle_steps, angle_medians_dict, object_id in results:
        all_calcs.append(df_calcs)
        all_angle_medians[object_id] = angle_medians_dict
        if all_angle_steps is None:
            all_angle_steps = angle_steps

    return all_calcs, all_angle_steps, all_angle_medians


if __name__ == '__main__':
    mp.set_start_method("spawn")
