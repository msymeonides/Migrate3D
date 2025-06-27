import numpy as np
import pandas as pd

def calculations(object_data, tau, object_id, parameters):
    num_rows, _ = object_data.shape

    instantaneous_displacement = np.zeros(num_rows)
    path_length = np.zeros(num_rows)
    instantaneous_velocity = np.zeros(num_rows)
    instantaneous_acceleration = np.zeros(num_rows)
    instantaneous_velocity_filtered = np.zeros(num_rows)
    instantaneous_acceleration_filtered = np.zeros(num_rows)

    timelapse = parameters['timelapse']
    arrest_limit = parameters['arrest_limit']

    x = object_data[:, 2].astype(float)
    y = object_data[:, 3].astype(float)
    z = object_data[:, 4].astype(float)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    disp = np.sqrt(dx**2 + dy**2 + dz**2)
    instantaneous_displacement[1:] = disp
    path_length[1:] = np.cumsum(disp)
    total_displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2 + (z - z[0])**2)
    instantaneous_velocity[1:] = instantaneous_displacement[1:] / timelapse
    instantaneous_acceleration[1:] = np.concatenate(([0], np.diff(instantaneous_velocity[1:]) / timelapse))
    filtered_mask = instantaneous_displacement > arrest_limit
    instantaneous_velocity_filtered[filtered_mask] = instantaneous_velocity[filtered_mask]
    acc_diff = np.diff(instantaneous_velocity, prepend=instantaneous_velocity[0])
    instantaneous_acceleration_filtered[filtered_mask] = acc_diff[filtered_mask] / timelapse

    coords = np.stack([x, y, z], axis=1)

    idx = np.arange(num_rows)
    lag_idx = np.arange(1, tau + 1)[:, None]
    valid = idx[None, :] >= lag_idx
    prev_idx = idx[None, :] - lag_idx
    curr_coords = coords[None, :, :]
    prev_coords = coords[prev_idx.clip(min=0), :]
    diffs = np.where(valid[..., None], curr_coords - prev_coords, 0)
    dists = np.linalg.norm(diffs, axis=2) * valid
    euclid_array = dists

    angle_steps = np.arange(1, tau + 1, 2)
    num_tps = len(angle_steps)
    idx = np.arange(num_rows)
    step_idx = angle_steps[:, None]
    valid_idx = idx[None, :] >= step_idx * 2

    curr = coords[None, :, :]
    back_idx = (idx[None, :] - step_idx).clip(min=0)
    backs_idx = (idx[None, :] - step_idx * 2).clip(min=0)
    back = coords[back_idx, :]
    backs = coords[backs_idx, :]

    vec0 = curr - back
    vec1 = curr - backs
    norm0 = np.linalg.norm(vec0, axis=2)
    norm1 = np.linalg.norm(vec1, axis=2)
    valid_norm = (norm0 > 0) & (norm1 > 0) & valid_idx

    vec0_norm = np.zeros_like(vec0)
    vec1_norm = np.zeros_like(vec1)
    vec0_norm[valid_norm] = vec0[valid_norm] / norm0[valid_norm, None]
    vec1_norm[valid_norm] = vec1[valid_norm] / norm1[valid_norm, None]

    dot_val = np.clip(np.sum(vec0_norm * vec1_norm, axis=2), -1.0, 1.0)
    angle_rad = np.arccos(dot_val)
    angle_deg = np.degrees(angle_rad)
    angle_array = np.zeros((num_tps, num_rows))
    angle_array[valid_norm] = angle_deg[valid_norm]

    arrest_multipliers = np.arange(1, num_tps + 1)[:, None]
    mask = (norm0 > arrest_limit * arrest_multipliers) & (norm1 > arrest_limit * arrest_multipliers) & valid_norm
    filtered_angle_array = np.zeros((num_tps, num_rows))
    filtered_angle_array[mask] = np.abs(angle_deg[mask])

    object_calcs = {
        'Object ID': object_id,
        'Time': object_data[:, 1],
        'Instantaneous Displacement': instantaneous_displacement,
        'Total Displacement': total_displacement,
        'Path Length': path_length,
        'Instantaneous Velocity': instantaneous_velocity,
        'Instantaneous Acceleration': instantaneous_acceleration,
        'Instantaneous Velocity Filtered': instantaneous_velocity_filtered,
        'Instantaneous Acceleration Filtered': instantaneous_acceleration_filtered,
    }
    new_cols = {}
    for i, arr in enumerate(euclid_array, 1):
        new_cols[f'Euclid {i} TP'] = arr
    for i, arr in enumerate(angle_array, 3):
        new_cols[f'Angle {3 + 2 * i} TP'] = arr
    for i, arr in enumerate(filtered_angle_array, 3):
        new_cols[f'Filtered Angle {3 + 2 * i} TP'] = arr

    df_object_calcs = pd.DataFrame({**object_calcs, **new_cols})

    return df_object_calcs

