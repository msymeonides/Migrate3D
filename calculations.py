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

    idx = np.arange(num_rows)

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

    coords = np.stack([x, y, z], axis=1)

    lag_idx = np.arange(1, tau + 1)[:, None]
    valid = idx[None, :] >= lag_idx
    prev_idx = idx[None, :] - lag_idx
    curr_coords = coords[None, :, :]
    prev_coords = coords[prev_idx.clip(min=0), :]
    diffs = np.where(valid[..., None], curr_coords - prev_coords, 0)
    dists = np.linalg.norm(diffs, axis=2) * valid
    euclid_array = dists

    angle_steps = np.arange(1, tau + 1).tolist()
    angle_medians = []

    for i, step in enumerate(angle_steps):
        col = np.full(num_rows, np.nan)
        if num_rows > step:
            col[step:] = euclid_array[step - 1, step:]
        data[f'Euclid {step}'] = col

    for step in angle_steps:
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

        data[f'Turning Angle {step}'] = angles
        if np.any(~np.isnan(angles)):
            angle_medians.append(np.nanmedian(angles))
        else:
            angle_medians.append(np.nan)

    df_object_calcs = pd.DataFrame(data)
    angle_medians_dict = {int(step): angle_medians[i] for i, step in enumerate(angle_steps)}

    return df_object_calcs, angle_steps, angle_medians_dict