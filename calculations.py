import numpy as np
import pandas as pd

def calculations(object_data, num_euclid_spaces, object_id, parameters):
    num_rows, _ = object_data.shape

    instantaneous_displacement = np.zeros(num_rows)
    path_length = np.zeros(num_rows)
    instantaneous_velocity = np.zeros(num_rows)
    instantaneous_acceleration = np.zeros(num_rows)
    instantaneous_velocity_filtered = np.zeros(num_rows)
    instantaneous_acceleration_filtered = np.zeros(num_rows)

    timelapse = parameters['timelapse']
    arrest_limit = parameters['arrest_limit']

    x = object_data[:, 2]
    y = object_data[:, 3]
    z = object_data[:, 4]

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

    euclid_array = np.zeros((num_euclid_spaces, num_rows))
    for num in range(1, num_euclid_spaces + 1):
        euclid_tp = np.zeros(num_rows)
        for index in range(num, num_rows):
            curr_row = object_data[index]
            prev_row = object_data[index - num]
            euclid_tp[index] = np.sqrt((curr_row[2] - prev_row[2])**2 +
                                       (curr_row[3] - prev_row[3])**2 +
                                       (curr_row[4] - prev_row[4])**2)
        euclid_array[num - 1] = euclid_tp

    mod = 3
    arrest_multiplier = 1
    space = list(range(1, num_euclid_spaces + 1, 2))
    num_tps = len(space)
    angle_array = np.zeros((num_tps, num_rows))
    filtered_angle_array = np.zeros((num_tps, num_rows))

    for back_angle in range(num_tps):
        angle_tp = np.zeros(num_rows)
        filtered_angle_tp = np.zeros(num_rows)
        for index in range(back_angle * 2, num_rows):
            curr = object_data[index]
            back = object_data[index - back_angle] if back_angle > 0 else object_data[index]
            backs = object_data[index - back_angle * 2] if back_angle > 0 else object_data[index]
            vec0 = np.array([curr[2] - back[2], curr[3] - back[3], curr[4] - back[4]])
            vec1 = np.array([curr[2] - backs[2], curr[3] - backs[3], curr[4] - backs[4]])
            norm0 = np.linalg.norm(vec0)
            norm1 = np.linalg.norm(vec1)
            if norm0 == 0 or norm1 == 0:
                continue
            vec0_norm = vec0 / norm0
            vec1_norm = vec1 / norm1
            dot_val = np.clip(np.dot(vec0_norm, vec1_norm), -1.0, 1.0)
            angle_rad = np.arccos(dot_val)
            angle_deg = angle_rad * 180 / np.pi
            angle_tp[index] = angle_deg
            if (norm0 > arrest_limit * arrest_multiplier and
                norm1 > arrest_limit * arrest_multiplier):
                filtered_angle_tp[index] = np.abs(angle_deg)
        angle_array[back_angle] = angle_tp
        filtered_angle_array[back_angle] = filtered_angle_tp
        arrest_multiplier += 1
        mod += 2

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
    df_object_calcs = pd.DataFrame(object_calcs)

    euclid_num = 1
    for arr in euclid_array:
        df_object_calcs['Euclid ' + str(euclid_num) + ' TP'] = arr
        euclid_num += 1

    angle_num = 3
    for arr in angle_array:
        df_object_calcs['Angle ' + str(angle_num) + ' TP'] = arr
        angle_num += 2

    filtered_angle_num = 3
    for arr in filtered_angle_array:
        df_object_calcs['Filtered Angle ' + str(filtered_angle_num) + ' TP'] = arr
        filtered_angle_num += 2

    return df_object_calcs

