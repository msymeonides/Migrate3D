import numpy as np
import pandas as pd


def calculations(object, object_data, num_euclid_spaces, object_id, parameters):
    """
    Calculates various migration parameters for a given object's data.
    Args:
        object_data (numpy.ndarray): Array of object data with columns [object_id, timepoint, x, y, z].
        parameters (dict): Dictionary containing user-defined parameters for the analysis.
    Returns:
        pandas.DataFrame: DataFrame containing calculated migration parameters for the object.
    """
    
    # Get number of rows and columns in object data
    num_rows, num_cols = np.shape(object_data)

    # Create empty arrays to hold results
    instantaneous_displacement = np.zeros(num_rows)
    total_displacement = np.zeros(num_rows)
    path_length = np.zeros(num_rows)
    instantaneous_velocity = np.zeros(num_rows)
    instantaneous_acceleration = np.zeros(num_rows)
    instantaneous_acceleration_filtered = np.zeros(num_rows)
    instantaneous_velocity_filtered = np.zeros(num_rows)

    # Perform calculations for each timepoint
    for index in range(num_rows):
        if index == 0:
            pass
        else:
            x0 = object_data[0, 2]
            x_curr = object_data[index, 2]
            x_prev = object_data[(index - 1), 2]
            y0 = object_data[0, 3]
            y_curr = object_data[index, 3]
            y_prev = object_data[(index - 1), 3]
            z0 = object_data[0, 4]
            z_curr = object_data[index, 4]
            z_prev = object_data[(index - 1), 4]
            pl = path_length[index - 1]
            sum_x = (x_curr - x_prev) ** 2
            sum_y = (y_curr - y_prev) ** 2
            sum_z = (z_curr - z_prev) ** 2
            sum_all = sum_x + sum_y + sum_z
            inst_displacement = np.sqrt(sum_all)
            instantaneous_displacement[index] = inst_displacement
            tot_x = (x_curr - x0) ** 2
            tot_y = (y_curr - y0) ** 2
            tot_z = (z_curr - z0) ** 2
            sum_tot = tot_x + tot_y + tot_z
            total_displacement[index] = np.sqrt(sum_tot)
            path_length[index] = pl + np.sqrt(sum_all)

            instantaneous_velocity[index] = instantaneous_displacement[index] / parameters['timelapse']
            instantaneous_acceleration[index] = (
                (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / parameters['timelapse'])
            if inst_displacement > parameters['arrest_limit']:
                instantaneous_velocity_filtered[index] = instantaneous_displacement[index] / parameters['timelapse']
                instantaneous_acceleration_filtered[index] = (
                    (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / parameters['timelapse'])

    # Create empty array for and calculate Euclidian angle
    euclid_array = np.zeros((num_euclid_spaces, num_rows))
    for num in range(1, num_euclid_spaces + 1):
        euclid_tp = np.zeros(num_rows)
        for index in range(num_rows):
            if num > index:
                pass
            else:
                x_curr = object_data[index, 2]
                x_prev = object_data[(index - num), 2]
                y_curr = object_data[index, 3]
                y_prev = object_data[(index - num), 3]
                z_curr = object_data[index, 4]
                z_prev = object_data[(index - num), 4]
                x_val = (x_curr - x_prev) ** 2
                y_val = (y_curr - y_prev) ** 2
                z_val = (z_curr - z_prev) ** 2
                euclid_tp[index] = np.sqrt(x_val + y_val + z_val)
        euclid_array[num - 1] = euclid_tp

    mod = 3
    arrest_multiplier = 1
    space = [s for s in range(1, num_euclid_spaces + 1, 2)]
    num_tps = len(space)

    angle_array = np.zeros((num_tps, num_rows))
    filtered_angle_array = np.zeros((num_tps, num_rows))

    # Calculate angle and filtered angle
    for back_angle in range(num_tps):
        angle_tp = np.zeros(num_rows)
        filtered_angle_tp = np.zeros(num_rows)
        for index in range(num_rows):
            try:
                if back_angle * 2 > index:
                    continue
                else:
                    object = object
                    x_curr = object_data[index, 2]
                    x_back = object_data[(index - back_angle), 2]
                    x_backsq = object_data[(index - (back_angle * 2)), 2]
                    y_curr = object_data[index, 3]
                    y_back = object_data[(index - back_angle), 3]
                    y_backsq = object_data[(index - (back_angle * 2)), 3]
                    z_curr = object_data[index, 4]
                    z_back = object_data[(index - back_angle), 4]
                    z_backsq = object_data[(index - (back_angle * 2)), 4]

                    x_magnitude0 = x_curr - x_back
                    y_magnitude0 = y_curr - y_back
                    z_magnitude0 = z_curr - z_back
                    x_magnitude1 = x_curr - x_backsq
                    y_magnitude1 = y_curr - y_backsq
                    z_magnitude1 = z_curr - z_backsq

                    np.seterr(invalid='ignore')
                    vec_0 = [x_magnitude0, y_magnitude0, z_magnitude0]
                    vec_1 = [x_magnitude1, y_magnitude1, z_magnitude1]
                    vec_0 = vec_0 / np.linalg.norm(vec_0)
                    vec_1 = vec_1 / np.linalg.norm(vec_1)

                    angle_ = np.arccos(np.clip(np.dot(vec_0, vec_1), -1.0, 1.0))
                    angle_ = angle_ * 180 / np.pi

                    angle_tp[index] = angle_

                    x_val0 = x_magnitude0 ** 2
                    y_val0 = y_magnitude0 ** 2
                    z_val0 = z_magnitude0 ** 2
                    x_val1 = x_magnitude1 ** 2
                    y_val1 = y_magnitude1 ** 2
                    z_val1 = z_magnitude1 ** 2

                    euclid_current = np.sqrt(x_val0 + y_val0 + z_val0)
                    euclid_previous = np.sqrt(x_val1 + y_val1 + z_val1)

                    if (euclid_current > parameters['arrest_limit'] * arrest_multiplier and
                            euclid_previous > parameters['arrest_limit'] * arrest_multiplier):
                        filtered_angle_tp[index] = np.abs(angle_)

            except RuntimeWarning:
                pass
        angle_array[back_angle] = angle_tp
        filtered_angle_array[back_angle] = filtered_angle_tp
        arrest_multiplier += 1
        mod += 2

    # Create dictionary of object calculations then convert to DataFrame
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
    df_object_calcs = pd.DataFrame.from_dict(object_calcs)

    # Add Euclidian angle, angle, and filtered angle to calculation DataFrame
    euclid_num = 1
    for x in euclid_array:
        df_object_calcs['Euclid ' + str(euclid_num) + ' TP'] = x
        euclid_num += 1
    angle_num = 3
    for x in angle_array:
        df_object_calcs['Angle ' + str(angle_num) + ' TP'] = x
        angle_num += 2
    filtered_angle_num = 3
    for x in filtered_angle_array:
        df_object_calcs['Filtered Angle ' + str(filtered_angle_num) + ' TP'] = x
        filtered_angle_num += 2

    return df_object_calcs
