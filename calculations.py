import numpy as np
import pandas as pd


def calculations(cell, x, y, z, list_, num_euclid_spaces, time, parent_id, parameters):
    instantaneous_displacement = [0]
    total_displacement = [0]
    path_length = [0]
    instantaneous_velocity = [0]
    instantaneous_acceleration = [0]
    instantaneous_acceleration_filtered = [0]
    instantaneous_velocity_filtered = [0]
    for index, e in enumerate(x):
        if index == 0:
            pass
        else:
            pl = path_length[index - 1]
            sum_x = (x[index] - x[index - 1]) ** 2
            sum_y = (y[index] - y[index - 1]) ** 2
            sum_z = (z[index] - z[index - 1]) ** 2
            sum_all = sum_x + sum_y + sum_z
            inst_displacement = np.sqrt(sum_all)
            instantaneous_displacement.append(inst_displacement)

            tot_x = (x[index] - x[0]) ** 2
            tot_y = (y[index] - y[0]) ** 2
            tot_z = (z[index] - z[0]) ** 2
            sum_tot = tot_x + tot_y + tot_z
            total_displacement.append(np.sqrt(sum_tot))
            path_length.append(pl + np.sqrt(sum_all))

            instantaneous_velocity.append(instantaneous_displacement[index] / parameters['timelapse'])
            instantaneous_acceleration.append(
                (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / parameters['timelapse'])
            if inst_displacement > parameters['arrest_limit']:
                instantaneous_velocity_filtered.append(instantaneous_displacement[index] / parameters['timelapse'])
                instantaneous_acceleration_filtered.append(
                    (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / parameters['timelapse'])
            else:
                instantaneous_velocity_filtered.append(0)
                instantaneous_acceleration_filtered.append(0)

    df_to_append = pd.DataFrame({parent_id: cell,
                                 'Time': time,
                                 'Instantaneous Displacement': instantaneous_displacement,
                                 'Total Displacement': total_displacement,
                                 'Path length': path_length,
                                 'Instantaneous Velocity': instantaneous_velocity,
                                 'Instantaneous Acceleration': instantaneous_acceleration,
                                 'Instantaneous Velocity Filtered': instantaneous_velocity_filtered,
                                 'Instantaneous Acceleration Filtered': instantaneous_acceleration_filtered,
                                 })

    for back in range(1, num_euclid_spaces + 1):
        euclid = []
        for index, element in enumerate(x):
            if back > index:
                euclid.append(0)
            else:
                x_val = (x[index] - x[index - back]) ** 2
                y_val = (y[index] - y[index - back]) ** 2
                z_val = (z[index] - z[index - back]) ** 2
                euclid.append(np.sqrt(x_val + y_val + z_val))
        df_to_append['Euclidean ' + str(back) + ' TP'] = euclid

    mod = 3
    arrest_multiplier = 1
    space = [s for s in range(num_euclid_spaces + 1) if s % 2 != 0]
    for back_angle in range(1, num_euclid_spaces - len(space) + 1):
        angle = []
        angle_filtered = []
        for index, element in enumerate(x):
            try:
                if back_angle * 2 > index:
                    angle.append(0)
                    angle_filtered.append(0)
                else:
                    cell = cell
                    x_magnitude0 = x[index] - x[index - back_angle]
                    y_magnitude0 = y[index] - y[index - back_angle]
                    z_magnitude0 = z[index] - z[index - back_angle]
                    x_magnitude1 = x[index - back_angle] - x[index - (back_angle * 2)]
                    y_magnitude1 = y[index - back_angle] - y[index - (back_angle * 2)]
                    z_magnitude1 = z[index - back_angle] - z[index - (back_angle * 2)]
                    np.seterr(invalid='ignore')
                    vec_0 = [x_magnitude0, y_magnitude0, z_magnitude0]
                    vec_1 = [x_magnitude1, y_magnitude1, z_magnitude1]
                    vec_0 = vec_0 / np.linalg.norm(vec_0)
                    vec_1 = vec_1 / np.linalg.norm(vec_1)
                    angle_ = np.arccos(np.clip(np.dot(vec_0, vec_1), -1.0, 1.0))
                    angle_ = angle_ * 180 / np.pi
                    angle.append(angle_)

                    x_val0 = (x[index] - x[index - back_angle]) ** 2
                    y_val0 = (y[index] - y[index - back_angle]) ** 2
                    z_val0 = (z[index] - z[index - back_angle]) ** 2

                    x_val1 = (x[index - back_angle] - x[index - (back_angle * 2)]) ** 2
                    y_val1 = (y[index - back_angle] - y[index - (back_angle * 2)]) ** 2
                    z_val1 = (z[index - back_angle] - z[index - (back_angle * 2)]) ** 2
                    euclid_current = np.sqrt(x_val0 + y_val0 + z_val0)
                    euclid_previous = np.sqrt(x_val1 + y_val1 + z_val1)
                    if euclid_current > parameters['arrest_limit'] * arrest_multiplier and euclid_previous > parameters['arrest_limit'] * arrest_multiplier:
                        angle_filtered.append(np.absolute(angle_))
                    else:
                        angle_filtered.append(0)
            except RuntimeWarning:
                pass

        arrest_multiplier += 1
        df_to_append['Angle ' + str(mod) + ' TP'] = angle
        df_to_append['Filtered Angle ' + str(mod) + ' TP'] = angle_filtered
        mod += 2
    list_.append(df_to_append)