import numpy as np

def multi_tracking(arr_segments):
    instances = {}
    for row in arr_segments:
        object_id, timepoint, x, y, z = row
        key = (object_id, timepoint)
        if key not in instances:
            instances[key] = [x, y, z, 1]
        else:
            instances[key][0] += x
            instances[key][1] += y
            instances[key][2] += z
            instances[key][3] += 1

    result_data = []
    for key, value in instances.items():
        object_id, timepoint = key
        x_avg = value[0] / value[3]
        y_avg = value[1] / value[3]
        z_avg = value[2] / value[3]
        result_data.append([object_id, timepoint, x_avg, y_avg, z_avg])

    arr_segments_multi = np.array(result_data)
    return arr_segments_multi


def interpolate_lazy(arr_segments, timelapse_interval, unique_objects):
    object_data_dict = {}
    for row in arr_segments:
        object_id, timepoint, x, y, z = row
        if object_id not in object_data_dict:
            object_data_dict[object_id] = []
        object_data_dict[object_id].append([timepoint, x, y, z])

    interpolated_data = []
    for object_id in object_data_dict.keys():
        timepoint_data = object_data_dict[object_id]
        timepoint_data.sort()
        final_tp_index = int(((timepoint_data[-1][0]) / timelapse_interval) - (timepoint_data[1][0] / timelapse_interval) + 2)
        if final_tp_index == len(timepoint_data):
            for i in range(len(timepoint_data)):
                timepoint, x, y, z = timepoint_data[i]
                interpolated_data.append([object_id, timepoint, x, y, z])

        else:
            for i in range(len(timepoint_data)):
                timepoint, x, y, z = timepoint_data[i]
                if [object_id, timepoint, x, y, z] not in interpolated_data:
                    interpolated_data.append([object_id, timepoint, x, y, z])
                elif i == 1:
                    pass

                else:
                    timepoint_prev, x_prev, y_prev, z_prev = timepoint_data[i - 1]
                    time_diff = timepoint - timepoint_prev
                    if time_diff != timelapse_interval and time_diff != 0:
                        x_interp = ((x - x_prev) / 2) + x_prev
                        y_interp = ((y - y_prev) / 2) + y_prev
                        z_interp = ((z - z_prev) / 2) + z_prev
                        time_interp = timepoint_prev + timelapse_interval
                        if time_interp > timepoint_data[-1][0]:
                            pass
                        else:
                            interpolated_data.append([object_id, time_interp, x_interp, y_interp, z_interp])
                            timepoint_data.append((time_interp, x_interp, y_interp, z_interp))
                            timepoint_data.sort()

    arr_segments_interpolated = np.array(interpolated_data)
    return arr_segments_interpolated
