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

def interpolate_lazy(arr_segments, timelapse_interval):
    object_data_dict = {}
    for row in arr_segments:
        object_id, timepoint, x, y, z = row
        if object_id not in object_data_dict:
            object_data_dict[object_id] = []
        object_data_dict[object_id].append([float(timepoint), float(x), float(y), float(z)])

    interpolated_data = []
    for object_id in object_data_dict:
        timepoint_data = sorted(object_data_dict[object_id], key=lambda r: r[0])
        times = [tp[0] for tp in timepoint_data]
        min_time, max_time = min(times), max(times)
        expected_times = np.arange(min_time, max_time + timelapse_interval, timelapse_interval)
        time_to_row = {tp[0]: tp for tp in timepoint_data}

        for t in expected_times:
            if t in time_to_row:
                interpolated_data.append([object_id, t, *time_to_row[t][1:]])
            else:
                prev_idx = max(i for i, tp in enumerate(times) if tp < t)
                next_idx = min(i for i, tp in enumerate(times) if tp > t)
                t0, x0, y0, z0 = timepoint_data[prev_idx]
                t1, x1, y1, z1 = timepoint_data[next_idx]
                alpha = (t - t0) / (t1 - t0)
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                z = z0 + alpha * (z1 - z0)
                interpolated_data.append([object_id, t, x, y, z])

    arr_segments_interpolated = np.array(interpolated_data)
    return arr_segments_interpolated

def remove_tracks_with_gaps(arr_segments, unique_objects, timelapse_interval):
    filtered_segments = []
    for obj in unique_objects:
        obj_rows = arr_segments[arr_segments[:, 0] == obj]
        times = np.array(obj_rows[:, 1], dtype=float)
        times_sorted = np.sort(times)
        diffs = np.diff(times_sorted)
        if np.allclose(diffs, timelapse_interval):
            filtered_segments.append(obj_rows)
    if filtered_segments:
        return np.vstack(filtered_segments)
    else:
        return np.empty((0, arr_segments.shape[1]))
