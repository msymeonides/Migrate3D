import numpy as np


def multi_tracking(arr_segments):
    instances = {}

    # Get unique combinations of cell ID and timepoint
    for row in arr_segments:
        cell_id, timepoint, x, y, z = row
        key = (cell_id, timepoint)

        # Initialize with first set of coordinates for cell ID and timepoint and begin count
        if key not in instances:
            instances[key] = [x, y, z, 1]

        # If cell ID and timepoint have already been spotted, add the coordinates and increment count
        else:
            instances[key][0] += x
            instances[key][1] += y
            instances[key][2] += z
            instances[key][3] += 1

    # Calculate averages for multitracked timepoints and append add to result data
    result_data = []
    for key, value in instances.items():
        cell_id, timepoint = key
        x_avg = value[0] / value[3]
        y_avg = value[1] / value[3]
        z_avg = value[2] / value[3]
        result_data.append([cell_id, timepoint, x_avg, y_avg, z_avg])

    # Save and return averaged data
    arr_segments_multi = np.array(result_data)
    return arr_segments_multi


def adjust_2D(arr_segments):
    # Set Z coordinate to zero for all timepoints
    arr_segments[:, 4] = 0
    return arr_segments


def interpolate_lazy(arr_segments, timelapse_interval, unique_cells):
    cell_data_dict = {}

    # Create dictionary of cell IDs with list of timepoint, x, y, and z as values
    for row in arr_segments:
        cell_id, timepoint, x, y, z = row
        if cell_id not in cell_data_dict:
            cell_data_dict[cell_id] = []
        cell_data_dict[cell_id].append([timepoint, x, y, z])

    interpolated_data = []

    # For each unique cell, sort sets of timepoint and coordinates by timepoint
    for cell_id in cell_data_dict.keys():
        timepoint_data = cell_data_dict[cell_id]
        timepoint_data.sort()

        # Calculate index of final timepoint
        final_tp_index = int(((timepoint_data[-1][0]) / timelapse_interval) - (timepoint_data[1][0] / timelapse_interval) + 2)

        # If index of final timepoint is equal to number of timepoints for that cell, no points need to be interpolated
        if final_tp_index == len(timepoint_data):
            for i in range(len(timepoint_data)):
                timepoint, x, y, z = timepoint_data[i]
                interpolated_data.append([cell_id, timepoint, x, y, z])

        else:
            for i in range(len(timepoint_data)):
                timepoint, x, y, z = timepoint_data[i]

                # Add each existing timepoint to interpolated data
                if [cell_id, timepoint, x, y, z] not in interpolated_data:
                    interpolated_data.append([cell_id, timepoint, x, y, z])

                # Skip first timepoint
                elif i == 1:
                    pass

                else:
                    # Calculate previous timepoint and different between timepoints
                    timepoint_prev, x_prev, y_prev, z_prev = timepoint_data[i - 1]
                    time_diff = timepoint - timepoint_prev

                    # If difference in timepoints is greater than timelapse interval given, interpolate missing data
                    if time_diff != timelapse_interval and time_diff != 0:
                        x_interp = ((x - x_prev) / 2) + x_prev
                        y_interp = ((y - y_prev) / 2) + y_prev
                        z_interp = ((z - z_prev) / 2) + z_prev
                        time_interp = timepoint_prev + timelapse_interval

                        # Stop interpolating at final timepoint
                        if time_interp > timepoint_data[-1][0]:
                            pass

                        # Add interpolated timepoint to data and sort
                        else:
                            interpolated_data.append([cell_id, time_interp, x_interp, y_interp, z_interp])
                            timepoint_data.append((time_interp, x_interp, y_interp, z_interp))
                            timepoint_data.sort()

    # Create and return array of interpolated data
    arr_segments_interpolated = np.array(interpolated_data)
    return arr_segments_interpolated
