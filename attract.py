import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

distance_threshold = 100
min_speed_attracted = 0.1
max_speed_attractor = 20
time_persistence = 3
timelapse = 4  # Should be set to the actual timelapse interval from main.py
max_gaps = 8  # Number of timepoint gaps allowed before chain is broken
allowed_attractor_types = [5, 6, 8]
allowed_attracted_types = [2, 4]


def detect_attractors(arr_segments, unique_objects, cell_types, df_object_calcs):
    """
    Detects attractor events between objects based on their positions, velocities, and cell types.
    Args:
        arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
        unique_objects (numpy.ndarray): Array of unique object IDs.
        cell_types (dict): Dictionary mapping object IDs to their cell types.
        df_object_calcs (pd.DataFrame): DataFrame containing precomputed velocities.
    Returns:
        list: A list of tuples containing attractor events.
    """
    all_positions = {obj: arr_segments[arr_segments[:, 0] == obj, 2:5] for obj in unique_objects}
    all_times = {obj: arr_segments[arr_segments[:, 0] == obj, 1] for obj in unique_objects}

    # Extract velocities from df_object_calcs
    velocity_dict = {obj: df_object_calcs[df_object_calcs['Object ID'] == obj]['Instantaneous Velocity'].to_numpy() for obj in unique_objects}

    attractor_events = []

    for attractor_id in unique_objects:
        if cell_types.get(attractor_id) not in allowed_attractor_types:
            continue

        mask = arr_segments[:, 0] == attractor_id
        attractor_positions = arr_segments[mask, 2:5]
        attractor_times = arr_segments[mask, 1]

        for other_id in unique_objects:
            if other_id == attractor_id or cell_types.get(other_id) not in allowed_attracted_types:
                continue

            other_positions = all_positions[other_id]
            other_times = all_times[other_id]
            other_velocities = velocity_dict[other_id]

            distances = cdist(other_positions, attractor_positions)
            within_threshold = distances < distance_threshold

            chain = []
            gap_count = 0
            matching_time_idxs = np.where(np.isin(other_times, attractor_times))[0]
            for idx in matching_time_idxs:
                current_time = other_times[idx]
                t_matches = np.where(attractor_times == current_time)[0]
                if t_matches.size == 0:
                    continue
                t_idx = t_matches[0]
                if t_idx >= len(attractor_positions):
                    continue

                attractor_velocity = velocity_dict[attractor_id][t_idx]
                if np.linalg.norm(attractor_velocity) >= max_speed_attractor:
                    chain.clear()
                    gap_count = 0
                    continue

                if not within_threshold[idx, t_idx]:
                    chain.clear()
                    gap_count = 0
                    continue

                direction = other_positions[idx] - attractor_positions[t_idx]
                distance = np.linalg.norm(direction)
                if distance == 0:
                    chain.clear()
                    gap_count = 0
                    continue

                unit_direction = direction / distance
                projection = np.dot(other_velocities[idx], unit_direction)
                if np.any(projection <= min_speed_attracted):
                    chain.clear()
                    gap_count = 0
                    continue

                if not chain:
                    chain.append((current_time, *other_positions[idx], distance, t_idx))
                    gap_count = 0
                else:
                    last_distance = chain[-1][-1]
                    if distance < last_distance:
                        chain.append((current_time, *other_positions[idx], distance, t_idx))
                        gap_count = 0
                    else:
                        if gap_count < max_gaps:
                            chain.append((current_time, *other_positions[idx], distance, t_idx))
                            gap_count += 1
                        else:
                            chain = [(current_time, *other_positions[idx], distance, t_idx)]
                            gap_count = 0

            if len(chain) >= time_persistence:
                start_distance = chain[0][-2]
                end_distance = chain[-2][-2]
                if start_distance > end_distance:
                    attractor_events.append((attractor_id, other_id, chain.copy()))

    return attractor_events


def save_results(attractor_events, output_file, cell_types, df_all_calcs):
    """
    Saves the detected attractor events to a XLSX file.
    Args:
        attractor_events (list): List of tuples containing attractor events.
        output_file (str): Path to the output XLSX file.
        cell_types (dict): Dictionary mapping object IDs to their cell types.
        df_all_calcs (dict): Dictionary containing precomputed velocities.
    """
    rows = []
    for attractor_id, attracted_id, events in attractor_events:
        attractor_type = cell_types.get(attractor_id)
        attracted_type = cell_types.get(attracted_id)

        for event in events:
            time, _, _, _, distance, t_idx = event[:6]
            attractor_velocity = df_all_calcs.loc[(df_all_calcs['Object ID'] == attractor_id) & (df_all_calcs['Time'] == time), 'Instantaneous Velocity'].values[0]
            attracted_velocity = df_all_calcs.loc[(df_all_calcs['Object ID'] == attracted_id) & (df_all_calcs['Time'] == time), 'Instantaneous Velocity'].values[0]

            rows.append([attractor_type, attractor_id, attractor_velocity, attracted_type, attracted_id, attracted_velocity, time, distance])

    df = pd.DataFrame(rows, columns=["Attractor_Type", "Attractor_ID", "Attractor_Velocity", "Attracted_Type", "Attracted_ID", "Attracted_Velocity", "Time", "Distance"])

    # Save to XLSX file with number format for Distance column
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attractors')
        workbook = writer.book
        worksheet = writer.sheets['Attractors']
        numformat = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('C:C', None, numformat)
        worksheet.set_column('F:F', None, numformat)
        worksheet.set_column('H:H', None, numformat)
        wrap_format = workbook.add_format()
        wrap_format.set_text_wrap()

        # Apply wrap format to the header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, wrap_format)


def attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile):
    """
    Main function to detect attractor events and save the results to a XLSX file.
    Args:
        unique_objects (numpy.ndarray): Array of unique object IDs.
        arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
        cell_types (dict): Dictionary mapping object IDs to their cell types.
        df_all_calcs (pd.DataFrame): DataFrame of all calculations.
        savefile (str): Path to the output XLSX file.
    """
    events = detect_attractors(arr_segments, unique_objects, cell_types, df_all_calcs)
    save_attract =  savefile + '_attract.xlsx'
    save_results(events, save_attract, cell_types, df_all_calcs)
