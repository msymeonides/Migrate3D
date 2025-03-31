import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

distance_threshold = 100  # Maximum distance between attractor and attracted objects to be considered
approach_ratio = 0.5  # Ratio of end distance to start distance must be less than this value
min_proximity = 20  # Attracted objects must get at least this close to attractors for at least one timepoint
time_persistence = 6  # Minimum number of consecutive timepoints for a chain to be included in results
max_gaps = 4  # Number of consecutive timepoints of increasing distance allowed before chain is broken
allowed_attractor_types = [5, 6, 8]  # Cell types allowed to be attractors
allowed_attracted_types = [2, 4]  # Cell types allowed to be attracted


def detect_attractors(arr_segments, unique_objects, cell_types):
    """
    Detects attractor events between objects based on their positions, velocities, and cell types.
    Args:
        arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
        unique_objects (numpy.ndarray): Array of unique object IDs.
        cell_types (dict): Dictionary mapping object IDs to their cell types.
    Returns:
        list: A list of tuples containing attractor events.
    """
    all_positions = {obj: arr_segments[arr_segments[:, 0] == obj, 2:5] for obj in unique_objects}
    all_times = {obj: arr_segments[arr_segments[:, 0] == obj, 1] for obj in unique_objects}

    attractor_events = []

    for attractor_id in unique_objects:
        if cell_types.get(attractor_id) not in allowed_attractor_types:
            continue

        mask = arr_segments[:, 0] == attractor_id
        attractor_positions = arr_segments[mask, 2:5]
        attractor_times = arr_segments[mask, 1]

        time_to_index = {t: i for i, t in enumerate(attractor_times) if i < len(attractor_positions)}

        for attracted_id in unique_objects:
            if attracted_id == attractor_id or cell_types.get(attracted_id) not in allowed_attracted_types:
                continue

            attracted_positions = all_positions[attracted_id]
            attracted_times = all_times[attracted_id]

            distances = cdist(attracted_positions, attractor_positions)
            within_threshold = distances < distance_threshold

            chain = []
            gap_count = 0

            for idx, current_time in enumerate(attracted_times):
                if current_time not in time_to_index:
                    continue
                t_idx = time_to_index[current_time]
                if t_idx >= len(attractor_positions):
                    continue

                if not within_threshold[idx, t_idx]:
                    evaluate_and_clear(chain, attractor_events, attractor_id, attracted_id)
                    gap_count = 0
                    continue

                direction = attracted_positions[idx] - attractor_positions[t_idx]
                distance = np.linalg.norm(direction)
                if distance == 0:
                    evaluate_and_clear(chain, attractor_events, attractor_id, attracted_id)
                    gap_count = 0
                    continue

                if not chain:
                    chain.append((current_time, *attracted_positions[idx], distance, t_idx))
                    gap_count = 0
                else:
                    last_item = chain[-1]
                    last_distance = last_item[-2]
                    if distance < last_distance:
                        chain.append((current_time, *attracted_positions[idx], distance, t_idx))
                        gap_count = 0
                    else:
                        if gap_count < max_gaps:
                            chain.append((current_time, *attracted_positions[idx], distance, t_idx))
                            gap_count += 1
                        else:
                            chain = [(current_time, *attracted_positions[idx], distance, t_idx)]
                            gap_count = 0

            evaluate_and_clear(chain, attractor_events, attractor_id, attracted_id)

            if len(chain) >= time_persistence:
                start_distance = chain[0][-2]
                end_distance = chain[-2][-2]
                if start_distance > end_distance:
                    attractor_events.append((attractor_id, attracted_id, chain.copy()))

    return attractor_events


def evaluate_and_clear(chain, attractor_events, attractor_id, other_id):
    if len(chain) >= time_persistence:
        start_distance = chain[0][-2]
        end_distance = chain[-1][-2]
        min_approach_distance = start_distance * approach_ratio
        if end_distance < min_approach_distance:
            min_distance = min(event[-2] for event in chain)
            if min_distance <= min_proximity:
                attractor_events.append((attractor_id, other_id, chain.copy()))
    chain.clear()


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
            attractor_velocity = df_all_calcs.loc[
                (df_all_calcs['Object ID'] == attractor_id) & (df_all_calcs['Time'] == time),
                'Instantaneous Velocity'
            ].values[0]
            attracted_velocity = df_all_calcs.loc[
                (df_all_calcs['Object ID'] == attracted_id) & (df_all_calcs['Time'] == time),
                'Instantaneous Velocity'
            ].values[0]
            diff_velocity = None
            if attracted_velocity is not None and attractor_velocity is not None:
                diff_velocity = attracted_velocity - attractor_velocity

            rows.append([
                attractor_type, attractor_id, attractor_velocity,
                attracted_type, attracted_id, attracted_velocity,
                time, distance, diff_velocity
            ])

    df = pd.DataFrame(rows, columns=[
        "Attractor_Type", "Attractor_ID", "Attractor_Velocity",
        "Attracted_Type", "Attracted_ID", "Attracted_Velocity",
        "Time", "Distance", "Velocity_Difference"
    ])

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attractors')
        workbook = writer.book
        worksheet = writer.sheets['Attractors']
        numformat = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('C:C', None, numformat)
        worksheet.set_column('F:F', None, numformat)
        worksheet.set_column('H:H', None, numformat)
        worksheet.set_column('I:I', None, numformat)
        wrap_format = workbook.add_format()
        wrap_format.set_text_wrap()
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
    events = detect_attractors(arr_segments, unique_objects, cell_types)
    save_attract = savefile + '_attract.xlsx'
    save_results(events, save_attract, cell_types, df_all_calcs)
