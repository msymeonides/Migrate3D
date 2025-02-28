import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

distance_threshold = 200
min_speed_attracted = 0
max_speed_attractor = 50
time_persistence = 6
timelapse = 4  # Should be set to the actual timelapse interval from main.py
max_gaps = 4  # Number of timepoint gaps allowed before chain is broken
allowed_attractor_types = [5, 6, 8]
allowed_attracted_types = [2, 4]

def compute_velocity(positions, times):
    """
        Computes the velocity of objects.
        Args:
            positions (numpy.ndarray): Array of positions with columns [x, y, z].
            times (numpy.ndarray): Array of timepoints.
        Returns:
            numpy.ndarray: Array of velocities with the same number of rows as positions.
        """
    diffs = np.diff(positions, axis=0)
    time_diffs = np.diff(times, axis=0).reshape(-1, 1)
    velocities = diffs / time_diffs
    return np.vstack((velocities, np.zeros((1, 3))))

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
    # Precompute positions, times, and velocities for each object
    obj_mask = {obj: arr_segments[:, 0] == obj for obj in unique_objects}
    velocity_dict = {
        obj: compute_velocity(arr_segments[mask, 2:5], arr_segments[mask, 1])
        for obj, mask in obj_mask.items()
    }
    all_positions = {obj: arr_segments[mask, 2:5] for obj, mask in obj_mask.items()}
    all_times = {obj: arr_segments[mask, 1] for obj, mask in obj_mask.items()}

    attractor_events = []

    for attractor_id in unique_objects:
        # Skip if the attractor cell type is not allowed
        if cell_types.get(attractor_id) not in allowed_attractor_types:
            continue

        mask = arr_segments[:, 0] == attractor_id
        attractor_positions = arr_segments[mask, 2:5]
        attractor_times = arr_segments[mask, 1]

        for other_id in unique_objects:
            # Skip self or if attracted cell type is not allowed
            if other_id == attractor_id or cell_types.get(other_id) not in allowed_attracted_types:
                continue

            other_positions = all_positions[other_id]
            other_times = all_times[other_id]
            other_velocities = velocity_dict[other_id]

            distances = cdist(other_positions, attractor_positions)
            within_threshold = distances < distance_threshold

            chain = []
            gap_count = 0  # Allows up to max_gaps timepoint gaps.
            matching_time_idxs = np.where(np.isin(other_times, attractor_times))[0]
            for idx in matching_time_idxs:
                current_time = other_times[idx]
                t_matches = np.where(attractor_times == current_time)[0]
                if t_matches.size == 0:
                    continue
                t_idx = t_matches[0]
                if t_idx >= len(attractor_positions):
                    continue

                # Check attractor instantaneous velocity.
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
                if projection <= min_speed_attracted:
                    chain.clear()
                    gap_count = 0
                    continue

                if not chain:
                    chain.append((current_time, *other_positions[idx], distance))
                    gap_count = 0
                else:
                    last_distance = chain[-1][-1]
                    if distance < last_distance:
                        chain.append((current_time, *other_positions[idx], distance))
                        gap_count = 0
                    else:
                        if gap_count < max_gaps:
                            chain.append((current_time, *other_positions[idx], distance))  # Append gap
                            gap_count += 1
                        else:
                            chain = [(current_time, *other_positions[idx], distance)]
                            gap_count = 0

            if len(chain) >= time_persistence:
                start_distance = chain[0][-1]
                end_distance = chain[-1][-1]
                if start_distance > end_distance:
                    attractor_events.append((attractor_id, other_id, chain.copy()))

    return attractor_events

def save_results(attractor_events, output_file):
    """
    Saves the detected attractor events to a XLSX file.
    Args:
        attractor_events (list): List of tuples containing attractor events.
        output_file (str): Path to the output XLSX file.
    """
    rows = []
    for attractor_id, other_id, events in attractor_events:
        for time, x, y, z, distance in events:
            rows.append([attractor_id, other_id, time, x, y, z, distance])

    df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z", "Distance"])

    # Save to XLSX file with number format for Distance column
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attractors')
        workbook = writer.book
        worksheet = writer.sheets['Attractors']
        distance_format = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('G:G', None, distance_format)


def attract(unique_objects, arr_segments, cell_types):
    """
        Main function to detect attractor events and save the results to a XLSX file.
        Args:
            unique_objects (numpy.ndarray): Array of unique object IDs.
            arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
            cell_types (dict): Dictionary mapping object IDs to their cell types.
        """
    events = detect_attractors(arr_segments, unique_objects, cell_types)
    save_results(events, "attraction_events.xlsx")
