import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

distance_threshold = 100
min_speed_attracted = 1
max_speed_attractor = 8
time_persistence = 4
timelapse = 4  # Should be set to the actual timelapse interval from main.py
allowed_attractor_types = [5, 6, 8]
allowed_attracted_types = [2, 4]

def compute_velocity(positions, times):
    diffs = np.diff(positions, axis=0)
    time_diffs = np.diff(times, axis=0).reshape(-1, 1)
    velocities = diffs / time_diffs
    return np.vstack((velocities, np.zeros((1, 3))))

def detect_attractors(arr_segments, unique_objects, cell_types):
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
            gap_count = 0  # Allows one timepoint gap.
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
                    if len(chain) >= time_persistence:
                        attractor_events.append((attractor_id, other_id, chain.copy()))
                    chain.clear()
                    gap_count = 0
                    continue

                if not within_threshold[idx, t_idx]:
                    # Not within threshold: end chain if already in progress.
                    if len(chain) >= time_persistence:
                        attractor_events.append((attractor_id, other_id, chain.copy()))
                    chain.clear()
                    gap_count = 0
                    continue

                direction = other_positions[idx] - attractor_positions[t_idx]
                distance = np.linalg.norm(direction)
                if distance == 0:
                    if len(chain) >= time_persistence:
                        attractor_events.append((attractor_id, other_id, chain.copy()))
                    chain.clear()
                    gap_count = 0
                    continue

                unit_direction = direction / distance
                projection = np.dot(other_velocities[idx], unit_direction)
                if projection <= min_speed_attracted:
                    if len(chain) >= time_persistence:
                        attractor_events.append((attractor_id, other_id, chain.copy()))
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
                        # Allow a single gap
                        if gap_count == 0:
                            gap_count = 1
                        else:
                            if len(chain) >= time_persistence:
                                attractor_events.append((attractor_id, other_id, chain.copy()))
                            chain = [(current_time, *other_positions[idx], distance)]
                            gap_count = 0
            if len(chain) >= time_persistence:
                attractor_events.append((attractor_id, other_id, chain.copy()))
    return attractor_events

def attract(unique_objects, arr_segments, cell_types):
    events = detect_attractors(arr_segments, unique_objects, cell_types)
    save_results(events, "attraction_events.csv")

def save_results(attractor_events, output_file):
    """Saves attraction events to CSV."""
    rows = []
    for attractor_id, other_id, events in attractor_events:
        for time, x, y, z, distance in events:
            rows.append([attractor_id, other_id, time, x, y, z, distance])

    df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z", "Distance"])
    df.to_csv(output_file, index=False)