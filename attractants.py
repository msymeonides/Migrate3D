import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Parameters
distance_threshold = 100
speed_threshold = 2
time_persistence = 6
timelapse = 4  # This should be set to the actual timelapse interval from main.py

def compute_velocity(positions, times):
    """Computes velocity vectors from position and time data."""
    diffs = np.diff(positions, axis=0)
    time_diffs = np.diff(times, axis=0).reshape(-1, 1)
    velocities = diffs / time_diffs  # Element-wise division
    return np.vstack((velocities, np.zeros((1, 3))))  # Pad last row

def detect_attractors(arr_segments, unique_objects):
    """Detects attraction events using vectorized operations with minimal loops."""

    # Step 1: Precompute velocities for all cells
    obj_mask = {obj: arr_segments[:, 0] == obj for obj in unique_objects}
    velocity_dict = {
        obj: compute_velocity(arr_segments[mask, 2:5], arr_segments[mask, 1])
        for obj, mask in obj_mask.items()
    }

    # Step 2: Prepare data for vectorized distance computation
    all_positions = {obj: arr_segments[mask, 2:5] for obj, mask in obj_mask.items()}
    all_times = {obj: arr_segments[mask, 1] for obj, mask in obj_mask.items()}

    attractor_events = []

    # Step 3: Compute distances & filter attraction events efficiently
    for attractor_id in unique_objects:
        mask = arr_segments[:, 0] == attractor_id
        attractor_positions = arr_segments[mask, 2:5]  # X, Y, Z
        attractor_times = arr_segments[mask, 1]        # Time

        for other_id in unique_objects:
            if other_id == attractor_id:
                continue

            other_positions = all_positions[other_id]  # (M, 3)
            other_times = all_times[other_id]  # (M,)
            velocities = velocity_dict[other_id]  # Precomputed velocities

            # Compute distances in one go
            distances = cdist(other_positions, attractor_positions)  # (M, N)
            within_threshold = distances < distance_threshold  # Boolean mask

            # Step 4: Find valid attraction chains with NumPy operations
            attraction_chain = []
            matching_time_idxs = np.where(np.isin(other_times, attractor_times))[0]

            for idx in matching_time_idxs:
                current_time = other_times[idx]

                if not np.any(attractor_times == current_time):
                    continue

                # Find closest time index in attractor's time list
                t_match = np.where(attractor_times == current_time)[0]
                if len(t_match) == 0:
                    continue  # Skip if no matching time found

                t_idx = t_match[0]  # Get the first valid index

                if t_idx < len(attractor_positions) and idx < len(attractor_positions) and within_threshold[idx, t_idx]:  # Check if within threshold
                    direction = other_positions[idx] - attractor_positions[t_idx]
                    if np.linalg.norm(direction) > 0:
                        unit_direction = direction / np.linalg.norm(direction)
                        projection = np.dot(velocities[idx], unit_direction)

                    if projection > speed_threshold:
                        attraction_chain.append((current_time, *other_positions[idx]))

                # Check if the chain is consecutive based on the timelapse interval
                if len(attraction_chain) > 1 and (attraction_chain[-1][0] - attraction_chain[-2][0] != timelapse):
                    attraction_chain.clear()

                if len(attraction_chain) >= time_persistence:
                    attractor_events.append((attractor_id, other_id, attraction_chain.copy()))
                    attraction_chain.clear()

    return attractor_events

def save_results(attractor_events, output_file):
    """Saves attraction events to CSV."""
    rows = []
    for attractor_id, other_id, events in attractor_events:
        for time, x, y, z in events:
            rows.append([attractor_id, other_id, time, x, y, z])

    df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z"])
    df.to_csv(output_file, index=False)

def attractants(unique_objects, arr_segments):
    attractor_events = detect_attractors(arr_segments, unique_objects)
    save_results(attractor_events, "attraction_events.csv")