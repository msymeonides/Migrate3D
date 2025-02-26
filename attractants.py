# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cdist
#
# # Parameters
# distance_threshold = 200  # Max distance for attraction detection
# speed_threshold = 5  # Min speed for attraction to be considered
# time_persistence = 4  # Min consecutive timepoints for persistent attraction
#
#
# def compute_velocity(positions, times):
#     """Computes velocity vectors from position and time data."""
#     diffs = np.diff(positions, axis=0)
#     time_diffs = np.diff(times, axis=0).reshape(-1, 1)
#     return diffs / time_diffs  # Element-wise division
#
#
# def detect_attractors(arr_segments, unique_objects):
#     """Detects attraction events and returns persistent ones."""
#     attractor_events = []
#
#     for attractor_id in unique_objects:
#         attractor_mask = arr_segments[:, 0] == attractor_id
#         attractor_positions = arr_segments[attractor_mask, 2:4]
#         attractor_times = arr_segments[attractor_mask, 1]
#
#         print(attractor_id)
#
#         for other_id in unique_objects:
#             if other_id == attractor_id:
#                 continue
#
#             other_mask = arr_segments[:, 0] == other_id
#             other_positions = arr_segments[other_mask, 1:4]
#             other_times = arr_segments[other_mask, 4]
#
#             attractor_positions = arr_segments[attractor_mask, 1:4]  # This should give (N, 3)
#             other_positions = arr_segments[other_mask, 1:4]  # Ensure this is also (M, 3)
#
#             velocities = compute_velocity(other_positions, other_times)
#
#             attraction_chain = []
#             for t_idx in range(len(attractor_times) - 1):
#                 current_time = attractor_times[t_idx]
#                 next_time = attractor_times[t_idx + 1]
#
#                 # Find corresponding time in the other cell's track
#                 if current_time in other_times and next_time in other_times:
#                     o_idx = np.where(other_times == current_time)[0][0]
#                     next_o_idx = np.where(other_times == next_time)[0][0]
#
#                     distance = np.linalg.norm(other_positions[o_idx] - attractor_positions[t_idx])
#                     if distance < distance_threshold:
#                         direction = attractor_positions[t_idx + 1] - attractor_positions[t_idx]
#                         unit_direction = direction / np.linalg.norm(direction)
#                         projection = np.dot(velocities[o_idx], unit_direction)
#
#                         if projection > speed_threshold:
#                             attraction_chain.append((current_time, other_positions[o_idx]))
#                         else:
#                             attraction_chain = []
#
#                     if len(attraction_chain) >= time_persistence:
#                         attractor_events.append((attractor_id, other_id, attraction_chain))
#                         attraction_chain = []
#
#     return attractor_events
#
#
# def save_results(attractor_events, output_file):
#     """Saves attraction events to CSV."""
#     rows = []
#     for attractor_id, other_id, events in attractor_events:
#         for time, position in events:
#             rows.append([attractor_id, other_id, time, *position])
#
#     df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z"])
#     df.to_csv(output_file, index=False)
#
#
# # Example usage
# def attractants(unique_objects, arr_segments):
#     attractor_events = []
#     for object_base in unique_objects:
#         attractor_events = detect_attractors(arr_segments, unique_objects)
#
#     save_results(attractor_events, "attraction_events.csv")

#
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cdist
#
# # Parameters
# distance_threshold = 200
# speed_threshold = 5
# time_persistence = 4
#
#
# def compute_velocity(positions, times):
#     diffs = np.diff(positions, axis=0)
#     time_diffs = np.diff(times, axis=0).reshape(-1, 1)
#     return np.vstack((diffs / time_diffs, np.zeros((1, 3))))  # Pad last row
#
#
# def detect_attractors(arr_segments, unique_objects):
#     attractor_events = []
#
#     # Precompute all velocities once
#     velocity_dict = {obj: compute_velocity(arr_segments[arr_segments[:, 0] == obj, 1:4],
#                                            arr_segments[arr_segments[:, 0] == obj, 4])
#                      for obj in unique_objects}
#
#     for attractor_id in unique_objects:
#         attractor_mask = arr_segments[:, 0] == attractor_id
#         attractor_positions = arr_segments[attractor_mask, 1:4]  # (N, 3)
#         attractor_times = arr_segments[attractor_mask, 4]  # (N,)
#
#         print(attractor_id)
#         for other_id in unique_objects:
#             if other_id == attractor_id:
#                 continue
#
#             other_mask = arr_segments[:, 0] == other_id
#             other_positions = arr_segments[other_mask, 1:4]  # (M, 3)
#             other_times = arr_segments[other_mask, 4]  # (M,)
#
#             velocities = velocity_dict[other_id]  # Precomputed velocities
#
#             # Compute all distances at once
#             distances = cdist(other_positions, attractor_positions)  # (M, N)
#             within_threshold = distances < distance_threshold
#
#             attraction_chain = []
#             for t_idx in range(len(attractor_times) - 1):
#                 current_time = attractor_times[t_idx]
#                 next_time = attractor_times[t_idx + 1]
#
#                 time_mask = (other_times == current_time) | (other_times == next_time)
#                 if not np.any(time_mask):
#                     continue
#
#                 o_idxs = np.where(time_mask)[0]
#                 if len(o_idxs) < 2:
#                     continue
#
#                 o_idx, next_o_idx = o_idxs[:2]  # Get first two matching timepoints
#
#                 if within_threshold[o_idx, t_idx]:
#                     direction = attractor_positions[t_idx + 1] - attractor_positions[t_idx]
#                     unit_direction = direction / np.linalg.norm(direction)
#                     projection = np.dot(velocities[o_idx], unit_direction)
#
#                     if projection > speed_threshold:
#                         attraction_chain.append((current_time, other_positions[o_idx]))
#                     else:
#                         attraction_chain = []
#
#                     if len(attraction_chain) >= time_persistence:
#                         attractor_events.append((attractor_id, other_id, attraction_chain))
#                         attraction_chain = []
#
#     return attractor_events
#
#
# def save_results(attractor_events, output_file):
#     rows = []
#     for attractor_id, other_id, events in attractor_events:
#         for time, position in events:
#             rows.append([attractor_id, other_id, time, *position])
#
#     df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z"])
#     df.to_csv(output_file, index=False)
#
#
# def attractants(unique_objects, arr_segments):
#     attractor_events = detect_attractors(arr_segments, unique_objects)
#     save_results(attractor_events, "attraction_events.csv")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Parameters
distance_threshold = 10000000000
speed_threshold = 0
time_persistence = 1


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
        obj: compute_velocity(arr_segments[mask, 1:4], arr_segments[mask, 4])
        for obj, mask in obj_mask.items()
    }

    # Step 2: Prepare data for vectorized distance computation
    all_positions = {obj: arr_segments[mask, 1:4] for obj, mask in obj_mask.items()}
    all_times = {obj: arr_segments[mask, 4] for obj, mask in obj_mask.items()}

    attractor_events = []

    # Step 3: Compute distances & filter attraction events efficiently
    for attractor_id in unique_objects:
        attractor_positions = all_positions[attractor_id]  # (N, 3)
        attractor_times = all_times[attractor_id]  # (N,)

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

            for idx in matching_time_idxs[:-1]:
                current_time, next_time = other_times[idx], other_times[idx + 1]

                if not np.any(attractor_times == current_time):
                    continue

                # Find closest time index in attractor's time list
                t_idx = np.where(attractor_times == current_time)[0][0]

                if within_threshold[idx, t_idx]:  # Check if within threshold
                    direction = attractor_positions[t_idx] - attractor_positions[t_idx]
                    unit_direction = direction / np.linalg.norm(direction)
                    projection = np.dot(velocities[idx], unit_direction)

                    if projection > speed_threshold:
                        attraction_chain.append((current_time, *other_positions[idx]))
                    else:
                        attraction_chain = []

                    if len(attraction_chain) >= time_persistence:
                        attractor_events.append((attractor_id, other_id, attraction_chain))
                        attraction_chain = []

    return attractor_events


def save_results(attractor_events, output_file):
    """Saves attraction events to CSV."""
    rows = []
    print('events')
    print(attractor_events)
    for attractor_id, other_id, events in attractor_events:
        for time, x, y, z in events:
            rows.append([attractor_id, other_id, time, x, y, z])
            print('rows')
            print(rows)

    df = pd.DataFrame(rows, columns=["Attractor_ID", "Attracted_Cell_ID", "Time", "X", "Y", "Z"])
    df.to_csv(output_file, index=False)


def attractants(unique_objects, arr_segments):
    attractor_events = detect_attractors(arr_segments, unique_objects)
    save_results(attractor_events, "attraction_events.csv")

