import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
import os
import pickle
import shutil


def calculate_euclidean(coords, tau):
    num_rows = len(coords)
    euclid_data = {}

    for lag in range(1, tau + 1):
        if num_rows > lag:
            curr_coords = coords[lag:]
            prev_coords = coords[:-lag]
            distances = np.linalg.norm(curr_coords - prev_coords, axis=1)

            col = np.full(num_rows, np.nan)
            col[lag:] = distances
            euclid_data[f'Euclid {lag}'] = col
        else:
            euclid_data[f'Euclid {lag}'] = np.full(num_rows, np.nan)

    return euclid_data


def calculate_angles(coords, tau):
    num_rows = len(coords)
    angle_data = {}
    angle_medians = []

    for step in range(1, tau + 1):
        angles = np.full(num_rows, np.nan)
        max_t = num_rows - 2 * step

        if max_t > 0:
            t_indices = np.arange(max_t)
            v1 = coords[t_indices + step] - coords[t_indices]
            v2 = coords[t_indices + 2 * step] - coords[t_indices + step]

            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            valid_mask = (norm1 > 0) & (norm2 > 0)

            if np.any(valid_mask):
                dot_products = np.sum(v1[valid_mask] * v2[valid_mask], axis=1)
                normalized_dots = np.clip(dot_products / (norm1[valid_mask] * norm2[valid_mask]), -1.0, 1.0)
                angles[t_indices[valid_mask]] = np.degrees(np.arccos(normalized_dots))

        angle_data[f'Turning Angle {step}'] = angles
        angle_medians.append(np.nanmedian(angles) if np.any(~np.isnan(angles)) else np.nan)

    return angle_data, angle_medians


def calculations(object_data, tau, object_id, parameters):
    num_rows, _ = object_data.shape
    timelapse = parameters['timelapse']
    arrest_limit = parameters['arrest_limit']

    coords = object_data[:, 2:5].astype(float)
    diffs = np.diff(coords, axis=0)
    disp = np.linalg.norm(diffs, axis=1)

    instantaneous_displacement = np.zeros(num_rows)
    instantaneous_displacement[1:] = disp

    path_length = np.zeros(num_rows)
    path_length[1:] = np.cumsum(disp)

    total_displacement = np.linalg.norm(coords - coords[0], axis=1)

    instantaneous_velocity = np.zeros(num_rows)
    instantaneous_velocity[1:] = disp / timelapse

    instantaneous_acceleration = np.zeros(num_rows)
    instantaneous_acceleration[2:] = np.diff(instantaneous_velocity[1:]) / timelapse

    filtered_mask = instantaneous_displacement > arrest_limit
    instantaneous_velocity_filtered = np.where(filtered_mask, instantaneous_velocity, 0)
    acc_diff = np.diff(instantaneous_velocity, prepend=instantaneous_velocity[0])
    instantaneous_acceleration_filtered = np.where(filtered_mask, acc_diff / timelapse, 0)

    data = {
        'Object ID': [object_id] * num_rows,
        'Time': object_data[:, 1],
        'Instantaneous Displacement': instantaneous_displacement,
        'Total Displacement': total_displacement,
        'Path Length': path_length,
        'Instantaneous Velocity': instantaneous_velocity,
        'Instantaneous Acceleration': instantaneous_acceleration,
        'Instantaneous Velocity Filtered': instantaneous_velocity_filtered,
        'Instantaneous Acceleration Filtered': instantaneous_acceleration_filtered,
    }

    euclid_data = calculate_euclidean(coords, tau)
    data.update(euclid_data)

    angle_data, angle_medians = calculate_angles(coords, tau)
    data.update(angle_data)

    angle_steps = list(range(1, tau + 1))
    angle_medians_dict = {step: angle_medians[i] for i, step in enumerate(angle_steps)}

    df_object_calcs = pd.DataFrame(data)

    return df_object_calcs, angle_steps, angle_medians_dict


def process_object_chunk_with_sorting(chunk_info):
    """
    Memory-safe version using shared memory for both data and sorting indices.
    Uses searchsorted to find objects without creating boolean arrays.
    Writes results IMMEDIATELY to disk instead of accumulating in memory.
    """
    (object_chunk, shm_data_name, data_shape, data_dtype,
     shm_sorted_idx_name, sorted_idx_shape, sorted_idx_dtype,
     shm_sorted_obj_name, sorted_obj_shape, sorted_obj_dtype,
     tau, parameters, output_dir, chunk_id) = chunk_info

    # Access shared memory for data
    shm_data = shared_memory.SharedMemory(name=shm_data_name)
    arr_segments = np.ndarray(data_shape, dtype=data_dtype, buffer=shm_data.buf)

    # Access shared memory for sorted indices
    shm_sorted_idx = shared_memory.SharedMemory(name=shm_sorted_idx_name)
    sorted_indices = np.ndarray(sorted_idx_shape, dtype=sorted_idx_dtype, buffer=shm_sorted_idx.buf)

    # Access shared memory for sorted object IDs
    shm_sorted_obj = shared_memory.SharedMemory(name=shm_sorted_obj_name)
    sorted_obj_ids = np.ndarray(sorted_obj_shape, dtype=sorted_obj_dtype, buffer=shm_sorted_obj.buf)

    # Prepare output files
    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")

    chunk_angle_medians = {}
    angle_steps = None
    first_write = True

    for obj in object_chunk:
        # Use searchsorted to find boundaries - NO boolean array allocation!
        start_idx = np.searchsorted(sorted_obj_ids, obj, side='left')
        end_idx = np.searchsorted(sorted_obj_ids, obj, side='right')

        if start_idx < end_idx:
            # Get the original indices for this object
            indices = sorted_indices[start_idx:end_idx]
            object_data = arr_segments[indices]
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            # Write to disk IMMEDIATELY - don't accumulate in memory!
            if first_write:
                df_calcs.to_parquet(output_file, engine='pyarrow', compression='snappy')
                first_write = False
            else:
                # Append to existing parquet file
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, df_calcs], ignore_index=True)
                combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
                del existing_df, combined_df  # Free memory immediately

            del df_calcs  # Free memory immediately

            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

    shm_data.close()
    shm_sorted_idx.close()
    shm_sorted_obj.close()

    # Write angle data
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file if not first_write else None, angle_file


def process_object_chunk_regular_with_sorting(chunk_info):
    """
    Memory-safe version for regular memory (no shared memory).
    Uses searchsorted with pre-sorted data to avoid boolean arrays.
    Writes results IMMEDIATELY to disk instead of accumulating in memory.
    """
    (object_chunk, arr_segments, sorted_indices, sorted_obj_ids,
     tau, parameters, output_dir, chunk_id) = chunk_info

    # Prepare output files
    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")

    chunk_angle_medians = {}
    angle_steps = None
    first_write = True

    for obj in object_chunk:
        # Use searchsorted to find boundaries - NO boolean array allocation!
        start_idx = np.searchsorted(sorted_obj_ids, obj, side='left')
        end_idx = np.searchsorted(sorted_obj_ids, obj, side='right')

        if start_idx < end_idx:
            # Get the original indices for this object
            indices = sorted_indices[start_idx:end_idx]
            object_data = arr_segments[indices]
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            # Write to disk IMMEDIATELY - don't accumulate in memory!
            if first_write:
                df_calcs.to_parquet(output_file, engine='pyarrow', compression='snappy')
                first_write = False
            else:
                # Append to existing parquet file
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, df_calcs], ignore_index=True)
                combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
                del existing_df, combined_df  # Free memory immediately

            del df_calcs  # Free memory immediately

            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

    # Write angle data
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file if not first_write else None, angle_file


def calculations_parallel(arr_segments, unique_objects, tau, parameters, n_workers=None):
    max_processes = max(1, min(61, mp.cpu_count() - 2))
    num_workers = n_workers if n_workers is not None else max_processes

    total_objects = len(unique_objects)

    if total_objects <= num_workers:
        chunk_size = 1
    else:
        chunk_size = max(50, total_objects // num_workers)

    gc.collect()

    object_chunks = []
    for i in range(0, total_objects, chunk_size):
        chunk = unique_objects[i:i + chunk_size]
        object_chunks.append(chunk)

    dataset_mb = arr_segments.nbytes / (1024 * 1024)
    print(f"Processing {total_objects} objects in {len(object_chunks)} chunks of size ~{chunk_size}")
    print(f"Dataset size: {dataset_mb:.1f} MB")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp_calculations")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Using temporary directory: {temp_dir}")

    # Pre-compute sorting arrays ONCE - these will be shared with all workers via shared memory
    # This avoids creating boolean arrays AND avoids pickling huge dictionaries
    print("Pre-computing sort indices for memory-safe object lookup...")
    obj_ids = arr_segments[:, 0]
    sorted_indices = np.argsort(obj_ids)
    sorted_obj_ids = obj_ids[sorted_indices]
    print("Sort computation complete")

    try:
        use_shared_memory = arr_segments.nbytes > 20 * 1024 * 1024

        if use_shared_memory:
            print(f"Using shared memory for large dataset ({dataset_mb:.1f} MB)")

            # Create shared memory for main data
            shm_data = shared_memory.SharedMemory(create=True, size=arr_segments.nbytes)
            shared_array = np.ndarray(arr_segments.shape, dtype=arr_segments.dtype, buffer=shm_data.buf)
            shared_array[:] = arr_segments[:]

            # Create shared memory for sorted indices
            shm_sorted_idx = shared_memory.SharedMemory(create=True, size=sorted_indices.nbytes)
            shared_sorted_idx = np.ndarray(sorted_indices.shape, dtype=sorted_indices.dtype, buffer=shm_sorted_idx.buf)
            shared_sorted_idx[:] = sorted_indices[:]

            # Create shared memory for sorted object IDs
            shm_sorted_obj = shared_memory.SharedMemory(create=True, size=sorted_obj_ids.nbytes)
            shared_sorted_obj = np.ndarray(sorted_obj_ids.shape, dtype=sorted_obj_ids.dtype, buffer=shm_sorted_obj.buf)
            shared_sorted_obj[:] = sorted_obj_ids[:]

            try:
                chunk_infos = []
                for i, chunk in enumerate(object_chunks):
                    chunk_info = (
                        chunk,
                        shm_data.name, arr_segments.shape, arr_segments.dtype,
                        shm_sorted_idx.name, sorted_indices.shape, sorted_indices.dtype,
                        shm_sorted_obj.name, sorted_obj_ids.shape, sorted_obj_ids.dtype,
                        tau, parameters, temp_dir, i
                    )
                    chunk_infos.append(chunk_info)

                actual_workers = min(num_workers, len(object_chunks))
                print(f"Starting pool with {actual_workers} workers...")

                with mp.Pool(processes=actual_workers) as pool:
                    chunk_results = pool.map(process_object_chunk_with_sorting, chunk_infos)
            finally:
                shm_data.close()
                shm_data.unlink()
                shm_sorted_idx.close()
                shm_sorted_idx.unlink()
                shm_sorted_obj.close()
                shm_sorted_obj.unlink()
        else:
            print(f"Using regular memory for dataset ({dataset_mb:.1f} MB)")

            chunk_infos = []
            for i, chunk in enumerate(object_chunks):
                chunk_info = (chunk, arr_segments, sorted_indices, sorted_obj_ids,
                            tau, parameters, temp_dir, i)
                chunk_infos.append(chunk_info)

            actual_workers = min(num_workers, len(object_chunks))
            print(f"Starting pool with {actual_workers} workers...")

            with mp.Pool(processes=actual_workers) as pool:
                chunk_results = pool.map(process_object_chunk_regular_with_sorting, chunk_infos)

        all_calcs = []
        all_angle_medians = {}
        all_angle_steps = None

        print("Reading results from temporary files...")
        for data_file, angle_file in chunk_results:
            with open(angle_file, 'rb') as f:
                chunk_angle_medians, angle_steps = pickle.load(f)
            all_angle_medians.update(chunk_angle_medians)
            if all_angle_steps is None:
                all_angle_steps = angle_steps

            if data_file is not None:
                df_chunk = pd.read_parquet(data_file)
                all_calcs.append(df_chunk)

        return all_calcs, all_angle_steps, all_angle_medians

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")


if __name__ == '__main__':
    mp.set_start_method("spawn")
