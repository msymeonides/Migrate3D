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


def process_object_chunk_to_file(chunk_info):
    object_chunk, shm_name, arr_shape, arr_dtype, tau, parameters, output_dir, chunk_id = chunk_info

    shm = shared_memory.SharedMemory(name=shm_name)
    arr_segments = np.ndarray(arr_shape, dtype=arr_dtype, buffer=shm.buf)

    chunk_dataframes = []
    chunk_angle_medians = {}
    angle_steps = None

    for obj in object_chunk:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        if len(object_data) > 0:
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            chunk_dataframes.append(df_calcs)
            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

    shm.close()

    if chunk_dataframes:
        combined_df = pd.concat(chunk_dataframes, ignore_index=True)
        output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    else:
        output_file = None

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file, angle_file


def process_object_chunk_to_file_regular(chunk_info):
    object_chunk, arr_segments, tau, parameters, output_dir, chunk_id = chunk_info

    chunk_dataframes = []
    chunk_angle_medians = {}
    angle_steps = None

    for obj in object_chunk:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        if len(object_data) > 0:
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            chunk_dataframes.append(df_calcs)
            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

    if chunk_dataframes:
        combined_df = pd.concat(chunk_dataframes, ignore_index=True)
        output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    else:
        output_file = None

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file, angle_file


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

    print(f"Processing {total_objects} objects in {len(object_chunks)} chunks of size ~{chunk_size}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp_calculations")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Using temporary directory: {temp_dir}")

    try:
        use_shared_memory = arr_segments.nbytes > 20 * 1024 * 1024

        if use_shared_memory:

            print(f"Using shared memory for large dataset ({arr_segments.nbytes / (1024*1024):.1f} MB)")

            shm = shared_memory.SharedMemory(create=True, size=arr_segments.nbytes)
            shared_array = np.ndarray(arr_segments.shape, dtype=arr_segments.dtype, buffer=shm.buf)
            shared_array[:] = arr_segments[:]

            try:
                chunk_infos = []
                for i, chunk in enumerate(object_chunks):
                    chunk_info = (chunk, shm.name, arr_segments.shape, arr_segments.dtype,
                                tau, parameters, temp_dir, i)
                    chunk_infos.append(chunk_info)

                with mp.Pool(processes=num_workers) as pool:
                    chunk_results = pool.map(process_object_chunk_to_file, chunk_infos)
            finally:
                shm.close()
                shm.unlink()
        else:
            print(f"Using regular memory for dataset ({arr_segments.nbytes / (1024*1024):.1f} MB)")

            chunk_infos = []
            for i, chunk in enumerate(object_chunks):
                chunk_info = (chunk, arr_segments, tau, parameters, temp_dir, i)
                chunk_infos.append(chunk_info)

            with mp.Pool(processes=num_workers) as pool:
                chunk_results = pool.map(process_object_chunk_to_file_regular, chunk_infos)

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
