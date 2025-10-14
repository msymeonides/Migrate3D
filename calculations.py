import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
import os
import pickle
import psutil
import shutil

OBJECT_COUNT_THRESHOLD = 200
MEMORY_SAFETY_THRESHOLD = 0.75

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

    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
    chunk_angle_medians = {}
    angle_steps = None

    writer = None
    try:
        for obj_idx, obj in enumerate(object_chunk):
            object_data = arr_segments[arr_segments[:, 0] == obj, :]
            if len(object_data) > 0:
                object_id = object_data[0, 0]
                df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

                if writer is None:
                    df_calcs.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
                else:
                    existing_df = pd.read_parquet(output_file)
                    combined_df = pd.concat([existing_df, df_calcs], ignore_index=True)
                    combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
                    del existing_df, combined_df

                writer = True

                chunk_angle_medians[object_id] = angle_medians_dict
                if angle_steps is None:
                    angle_steps = angle_steps_temp

                del df_calcs
                gc.collect()
    finally:
        shm.close()

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file if writer else None, angle_file


def process_object_chunk_per_object_file(chunk_info):
    object_chunk, shm_name, arr_shape, arr_dtype, tau, parameters, output_dir, chunk_id = chunk_info

    shm = shared_memory.SharedMemory(name=shm_name)
    arr_segments = np.ndarray(arr_shape, dtype=arr_dtype, buffer=shm.buf)

    chunk_angle_medians = {}
    angle_steps = None
    chunk_dfs = []

    try:
        for obj_idx, obj in enumerate(object_chunk):
            object_data = arr_segments[arr_segments[:, 0] == obj, :]
            if len(object_data) > 0:
                object_id = object_data[0, 0]
                df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

                chunk_dfs.append(df_calcs)

                chunk_angle_medians[object_id] = angle_medians_dict
                if angle_steps is None:
                    angle_steps = angle_steps_temp

                del df_calcs

                if (obj_idx + 1) % 100 == 0:
                    gc.collect()
    finally:
        shm.close()

    if chunk_dfs:
        output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
        chunk_combined = pd.concat(chunk_dfs, ignore_index=True, copy=False)
        del chunk_dfs
        gc.collect()

        chunk_combined.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        output_files = output_file
    else:
        output_files = None

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_files, angle_file


def process_object_chunk_to_file_regular(chunk_info):
    object_chunk, arr_segments, tau, parameters, output_dir, chunk_id = chunk_info

    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
    chunk_angle_medians = {}
    angle_steps = None

    writer = None
    for obj_idx, obj in enumerate(object_chunk):
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        if len(object_data) > 0:
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            if writer is None:
                df_calcs.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
            else:
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, df_calcs], ignore_index=True)
                combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
                del existing_df, combined_df

            writer = True

            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

            del df_calcs
            gc.collect()

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_file if writer else None, angle_file


def process_object_chunk_per_object_file_regular(chunk_info):
    object_chunk, arr_segments, tau, parameters, output_dir, chunk_id = chunk_info

    chunk_angle_medians = {}
    angle_steps = None
    chunk_dfs = []

    for obj_idx, obj in enumerate(object_chunk):
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        if len(object_data) > 0:
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            chunk_dfs.append(df_calcs)

            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

            del df_calcs

            if (obj_idx + 1) % 100 == 0:
                gc.collect()

    if chunk_dfs:
        output_file = os.path.join(output_dir, f"chunk_{chunk_id}_data.parquet")
        chunk_combined = pd.concat(chunk_dfs, ignore_index=True, copy=False)
        del chunk_dfs
        gc.collect()

        chunk_combined.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        output_files = output_file
    else:
        output_files = None

    angle_file = os.path.join(output_dir, f"chunk_{chunk_id}_angles.pkl")
    with open(angle_file, 'wb') as f:
        pickle.dump((chunk_angle_medians, angle_steps), f)

    return output_files, angle_file


def process_object_chunk_in_memory(chunk_info):
    object_chunk, shm_name, arr_shape, arr_dtype, tau, parameters, chunk_id = chunk_info

    shm = shared_memory.SharedMemory(name=shm_name)
    arr_segments = np.ndarray(arr_shape, dtype=arr_dtype, buffer=shm.buf)

    chunk_angle_medians = {}
    angle_steps = None
    chunk_dfs = []

    try:
        for obj in object_chunk:
            object_data = arr_segments[arr_segments[:, 0] == obj, :]
            if len(object_data) > 0:
                object_id = object_data[0, 0]
                df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

                chunk_dfs.append(df_calcs)
                chunk_angle_medians[object_id] = angle_medians_dict
                if angle_steps is None:
                    angle_steps = angle_steps_temp
    finally:
        shm.close()

    if chunk_dfs:
        chunk_combined = pd.concat(chunk_dfs, ignore_index=True, copy=False)
        del chunk_dfs
        gc.collect()
        return chunk_combined, chunk_angle_medians, angle_steps
    else:
        return None, chunk_angle_medians, angle_steps


def process_object_chunk_in_memory_regular(chunk_info):
    object_chunk, arr_segments, tau, parameters, chunk_id = chunk_info

    chunk_angle_medians = {}
    angle_steps = None
    chunk_dfs = []

    for obj in object_chunk:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        if len(object_data) > 0:
            object_id = object_data[0, 0]
            df_calcs, angle_steps_temp, angle_medians_dict = calculations(object_data, tau, object_id, parameters)

            chunk_dfs.append(df_calcs)
            chunk_angle_medians[object_id] = angle_medians_dict
            if angle_steps is None:
                angle_steps = angle_steps_temp

    if chunk_dfs:
        chunk_combined = pd.concat(chunk_dfs, ignore_index=True, copy=False)
        del chunk_dfs
        gc.collect()
        return chunk_combined, chunk_angle_medians, angle_steps
    else:
        return None, chunk_angle_medians, angle_steps


def calculations_parallel(arr_segments, unique_objects, tau, parameters, n_workers=None):
    max_processes = max(1, min(61, mp.cpu_count() - 2))
    num_workers = n_workers if n_workers is not None else max_processes
    total_objects = len(unique_objects)

    mem_est = estimate_memory_requirements(arr_segments, unique_objects, tau)

    use_parquet = mem_est['use_parquet']

    if total_objects <= num_workers:
        chunk_size = 1
    else:
        chunk_size = 50

    gc.collect()

    object_chunks = []
    for i in range(0, total_objects, chunk_size):
        chunk = unique_objects[i:i + chunk_size]
        object_chunks.append(chunk)

    use_shared_memory = arr_segments.nbytes > 20 * 1024 * 1024

    if not use_parquet:
        all_angle_medians = {}
        all_angle_steps = None
        all_calcs = []

        if use_shared_memory:
            shm = shared_memory.SharedMemory(create=True, size=arr_segments.nbytes)
            shared_array = np.ndarray(arr_segments.shape, dtype=arr_segments.dtype, buffer=shm.buf)
            shared_array[:] = arr_segments[:]

            try:
                chunk_infos = [
                    (chunk, shm.name, arr_segments.shape, arr_segments.dtype, tau, parameters, i)
                    for i, chunk in enumerate(object_chunks)
                ]

                with mp.Pool(processes=num_workers) as pool:
                    for i, (df_chunk, angle_medians, angle_steps) in enumerate(pool.imap_unordered(process_object_chunk_in_memory, chunk_infos)):
                        if df_chunk is not None:
                            all_calcs.append(df_chunk)
                        all_angle_medians.update(angle_medians)
                        if all_angle_steps is None:
                            all_angle_steps = angle_steps
            finally:
                shm.close()
                shm.unlink()
        else:
            chunk_infos = [
                (chunk, arr_segments, tau, parameters, i)
                for i, chunk in enumerate(object_chunks)
            ]

            with mp.Pool(processes=num_workers) as pool:
                for i, (df_chunk, angle_medians, angle_steps) in enumerate(pool.imap_unordered(process_object_chunk_in_memory_regular, chunk_infos)):
                    if df_chunk is not None:
                        all_calcs.append(df_chunk)
                    all_angle_medians.update(angle_medians)
                    if all_angle_steps is None:
                        all_angle_steps = angle_steps

        return all_calcs, all_angle_steps, all_angle_medians

    else:
        base_savefile = parameters.get('savefile', None)
        if base_savefile:
            calcs_dir = base_savefile + "_calcs"
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            calcs_dir = os.path.join(project_dir, "temp_calculations")

        if os.path.exists(calcs_dir):
            try:
                shutil.rmtree(calcs_dir)
            except Exception:
                pass
        os.makedirs(calcs_dir, exist_ok=True)

        all_calcs = []
        all_angle_medians = {}
        all_angle_steps = None
        parquet_manifest = []

        try:
            if use_shared_memory:
                shm = shared_memory.SharedMemory(create=True, size=arr_segments.nbytes)
                shared_array = np.ndarray(arr_segments.shape, dtype=arr_segments.dtype, buffer=shm.buf)
                shared_array[:] = arr_segments[:]

                try:
                    chunk_infos = [
                        (chunk, shm.name, arr_segments.shape, arr_segments.dtype, tau, parameters, calcs_dir, i)
                        for i, chunk in enumerate(object_chunks)
                    ]

                    with mp.Pool(processes=num_workers) as pool:
                        chunk_results = pool.map(process_object_chunk_per_object_file, chunk_infos)
                finally:
                    shm.close()
                    shm.unlink()
            else:
                chunk_infos = [
                    (chunk, arr_segments, tau, parameters, calcs_dir, i)
                    for i, chunk in enumerate(object_chunks)
                ]

                with mp.Pool(processes=num_workers) as pool:
                    chunk_results = pool.map(process_object_chunk_per_object_file_regular, chunk_infos)

            for data_files, angle_file in chunk_results:
                with open(angle_file, 'rb') as f:
                    chunk_angle_medians, angle_steps = pickle.load(f)
                all_angle_medians.update(chunk_angle_medians)
                if all_angle_steps is None:
                    all_angle_steps = angle_steps

                if data_files is not None:
                    parquet_manifest.append(data_files)

            parameters['calcs_manifest'] = parquet_manifest
            parameters['calcs_dir'] = calcs_dir

            return all_calcs, all_angle_steps, all_angle_medians

        except Exception:
            try:
                shutil.rmtree(calcs_dir)
            except Exception:
                pass
            raise


def estimate_memory_requirements(arr_segments, unique_objects, tau):
    mem_info = psutil.virtual_memory()
    available_gb = mem_info.available / (1024**3)
    input_size_gb = arr_segments.nbytes / (1024**3)

    total_rows = len(arr_segments)
    output_columns = 9 + (2 * tau)

    bytes_per_cell = 8.5
    estimated_output_bytes = total_rows * output_columns * bytes_per_cell
    estimated_output_gb = estimated_output_bytes / (1024**3)

    peak_multiplier = 2.3 if estimated_output_gb < 0.1 else 2.1
    estimated_peak_gb = estimated_output_gb * peak_multiplier

    use_parquet = estimated_peak_gb > (available_gb * MEMORY_SAFETY_THRESHOLD)

    return {
        'input_size_gb': input_size_gb,
        'estimated_output_size_gb': estimated_output_gb,
        'estimated_peak_gb': estimated_peak_gb,
        'available_gb': available_gb,
        'total_gb': mem_info.total / (1024**3),
        'use_parquet': use_parquet,
        'num_objects': len(unique_objects),
        'total_rows': total_rows
    }


def concatenate_dataframes_memory_safe(df_list, output_file=None, batch_size=50):
    if not df_list:
        return pd.DataFrame()

    if len(df_list) == 1:
        return df_list[0]

    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024**3)

    sample_df = df_list[0]
    avg_memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
    total_rows = sum(len(df) for df in df_list)
    estimated_memory_gb = (total_rows * avg_memory_per_row) / (1024**3)
    estimated_peak_memory_gb = estimated_memory_gb * 2.5

    memory_threshold = 0.5

    if estimated_peak_memory_gb < available_memory_gb * memory_threshold:
        return _concatenate_direct(df_list)
    elif estimated_peak_memory_gb < available_memory_gb * 0.85:
        return _concatenate_incremental_smart(df_list, available_memory_gb, avg_memory_per_row)
    else:
        return _concatenate_chunked_minimal_memory(df_list, available_memory_gb, avg_memory_per_row)


def _concatenate_direct(df_list):
    result = pd.concat(df_list, ignore_index=True, copy=False)
    return result


def _concatenate_incremental_smart(df_list, available_memory_gb, avg_memory_per_row):
    target_batch_memory_gb = available_memory_gb * 0.2
    rows_per_batch = int((target_batch_memory_gb * (1024**3)) / avg_memory_per_row)

    batch_sizes = []
    current_batch_size = 0
    current_batch_rows = 0

    for i, df in enumerate(df_list):
        if current_batch_rows + len(df) > rows_per_batch and current_batch_size > 0:
            batch_sizes.append(current_batch_size)
            current_batch_size = 0
            current_batch_rows = 0
        current_batch_size += 1
        current_batch_rows += len(df)

    if current_batch_size > 0:
        batch_sizes.append(current_batch_size)

    if len(batch_sizes) <= 3:
        return _concatenate_direct(df_list)

    result_dfs = []
    df_idx = 0

    for batch_num, batch_size in enumerate(batch_sizes, 1):
        batch = df_list[df_idx:df_idx + batch_size]
        batch_df = pd.concat(batch, ignore_index=True, copy=False)
        result_dfs.append(batch_df)
        del batch
        df_idx += batch_size
        if batch_num % 3 == 0:
            gc.collect()

    final_df = pd.concat(result_dfs, ignore_index=True, copy=False)
    del result_dfs
    gc.collect()
    return final_df


def _concatenate_chunked_minimal_memory(df_list, available_memory_gb, avg_memory_per_row):
    result = df_list[0].copy()
    batch_size = max(5, int(len(df_list) / 20))

    for i in range(1, len(df_list), batch_size):
        batch = df_list[i:i + batch_size]
        if len(batch) == 1:
            batch_df = batch[0]
        else:
            batch_df = pd.concat(batch, ignore_index=True, copy=False)
        result = pd.concat([result, batch_df], ignore_index=True, copy=False)
        del batch, batch_df
        if (i // batch_size) % 2 == 0:
            gc.collect()

    gc.collect()
    return result


def _concatenate_in_batches(df_list, batch_size):
    print(f"Using legacy batch concatenation with batch_size={batch_size}")
    result_dfs = []

    for i in range(0, len(df_list), batch_size):
        batch = df_list[i:i + batch_size]
        print(f"Concatenating batch {i//batch_size + 1}/{(len(df_list) + batch_size - 1)//batch_size}...")

        batch_df = pd.concat(batch, ignore_index=True, copy=False)
        result_dfs.append(batch_df)

        del batch
        if (i // batch_size) % 3 == 0:
            gc.collect()

    if len(result_dfs) == 1:
        return result_dfs[0]
    else:
        print("Performing final concatenation...")
        final_df = pd.concat(result_dfs, ignore_index=True, copy=False)
        del result_dfs
        gc.collect()
        return final_df


def _concatenate_via_disk(df_list, output_file, batch_size):
    print("WARNING: Using slow disk-based concatenation. This should only happen for extreme memory constraints.")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_concat_dir = os.path.join(project_dir, "temp_concatenation")

    if os.path.exists(temp_concat_dir):
        shutil.rmtree(temp_concat_dir)
    os.makedirs(temp_concat_dir)

    try:
        batch_files = []
        for i in range(0, len(df_list), batch_size):
            batch = df_list[i:i + batch_size]
            print(f"Writing batch {i//batch_size + 1}/{(len(df_list) + batch_size - 1)//batch_size} to disk...")

            batch_df = pd.concat(batch, ignore_index=True, copy=False)
            batch_file = os.path.join(temp_concat_dir, f"batch_{i}.parquet")
            batch_df.to_parquet(batch_file, engine='pyarrow', compression='snappy', index=False)
            batch_files.append(batch_file)

            del batch, batch_df
            gc.collect()

        print("Reading and combining batches from disk...")
        result_dfs = []

        read_batch_size = max(3, len(batch_files) // 10)

        for i in range(0, len(batch_files), read_batch_size):
            read_batch = batch_files[i:i + read_batch_size]
            print(f"  Reading group {i//read_batch_size + 1}/{(len(batch_files) + read_batch_size - 1)//read_batch_size}...")

            dfs_to_concat = [pd.read_parquet(f) for f in read_batch]
            combined = pd.concat(dfs_to_concat, ignore_index=True, copy=False)
            result_dfs.append(combined)

            del dfs_to_concat
            gc.collect()

        print("Final merge...")
        result_df = pd.concat(result_dfs, ignore_index=True, copy=False)

        del result_dfs
        gc.collect()

        return result_df

    finally:
        try:
            shutil.rmtree(temp_concat_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_concat_dir}: {e}")


if __name__ == '__main__':
    mp.set_start_method("spawn")
