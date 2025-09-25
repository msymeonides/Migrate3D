from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd

def worker_chunked(chunk_data, tau_range):
    """Process a chunk of objects for MSD calculation"""
    msd_list = []

    for obj_data in chunk_data:
        obj_id, coordinates = obj_data
        n = coordinates.shape[0]
        x_val, y_val, z_val = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        msd = []
        for lag in tau_range:
            if n <= lag:
                msd.append(np.nan)
                continue

            sq_diff = (np.square(x_val[lag:] - x_val[:-lag]) +
                       np.square(y_val[lag:] - y_val[:-lag]) +
                       np.square(z_val[lag:] - z_val[:-lag]))

            valid = sq_diff > 0
            msd_val = np.mean(sq_diff[valid]) if np.any(valid) else np.nan
            msd.append(msd_val)

        msd_list.append([obj_id] + msd)

    return msd_list


def prepare_object_data(arr_segments, unique_objects, chunk_size):
    chunks = []
    current_chunk = []

    sort_indices = np.argsort(arr_segments[:, 0])
    sorted_segments = arr_segments[sort_indices]

    object_boundaries = np.where(np.diff(sorted_segments[:, 0]))[0] + 1
    object_boundaries = np.concatenate(([0], object_boundaries, [len(sorted_segments)]))

    for i, obj in enumerate(unique_objects):
        start_idx = object_boundaries[i]
        end_idx = object_boundaries[i + 1]
        object_coords = sorted_segments[start_idx:end_idx, 2:5]

        current_chunk.append((obj, object_coords))

        if len(current_chunk) >= chunk_size or i == len(unique_objects) - 1:
            chunks.append(current_chunk)
            current_chunk = []

    return chunks


def main(arr_segments, unique_objects, tau, n_workers=None, chunk_size=None):
    max_processes = max(1, min(61, mp.cpu_count() - 1))
    num_workers = n_workers if n_workers is not None else max_processes

    if chunk_size is None:
        total_objects = len(unique_objects)
        chunk_size = max(10, total_objects // (num_workers * 3))

    object_chunks = prepare_object_data(arr_segments, unique_objects, chunk_size)

    tau_range = np.arange(1, tau + 1)
    worker_func = partial(worker_chunked, tau_range=tau_range)

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_func, object_chunks)
    except Exception as e:
        print(f"Multiprocessing failed ({e}), falling back to single-threaded processing...")
        results = [worker_func(chunk) for chunk in object_chunks]

    msd_dict = {}
    for chunk_result in results:
        for row in chunk_result:
            obj = row[0]
            msd_vals = row[1:]
            msd_dict[obj] = msd_vals

    columns = ['Object ID'] + list(range(1, tau + 1))
    msd_data = []
    for obj in unique_objects:
        if obj in msd_dict:
            msd_data.append([obj] + msd_dict[obj])
        else:
            msd_data.append([obj] + [np.nan] * tau)

    df_msd = pd.DataFrame(msd_data, columns=columns)

    for col in columns[1:]:
        df_msd[col] = pd.to_numeric(df_msd[col], errors='coerce')

    return df_msd

if __name__ == '__main__':
    mp.set_start_method("spawn")
