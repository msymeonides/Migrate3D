import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


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
    """Prepare object data in chunks to reduce memory overhead"""
    chunks = []
    current_chunk = []

    for i, obj in enumerate(unique_objects):
        # Extract coordinates for this object
        object_mask = arr_segments[:, 0] == obj
        object_coords = arr_segments[object_mask, 2:5]  # x, y, z columns
        current_chunk.append((obj, object_coords))

        # Create chunk when we reach chunk_size or at the end
        if len(current_chunk) >= chunk_size or i == len(unique_objects) - 1:
            chunks.append(current_chunk)
            current_chunk = []

    return chunks


def main(arr_segments, unique_objects, tau, n_workers=None, chunk_size=None):
    """
    Optimized MSD calculation with reduced memory usage

    Parameters:
    - chunk_size: Number of objects to process per chunk (default: auto-calculate)
    """
    # Determine optimal parameters
    max_processes = max(1, min(8, mp.cpu_count() - 1))  # Reduced max processes
    num_workers = n_workers if n_workers is not None else max_processes

    # Auto-calculate chunk_size based on data size and available workers
    if chunk_size is None:
        total_objects = len(unique_objects)
        # Aim for 2-4 chunks per worker, with minimum chunk size of 10
        chunk_size = max(10, total_objects // (num_workers * 3))

    # Prepare data in chunks
    object_chunks = prepare_object_data(arr_segments, unique_objects, chunk_size)

    # Create partial function with tau_range
    tau_range = np.arange(1, tau + 1)
    worker_func = partial(worker_chunked, tau_range=tau_range)

    # Process chunks with multiprocessing
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_func, object_chunks)
    except Exception as e:
        # Fallback to single-threaded processing if multiprocessing fails
        print(f"Multiprocessing failed ({e}), falling back to single-threaded processing...")
        results = [worker_func(chunk) for chunk in object_chunks]

    # Combine results
    msd_dict = {}
    for chunk_result in results:
        for row in chunk_result:
            obj = row[0]
            msd_vals = row[1:]
            msd_dict[obj] = msd_vals

    # Create DataFrame
    columns = ['Object ID'] + list(range(1, tau + 1))
    msd_data = [[obj] + msd_dict[obj] for obj in unique_objects]
    df_msd = pd.DataFrame(msd_data, columns=columns)

    # Ensure numeric columns
    for col in columns[1:]:
        df_msd[col] = pd.to_numeric(df_msd[col], errors='coerce')

    return df_msd


if __name__ == '__main__':
    mp.set_start_method("spawn")
