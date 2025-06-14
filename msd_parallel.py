import numpy as np
import pandas as pd
import multiprocessing as mp

def worker(task):
    arr_segments, unique_objects, tau_range, max_len, worker_id = task
    msd_list = []
    for obj in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == obj]
        n = object_data.shape[0]
        x_val = object_data[:, 2]
        y_val = object_data[:, 3]
        z_val = object_data[:, 4]
        msd = []
        for lag in tau_range:
            if n <= lag:
                msd.append(np.nan)
                continue
            sq_diff = np.square(x_val[lag:] - x_val[:-lag]) + \
                      np.square(y_val[lag:] - y_val[:-lag]) + \
                      np.square(z_val[lag:] - z_val[:-lag])
            valid = sq_diff > 0
            msd_val = np.mean(sq_diff[valid]) if np.any(valid) else np.nan
            msd.append(msd_val)
        msd_list.append([obj] + msd)
    return msd_list

def main(arr_segments, unique_objects, tau, n_workers=None):
    max_processes = max(1, min(61, mp.cpu_count() - 1))
    num_workers = n_workers if n_workers is not None else max_processes
    max_len = max([(arr_segments[:, 0] == obj).sum() for obj in unique_objects])
    tau_ranges = np.array_split(np.arange(1, tau + 1), num_workers)
    tasks = [
        (arr_segments, unique_objects, tau_range, max_len, i)
        for i, tau_range in enumerate(tau_ranges) if len(tau_range) > 0
    ]
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker, tasks)

    msd_dict = {}
    for msd_chunk in results:
        for row in msd_chunk:
            obj = row[0]
            msd_vals = row[1:]
            if obj not in msd_dict:
                msd_dict[obj] = []
            msd_dict[obj].extend(msd_vals)
    columns = ['Object ID'] + [i + 1 for i in range(tau)]
    msd_data = [[obj] + msd_dict[obj] for obj in unique_objects]
    df_msd = pd.DataFrame(msd_data, columns=columns)
    for col in columns[1:]:
        df_msd[col] = pd.to_numeric(df_msd[col], errors='coerce')
    return df_msd

if __name__ == '__main__':
    mp.set_start_method("spawn")
