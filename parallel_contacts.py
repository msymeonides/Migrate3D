import pandas as pd
import numpy as np
from contacts import contacts, contacts_moving, no_daughter_contacts
import multiprocessing as mp


def worker(task):
    timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval, worker_id = task
    result = process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval)

    return result


def process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval):

    is_in_chunk = np.isin(arr_segments[:, 1], timepoint_chunk)
    indices = np.where(is_in_chunk)[0]
    arr_segments_chunk = arr_segments[indices]
    if arr_segments_chunk.size == 0:
        return None, None, None

    unique_objects_chunk = np.unique(arr_segments_chunk[:, 0])
    df_cont = contacts(unique_objects_chunk, arr_segments_chunk, contact_length)
    if not df_cont or all(len(df) == 0 for df in df_cont):
        return None, None, None

    df_contacts = pd.concat(df_cont, ignore_index=True)
    df_contacts.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)
    df_no_daughter_func = no_daughter_contacts(unique_objects_chunk, df_contacts)
    df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True) if df_no_daughter_func else pd.DataFrame()
    df_no_daughter.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)
    df_alive = contacts_moving(df_sum, df_no_daughter, arrested)
    df_no_dead_ = pd.concat(df_alive, ignore_index=True) if df_alive else pd.DataFrame()
    df_no_dead_.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)

    return df_contacts, df_no_daughter, df_no_dead_


# Custom split with overlap
def split_with_overlap(timepoints, num_chunks, overlap):
    chunks = np.array_split(timepoints, num_chunks)
    for i in range(1, len(chunks)):
        overlap_elements = chunks[i-1][-overlap:]
        chunks[i] = np.concatenate((overlap_elements, chunks[i]))
    return chunks


def main(timepoints, arr_segments, contact_length, df_sum, arrested, time_interval):
    # Determine the number of CPU cores available for parallel processing
    max_processes = max(1, min(61, mp.cpu_count() - 1))
    num_workers = max_processes
    # num_workers = mp.cpu_count()

    unique_timepoints = np.unique(timepoints)
    timepoint_chunks = split_with_overlap(unique_timepoints, num_workers, overlap = 3)
    with mp.Pool(processes=num_workers) as pool:
        worker_tasks = [(chunk, arr_segments, contact_length, df_sum, arrested, time_interval, i)
                        for i, chunk in enumerate(timepoint_chunks)]
        results = pool.map(worker, worker_tasks)
    df_contacts = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True)
    df_no_daughter = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
    df_no_dead_ = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True)
    df_contacts.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)

    return df_contacts, df_no_daughter, df_no_dead_


if __name__ == '__main__':
    mp.set_start_method("spawn")
