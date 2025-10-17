import pandas as pd
import numpy as np
from contacts import contacts, contacts_notdead, contacts_notdividing
import multiprocessing as mp


def worker(task):
    timepoint_chunk, arr_segments, contact_length, df_sum, arrested, divfilter, worker_id = task
    result = process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, divfilter)
    return result


def process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, divfilter):
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
    df_contacts = drop_symmetric_duplicates(df_contacts)

    if divfilter:
        df_no_div_func = contacts_notdividing(unique_objects_chunk, df_contacts)
        valid_frames = [
            df.dropna(axis=1, how='all')
            for df in df_no_div_func
            if not df.dropna(axis=1, how='all').empty
        ]
        df_no_div = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame(
            columns=["Object ID", "Object Compare", "Time of Contact"])
        df_no_div.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)
    else:
        df_no_div = df_contacts.copy()

    df_alive = contacts_notdead(df_sum, df_no_div, arrested)
    df_no_dead = pd.concat(df_alive, ignore_index=True) if df_alive else pd.DataFrame(columns=["Object ID", "Object Compare", "Time of Contact"])
    df_no_dead.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"], inplace=True)

    return df_contacts, df_no_div, df_no_dead


def split_with_overlap(timepoints, num_chunks, overlap):
    chunks = np.array_split(timepoints, num_chunks)
    for i in range(1, len(chunks)):
        overlap_elements = chunks[i-1][-overlap:]
        chunks[i] = np.concatenate((overlap_elements, chunks[i]))
    return chunks


def drop_symmetric_duplicates(df):
    df_copy = df.copy()
    mask = df_copy["Object ID"] > df_copy["Object Compare"]
    df_copy.loc[mask, ["Object ID", "Object Compare"]] = df_copy.loc[mask, ["Object Compare", "Object ID"]].values
    return df_copy.drop_duplicates(subset=["Object ID", "Object Compare", "Time of Contact"])


def main(timepoints, arr_segments, contact_length, df_sum, arrested, divfilter):
    max_processes = max(1, min(61, mp.cpu_count() - 2))
    num_workers = max_processes

    unique_timepoints = np.unique(timepoints)
    timepoint_chunks = split_with_overlap(unique_timepoints, num_workers, overlap = 3)
    with mp.Pool(processes=num_workers) as pool:
        worker_tasks = [(chunk, arr_segments, contact_length, df_sum, arrested, divfilter, i)
                        for i, chunk in enumerate(timepoint_chunks)]
        results = pool.map(worker, worker_tasks)
    contacts_list = [r[0] for r in results if r[0] is not None]
    no_div_list = [r[1] for r in results if r[1] is not None]
    no_dead_list = [r[2] for r in results if r[2] is not None]

    if contacts_list:
        df_contacts = pd.concat(contacts_list, ignore_index=True)
        df_contacts = drop_symmetric_duplicates(df_contacts)
    else:
        df_contacts = pd.DataFrame(columns=["Object ID", "Object Compare", "Time of Contact"])

    if no_div_list:
        df_no_div = pd.concat(no_div_list, ignore_index=True)
        df_no_div = drop_symmetric_duplicates(df_no_div)
    else:
        df_no_div = pd.DataFrame(columns=["Object ID", "Object Compare", "Time of Contact"])

    if no_dead_list:
        df_no_dead = pd.concat(no_dead_list, ignore_index=True)
        df_no_dead = drop_symmetric_duplicates(df_no_dead)
    else:
        df_no_dead = pd.DataFrame(columns=["Object ID", "Object Compare", "Time of Contact"])

    return df_contacts, df_no_div, df_no_dead


if __name__ == '__main__':
    mp.set_start_method("spawn")
