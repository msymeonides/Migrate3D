import pandas as pd
import numpy as np
import queue
import threading
from contacts import contacts, contacts_moving, no_daughter_contacts
import multiprocessing        # needed for max_processes calculation but right now we're not using it
import time

def worker(input_queue: queue.Queue, output_queue: queue.Queue, worker_id: int):
    """
    Worker function that continuously processes tasks from the input queue.
    """
    while True:
        task = input_queue.get()
        if task is None:
            input_queue.task_done()
            break
        timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval = task
        print(f"Worker {worker_id} processing chunk with {len(timepoint_chunk)} timepoints")
        result = process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval)
        output_queue.put(result)
        print(f"Worker {worker_id} finished processing chunk.")
        input_queue.task_done()
    print(f"Worker {worker_id} shutting down.")


def process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval):
    is_in_chunk = np.isin(arr_segments[:, 1], timepoint_chunk)
    arr_segments_chunk = arr_segments[is_in_chunk]

    if arr_segments_chunk.size == 0:
        return None, None, None

    unique_objects_chunk = np.unique(arr_segments_chunk[:, 0])

    df_cont = contacts(unique_objects_chunk, arr_segments_chunk, contact_length)
    if not df_cont or all(len(df) == 0 for df in df_cont):
        return None, None, None

    df_contacts = pd.concat(df_cont, ignore_index=True)
    print(f"[DEBUG] df_contacts rows: {len(df_contacts)}")

    df_no_daughter_func = no_daughter_contacts(unique_objects_chunk, df_contacts)
    if not df_no_daughter_func or all(len(df) == 0 for df in df_no_daughter_func):
        df_no_daughter = pd.DataFrame()
    else:
        df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True)

    # Update: Remove unpacking as contacts_moving now returns a single value.
    df_alive = contacts_moving(df_sum, df_no_daughter, arrested)
    if not df_alive or all(len(df) == 0 for df in df_alive):
        df_no_dead_ = pd.DataFrame()
    else:
        df_no_dead_ = pd.concat(df_alive, ignore_index=True)

    return df_contacts, df_no_daughter, df_no_dead_


def main(timepoints, arr_segments, contact_length, df_sum, arrested, time_interval):
    """
    Synchronous main function that distributes work among multiple worker threads.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    max_processes = max(1, min(61, multiprocessing.cpu_count() - 1))      # Leaving this here in case things improve later and we can use more threads
    num_workers = 3    # 3 threads is currently the sweet spot for performance improvement
    threads = []

    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(input_queue, output_queue, i), daemon=True)
        t.start()
        threads.append(t)

    unique_timepoints = np.unique(timepoints)
    timepoint_chunks = np.array_split(unique_timepoints, num_workers)
    print(f"Split {len(unique_timepoints)} timepoints into {len(timepoint_chunks)} chunks.")

    for chunk in timepoint_chunks:
        input_queue.put((chunk, arr_segments, contact_length, df_sum, arrested, time_interval))

    input_queue.join()
    print("All tasks have been processed.")

    results = []
    tasks_count = len(timepoint_chunks)
    while len(results) < tasks_count:
        try:
            result = output_queue.get(timeout=0.1)
            results.append(result)
        except queue.Empty:
            time.sleep(0.1)

    for _ in range(num_workers):
        input_queue.put(None)
    input_queue.join()
    for t in threads:
        t.join()
    print("All workers shut down.")

    df_contacts = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True)
    df_no_daughter = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
    df_no_dead_ = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True)

    return df_contacts, df_no_daughter, df_no_dead_

'''
if __name__ == '__main__':
    main(timepoints=[], arr_segments=np.array([]), contact_length=0, df_sum=pd.DataFrame(), arrested=0, time_interval=0)'''