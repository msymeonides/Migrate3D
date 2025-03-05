import pandas as pd
import numpy as np
import asyncio
import queue
import threading
from contacts import contacts, contacts_moving, no_daughter_contacts


def worker(input_queue: queue.Queue, output_queue: queue.Queue, worker_id: int):
    """
    Worker function that continuously processes tasks from the input queue.
    When a task is received, it processes a chunk of timepoints and puts the results in the output queue.
    """
    while True:
        task = input_queue.get()  # Get a task from the queue
        if task is None:  # If None is received, terminate the worker
            print(f"Worker {worker_id} shutting down.")
            input_queue.task_done()
            break
        timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval = task
        result = process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval)
        output_queue.put(result)  # Place the result into the output queue
        input_queue.task_done()


def process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval):
    """
    Processes a chunk of timepoints by calling the required functions for contact analysis.
    Returns processed DataFrames with contact details.
    """
    df_cont = contacts(timepoint_chunk, arr_segments, contact_length)  # Compute contacts for the chunk
    if len(df_cont) == 0:
        return None, None, None, None  # Return None if no data is found

    df_contacts = pd.concat(df_cont, ignore_index=True)  # Merge results into a single DataFrame
    df_no_daughter_func = no_daughter_contacts(timepoint_chunk, df_contacts)  # Process no-daughter contacts
    df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True)

    # Analyze contacts for movement and filter arrested cells
    df_alive, df_contact_sum = contacts_moving(df_sum, df_no_daughter, arrested, time_interval)
    df_no_dead_ = pd.concat(df_alive, ignore_index=True)

    # Filter out entries with meaningful contact durations
    with_contacts = [df for df in df_contact_sum if df['Median Contact Duration'].notna().any()]
    df_contact_summary = pd.concat(with_contacts, ignore_index=True).replace({0: None}).dropna()

    return df_contacts, df_no_daughter, df_no_dead_, df_contact_summary


async def monitor_output(output_queue: queue.Queue):
    """
    Asynchronous function to monitor the output queue and print results.
    """
    while True:
        item = await asyncio.to_thread(get_from_queue, output_queue, 0.1)
        if item is not None:
            print(f"Async monitor got result from worker {item[0]}")
            output_queue.task_done()
        await asyncio.sleep(0.1)  # Small sleep to avoid excessive CPU usage


def get_from_queue(q: queue.Queue, timeout: float = 0.1):
    """
    Helper function to safely retrieve an item from a queue with a timeout.
    Returns None if the queue is empty.
    """
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


async def main(timepoints, arr_segments, contact_length, df_sum, arrested, time_interval):
    """
    Main function that distributes work among multiple worker threads,
    collects results asynchronously, and processes them into final DataFrames.
    """
    input_queue = queue.Queue()  # Queue to send tasks to workers
    output_queue = queue.Queue()  # Queue to collect results from workers
    num_workers = 4  # Number of worker threads
    threads = []

    # Start worker threads
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(input_queue, output_queue, i), daemon=True)
        t.start()
        threads.append(t)

    # Split timepoints into roughly equal chunks for parallel processing
    unique_timepoints = np.unique(timepoints)
    chunk_size = len(unique_timepoints) // num_workers if len(unique_timepoints) > 0 else 1
    timepoint_chunks = [unique_timepoints[i:i + chunk_size] for i in range(0, len(unique_timepoints), chunk_size)]

    # Add timepoint chunks to the input queue for processing
    for chunk in timepoint_chunks:
        input_queue.put((chunk, arr_segments, contact_length, df_sum, arrested, time_interval))

    results = []

    # Collect results from workers asynchronously
    while len(results) < len(timepoint_chunks):
        item = await asyncio.to_thread(get_from_queue, output_queue, 0.1)
        if item is not None:
            results.append(item)
            output_queue.task_done()
        await asyncio.sleep(0.1)

    # Combine results into final DataFrames
    df_contacts = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True)
    df_no_daughter = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
    df_no_dead_ = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True)
    df_contact_summary = pd.concat([r[3] for r in results if r[3] is not None], ignore_index=True)

    # Signal workers to shut down by sending None to the input queue
    for _ in range(num_workers):
        input_queue.put(None)
    for t in threads:
        t.join()  # Ensure all threads have finished execution

    print("All tasks completed and workers shut down.")
    return df_contacts, df_no_daughter, df_no_dead_, df_contact_summary


if __name__ == '__main__':
    asyncio.run(main())  # Run the main function asynchronously
