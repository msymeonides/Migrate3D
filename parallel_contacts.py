import pandas as pd
import numpy as np
from contacts import contacts, contacts_moving, no_daughter_contacts
import multiprocessing as mp


def worker(task):
    # Each worker processes a chunk of timepoints and returns the results.
    timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval, worker_id = task
    print(f"Worker {worker_id} processing chunk with {len(timepoint_chunk)} timepoints")

    # Process the chunk and get the results
    result = process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval)
    print(f"Worker {worker_id} finished processing chunk.")

    # Return the processed results (dataframes for contacts, no daughter contacts, and no dead cells)
    return result


def process_chunk(timepoint_chunk, arr_segments, contact_length, df_sum, arrested, time_interval):
    # Filter arr_segments where the timepoints in the second column are in the current chunk.
    is_in_chunk = np.isin(arr_segments[:, 1], timepoint_chunk)

    # Find the indices of the segments that are in the current chunk
    indices = np.where(is_in_chunk)[0]

    # Get the subset of arr_segments that correspond to the timepoint chunk
    arr_segments_chunk = arr_segments[indices]

    # If no segments are in the chunk, return None for all outputs
    if arr_segments_chunk.size == 0:
        return None, None, None

    # Get unique objects (first column of arr_segments) in the current chunk
    unique_objects_chunk = np.unique(arr_segments_chunk[:, 0])

    # Calculate contacts between objects in the current chunk using the contacts function
    df_cont = contacts(unique_objects_chunk, arr_segments_chunk, contact_length)

    # If no contacts are found or all contacts are empty, return None
    if not df_cont or all(len(df) == 0 for df in df_cont):
        return None, None, None

    # Concatenate the list of contact dataframes into a single dataframe
    df_contacts = pd.concat(df_cont, ignore_index=True)

    # Remove daughter contacts from the contact dataframe using the no_daughter_contacts function
    df_no_daughter_func = no_daughter_contacts(unique_objects_chunk, df_contacts)

    # If any daughter contacts are found, concatenate them into a single dataframe
    df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True) if df_no_daughter_func else pd.DataFrame()

    # Get the contacts for the moving cells using contacts_moving function
    df_alive = contacts_moving(df_sum, df_no_daughter, arrested)

    # Concatenate the list of results from contacts_moving into a single dataframe
    df_no_dead_ = pd.concat(df_alive, ignore_index=True) if df_alive else pd.DataFrame()

    # Return the final dataframes for contacts, no daughter contacts, and no dead cells
    return df_contacts, df_no_daughter, df_no_dead_


def main(timepoints, arr_segments, contact_length, df_sum, arrested, time_interval):
    # Determine the number of CPU cores available for parallel processing
    num_workers = mp.cpu_count()

    # Get the unique timepoints from the provided timepoints
    unique_timepoints = np.unique(timepoints)

    # Split the unique timepoints into chunks, one for each worker
    timepoint_chunks = np.array_split(unique_timepoints, num_workers)
    print(f"Split {len(unique_timepoints)} timepoints into {len(timepoint_chunks)} chunks.")

    # Create a pool of workers, and distribute the work of processing each timepoint chunk
    with mp.Pool(processes=num_workers) as pool:
        # Create a list of tasks to be sent to each worker
        worker_tasks = [(chunk, arr_segments, contact_length, df_sum, arrested, time_interval, i)
                        for i, chunk in enumerate(timepoint_chunks)]

        # Map the tasks to the workers and get the results
        results = pool.map(worker, worker_tasks)

    # Concatenate the results from all workers into final dataframes
    df_contacts = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True)
    df_no_daughter = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
    df_no_dead_ = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True)

    # Return the final dataframes with contacts, no daughter contacts, and no dead cells
    return df_contacts, df_no_daughter, df_no_dead_


if __name__ == '__main__':
    # Primarily used to stop the issues the entire program restarting in each worker
    # Also ensures cross-platform compatibility maybe???????
    mp.set_start_method("spawn")
