import concurrent.futures
import numpy as np
import pandas as pd
import re
import time as tempo
import warnings
from pandas.errors import PerformanceWarning
from scipy.spatial import ConvexHull

from msd_parallel import main as msd_parallel_main
from msd_loglogfits import main as msd_loglogfits
from PCA import pca
from xgb import xgboost, XGBAbortException
from shared_state import messages, thread_lock, complete_progress_step
from overall_medians import overall_medians

def compute_object_summary(obj, arr_segments, df_obj_calcs, arr_tracks, parameters):
    object_data = arr_segments[arr_segments[:, 0] == obj, :]
    x_val, y_val, z_val = object_data[:, 2], object_data[:, 3], object_data[:, 4]

    category = ''
    if arr_tracks.shape[0] > 0:
        obj_id_str = str(int(float(object_data[0, 0])))
        matching_index = np.where(arr_tracks[:, 0].astype(str) == obj_id_str)[0]
        if matching_index.size > 0:
            category = arr_tracks[matching_index[0], 1]

    convex_coords = np.column_stack((x_val, y_val, z_val))
    convex_hull_volume = 0
    if convex_coords.shape[0] >= 4:
        try:
            convex_hull_volume = ConvexHull(convex_coords).volume
        except Exception:
            pass

    path_lengths = df_obj_calcs['Path Length'].values
    max_path = path_lengths.max() if path_lengths.size > 0 else 0
    times = df_obj_calcs['Time'].values
    time_interval = abs(times[1] - times[0]) if times.size > 1 else 0
    duration_val = times.size * time_interval

    total_disp = df_obj_calcs['Total Displacement'].values
    final_euclid = total_disp[-1] if total_disp.size > 0 else 0
    max_euclid = total_disp.max() if total_disp.size > 0 else 0
    straightness = (final_euclid / max_path if max_path != 0 else 0) * np.sqrt(duration_val)
    displacement_ratio = final_euclid / max_euclid if max_euclid != 0 else 0
    convex = convex_hull_volume / np.sqrt(duration_val) if duration_val != 0 else 0
    outreach_ratio = max_euclid / max_path if max_path != 0 else 0

    velocity = df_obj_calcs['Instantaneous Velocity'].values[1:] if 'Instantaneous Velocity' in df_obj_calcs else np.array([])
    valid_velocity = velocity[(~np.isnan(velocity)) & (velocity != 0)] if velocity.size > 0 else np.array([])
    velocity_mean = valid_velocity.mean() if valid_velocity.size > 0 else 0
    velocity_median = np.median(valid_velocity) if valid_velocity.size > 0 else 0

    acceleration = df_obj_calcs['Instantaneous Acceleration'].values if 'Instantaneous Acceleration' in df_obj_calcs else np.array([])
    if acceleration.size >= parameters['moving']:
        valid_acc = acceleration[(~np.isnan(acceleration)) & (acceleration != 0)]
        acceleration_mean = valid_acc.mean() if valid_acc.size > 0 else 0
        acceleration_median = np.median(valid_acc) if valid_acc.size > 0 else 0
        accel_abs = np.abs(valid_acc)
        accel_abs_mean = accel_abs.mean() if accel_abs.size > 0 else 0
        accel_abs_median = np.median(accel_abs) if accel_abs.size > 0 else 0
    else:
        acceleration_mean = acceleration_median = accel_abs_mean = accel_abs_median = 0

    acceleration_filtered = df_obj_calcs['Instantaneous Acceleration Filtered'].values if 'Instantaneous Acceleration Filtered' in df_obj_calcs else np.array([])
    valid_acc_f = acceleration_filtered[(~np.isnan(acceleration_filtered)) & (acceleration_filtered != 0)] if acceleration_filtered.size > 0 else np.array([])
    if valid_acc_f.size >= parameters['moving']:
        velocity_filtered = df_obj_calcs['Instantaneous Velocity Filtered'].values if 'Instantaneous Velocity Filtered' in df_obj_calcs else np.array([])
        valid_velocity_f = velocity_filtered[(~np.isnan(velocity_filtered)) & (velocity_filtered != 0)] if velocity_filtered.size > 0 else np.array([])
        velocity_filtered_mean = valid_velocity_f.mean() if valid_velocity_f.size > 0 else 0
        velocity_filtered_median = np.median(valid_velocity_f) if valid_velocity_f.size > 0 else 0
        acceleration_filtered_mean = valid_acc_f.mean()
        acceleration_filtered_median = np.median(valid_acc_f)
        acceleration_filtered_stdev = valid_acc_f.std(ddof=1) if valid_acc_f.size > 1 else 0
        accel_filtered_abs = np.abs(valid_acc_f)
        accel_filtered_abs_mean = accel_filtered_abs.mean() if accel_filtered_abs.size > 0 else 0
        accel_filtered_abs_median = np.median(accel_filtered_abs) if accel_filtered_abs.size > 0 else 0
        accel_filtered_abs_stdev = accel_filtered_abs.std(ddof=1) if accel_filtered_abs.size > 1 else 0
        velocity_filtered_stdev = valid_velocity_f.std(ddof=1) if valid_velocity_f.size > 1 else 0
    else:
        velocity_filtered_mean = velocity_filtered_median = 0
        acceleration_filtered_mean = acceleration_filtered_median = acceleration_filtered_stdev = 0
        accel_filtered_abs_mean = accel_filtered_abs_median = accel_filtered_abs_stdev = 0
        velocity_filtered_stdev = 0

    cols_angles = [col for col in df_obj_calcs.columns if 'Filtered Angle' in col]
    cols_euclidean = [col for col in df_obj_calcs.columns if 'Euclid' in col]
    overall_euclidean_median, overall_angle_median, single_euclid, single_angle = overall_medians(
        obj, df_obj_calcs, cols_angles, cols_euclidean)

    inst_disp = df_obj_calcs['Instantaneous Displacement'].values if 'Instantaneous Displacement' in df_obj_calcs else np.array([])
    valid_disp = inst_disp[(~np.isnan(inst_disp)) & (inst_disp != 0)] if inst_disp.size > 0 else np.array([])
    time_under = valid_disp[valid_disp < parameters['arrest_limit']] if valid_disp.size > 0 else np.array([])
    arrest_coefficient = (time_under.size * time_interval) / duration_val if duration_val != 0 else 0

    summary_tuple = (
        obj, duration_val, final_euclid, max_euclid, max_path, straightness,
        displacement_ratio, outreach_ratio, velocity_mean,
        velocity_median, velocity_filtered_mean, velocity_filtered_median,
        velocity_filtered_stdev, acceleration_mean, acceleration_median,
        accel_abs_mean, accel_abs_median, acceleration_filtered_mean, acceleration_filtered_median,
        acceleration_filtered_stdev, accel_filtered_abs_mean, accel_filtered_abs_median,
        accel_filtered_abs_stdev, arrest_coefficient, overall_angle_median,
        overall_euclidean_median, convex, category
    )
    return obj, summary_tuple, single_euclid, single_angle

def summary_sheet(arr_segments, df_all_calcs, unique_objects, tau, parameters, arr_tracks, savefile):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented")

    summary_columns = [
        'Object ID', 'Duration', 'Final Euclidean', 'Max Euclidean', 'Path Length',
        'Straightness', 'Displacement Ratio', 'Outreach Ratio',
        'Velocity Mean', 'Velocity Median', 'Velocity filtered Mean', 'Velocity Filtered Median',
        'Velocity Filtered Standard Deviation', 'Acceleration Mean', 'Acceleration Median',
        'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
        'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
        'Absolute Acceleration Filtered Mean', 'Absolute Acceleration Filtered Median',
        'Absolute Acceleration Filtered Standard Deviation', 'Arrest Coefficient',
        'Overall Angle Median', 'Overall Euclidean Median', 'Convex Hull Volume', 'Category'
    ]
    n_summary_cols = len(summary_columns)

    with thread_lock:
        messages.append("Calculating mean square displacements...")
    tic = tempo.time()
    df_msd = msd_parallel_main(arr_segments, unique_objects, tau)

    toc = tempo.time()
    with thread_lock:
        msg = " MSD calculations done in {:.0f} seconds.".format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append("")
    complete_progress_step("MSD")

    df_all_calcs_by_obj = dict(tuple(df_all_calcs.groupby("Object ID")))

    with thread_lock:
        messages.append("Calculating summary statistics...")
    tic = tempo.time()

    sum_ = {}
    single_euclid_dict = {}
    single_angle_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                compute_object_summary, obj, arr_segments,
                df_all_calcs_by_obj.get(obj, pd.DataFrame()), arr_tracks, parameters): obj
            for obj in unique_objects}
        for future in concurrent.futures.as_completed(futures):
            obj = futures[future]
            try:
                obj, summary_tuple, single_euclid, single_angle = future.result()
                sum_[obj] = summary_tuple
                single_euclid_dict[obj] = single_euclid
                single_angle_dict[obj] = single_angle
            except Exception:
                sum_[obj] = (obj,) + (np.nan,) * (n_summary_cols - 1)
                single_euclid_dict[obj] = {}
                single_angle_dict[obj] = {}

    for obj in unique_objects:
        if obj not in sum_:
            sum_[obj] = (obj,) + (np.nan,) * (n_summary_cols - 1)
            single_euclid_dict[obj] = {}
            single_angle_dict[obj] = {}

    summary_rows = [sum_[obj] for obj in unique_objects]
    df_sum = pd.DataFrame(summary_rows, columns=summary_columns)
    df_sum["Object ID"] = df_sum["Object ID"].astype(float)

    df_single_euclids_df = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_euclids_df.reset_index(inplace=True)
    cols_euclidean_numbers = [
        int(re.search(r"\d+", str(col)).group()) if re.search(r"\d+", str(col)) else col
        for col in df_single_euclids_df.columns[1:]]
    df_single_euclids_df.columns = ["Object ID"] + cols_euclidean_numbers
    df_single_euclids_df = df_single_euclids_df.sort_values(by="Object ID").reset_index(drop=True)

    df_single_angles_df = pd.DataFrame.from_dict(single_angle_dict, orient='index')
    df_single_angles_df.reset_index(inplace=True)
    n_angle_cols = df_single_angles_df.shape[1] - 1
    angle_intervals = [3 + 2 * i for i in range(n_angle_cols)]
    df_single_angles_df.columns = ["Object ID"] + angle_intervals
    df_single_angles_df = df_single_angles_df.sort_values(by="Object ID").reset_index(drop=True)

    existing_cols = [col for col in range(1, tau + 1) if col in df_msd.columns]
    df_msd = df_msd[["Object ID"] + existing_cols]
    df_msd["Object ID"] = df_msd["Object ID"].astype(float)

    msd_vals = df_msd.set_index("Object ID")[existing_cols]
    if parameters.get("infile_tracks", False):
        category_tracks = arr_tracks[:, 1]
        if len(category_tracks) == len(msd_vals):
            msd_vals["Category"] = category_tracks
            grouped = msd_vals.groupby("Category")
            df_msd_avg_per_cat = grouped.mean().T
            df_msd_std_per_cat = grouped.std().T
            df_msd_avg_per_cat.index.name = "MSD"
        else:
            df_msd_avg_per_cat = pd.DataFrame()
            df_msd_std_per_cat = pd.DataFrame()
    else:
        df_msd_avg_per_cat = pd.DataFrame()
        df_msd_std_per_cat = pd.DataFrame()

    msd_vals_summary = msd_vals.drop(columns="Category", errors="ignore")
    df_msd_sum_all = pd.DataFrame({
        "Avg": msd_vals_summary.mean(),
        "StDev": msd_vals_summary.std()})
    df_msd_sum_all.index.name = "MSD"

    category_df = pd.DataFrame(arr_tracks, columns=["Object ID", "Category"])
    if not category_df.empty:
        category_df["Object ID"] = category_df["Object ID"].astype(float)
        def insert_category(df):
            merged = pd.merge(df, category_df, on="Object ID", how="left")
            cols = merged.columns.tolist()
            cols.insert(1, cols.pop(cols.index("Category")))
            return merged[cols]
        df_single_euclids_df = insert_category(df_single_euclids_df)
        df_single_angles_df = insert_category(df_single_angles_df)
        df_msd = insert_category(df_msd)

    max_msd_dict = {}
    msd_cols = [col for col in df_msd.columns if isinstance(col, int)]
    for obj in unique_objects:
        obj_val = float(obj)
        msd_row = df_msd[df_msd["Object ID"] == obj_val]
        if not msd_row.empty and msd_cols:
            max_msd = msd_row[msd_cols].max(axis=1, skipna=True).values[0]
        else:
            max_msd = np.nan
        max_msd_dict[obj_val] = max_msd

    df_sum["Maximum MSD"] = df_sum["Object ID"].map(max_msd_dict)

    cols = list(df_sum.columns)
    final_msd_idx = cols.index("Maximum MSD")
    cat_idx = cols.index("Category")
    cols.insert(cat_idx, cols.pop(final_msd_idx))
    df_sum = df_sum[cols]
    df_msdloglogfits = msd_loglogfits(df_msd)

    toc = tempo.time()
    with thread_lock:
        msg = " Summary statistics done in {:.0f} seconds.".format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append("")
    complete_progress_step("Summary")

    if parameters.get("infile_tracks", False):
        df_pca = pca(df_sum.copy(), parameters, savefile)
        try:
            xgboost(df_sum.copy(), parameters, savefile)
        except XGBAbortException:
            pass
    else:
        df_pca = None

    return (df_sum, df_single_euclids_df, df_single_angles_df, df_msd, df_msd_sum_all,
            df_msd_avg_per_cat, df_msd_std_per_cat, df_msdloglogfits, df_pca)
