import concurrent.futures
import multiprocessing as mp
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import re
from scipy.spatial import ConvexHull
import time as tempo
import warnings

from msd_parallel import main as msd_parallel_main
from msd_loglogfits import main as msd_loglogfits
from helicity import compute_helicity_analysis
from machine_learning import ml_analysis, XGBAbortException
from shared_state import messages, thread_lock, complete_progress_step

def compute_object_summary(obj_data_tuple):
    obj, object_data, df_obj_calcs, category, parameters, summary_columns = obj_data_tuple

    if object_data.size == 0:
        nan_values = []
        for col in summary_columns:
            if col == 'Object ID':
                nan_values.append(obj)
            elif col == 'Category':
                nan_values.append(category)
            else:
                nan_values.append(np.nan)
        return obj, tuple(nan_values), {}

    x_val, y_val, z_val = object_data[:, 2], object_data[:, 3], object_data[:, 4]
    num_timepoints = len(object_data)
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
    outreach_ratio = max_euclid / max_path if max_path != 0 else 0

    inst_disp = df_obj_calcs['Instantaneous Displacement'].values
    valid_disp = inst_disp[(~np.isnan(inst_disp)) & (inst_disp != 0)]
    time_under = valid_disp[valid_disp < parameters['arrest_limit']] if valid_disp.size > 0 else np.array([])
    arrest_coefficient = (time_under.size * time_interval) / duration_val if duration_val != 0 else 0

    if num_timepoints < parameters['moving']:
        summary_dict = {
            'Object ID': obj,
            'Duration': duration_val,
            'Final Euclidean': final_euclid,
            'Max Euclidean': max_euclid,
            'Path Length': max_path,
            'Straightness': straightness,
            'Displacement Ratio': displacement_ratio,
            'Outreach Ratio': outreach_ratio,
            'Arrest Coefficient': arrest_coefficient,
            'Category': category,
            'Overall Euclidean Median': np.nan,
            'Maximum MSD': np.nan,
            'Median Turning Angle': np.nan,
            'Mean Velocity': np.nan,
            'Median Velocity': np.nan,
            'StDev Velocity': np.nan,
            'Mean Acceleration': np.nan,
            'Median Acceleration': np.nan,
            'StDev Acceleration': np.nan,
            'Mean Absolute Acceleration': np.nan,
            'Median Absolute Acceleration': np.nan,
            'StDev Absolute Acceleration': np.nan,
            'Convex Hull Volume': np.nan
        }

        summary_tuple = tuple(summary_dict.get(col, np.nan) for col in summary_columns)
        return obj, summary_tuple, {}

    convex_coords = np.column_stack((x_val, y_val, z_val))
    if convex_coords.shape[0] >= parameters['moving']:
        try:
            convex_hull_volume = ConvexHull(convex_coords).volume
            convex = convex_hull_volume / np.sqrt(duration_val) if duration_val != 0 else 0
        except Exception:
            convex = 0
    else:
        convex = 0

    velocity = df_obj_calcs['Instantaneous Velocity'].values[1:] if 'Instantaneous Velocity' in df_obj_calcs else np.array([])
    valid_velocity = velocity[(~np.isnan(velocity)) & (velocity != 0)] if velocity.size > 0 else np.array([])
    velocity_mean = valid_velocity.mean() if valid_velocity.size > 0 else 0
    velocity_median = np.median(valid_velocity) if valid_velocity.size > 0 else 0
    velocity_stdev = valid_velocity.std(ddof=1) if valid_velocity.size > 1 else 0

    acceleration = df_obj_calcs['Instantaneous Acceleration'].values if 'Instantaneous Acceleration' in df_obj_calcs else np.array([])
    if acceleration.size >= parameters['moving']:
        valid_acc = acceleration[(~np.isnan(acceleration)) & (acceleration != 0)]
        acceleration_mean = valid_acc.mean() if valid_acc.size > 0 else 0
        acceleration_median = np.median(valid_acc) if valid_acc.size > 0 else 0
        acceleration_stdev = valid_acc.std(ddof=1) if valid_acc.size > 1 else 0
        accel_abs = np.abs(valid_acc)
        accel_abs_mean = accel_abs.mean() if accel_abs.size > 0 else 0
        accel_abs_median = np.median(accel_abs) if accel_abs.size > 0 else 0
        accel_abs_stdev = accel_abs.std(ddof=1) if accel_abs.size > 1 else 0
    else:
        acceleration_mean = acceleration_median = acceleration_stdev = 0
        accel_abs_mean = accel_abs_median = accel_abs_stdev = 0

    acceleration_filtered = df_obj_calcs['Instantaneous Acceleration Filtered'].values if 'Instantaneous Acceleration Filtered' in df_obj_calcs else np.array([])
    valid_acc_f = acceleration_filtered[(~np.isnan(acceleration_filtered)) & (acceleration_filtered != 0)] if acceleration_filtered.size > 0 else np.array([])
    if valid_acc_f.size >= parameters['moving']:
        velocity_filtered = df_obj_calcs['Instantaneous Velocity Filtered'].values if 'Instantaneous Velocity Filtered' in df_obj_calcs else np.array([])
        valid_velocity_f = velocity_filtered[(~np.isnan(velocity_filtered)) & (velocity_filtered != 0)] if velocity_filtered.size > 0 else np.array([])
        velocity_filtered_mean = valid_velocity_f.mean() if valid_velocity_f.size > 0 else 0
        velocity_filtered_median = np.median(valid_velocity_f) if valid_velocity_f.size > 0 else 0
        velocity_filtered_stdev = valid_velocity_f.std(ddof=1) if valid_velocity_f.size > 1 else 0
        acceleration_filtered_mean = valid_acc_f.mean()
        acceleration_filtered_median = np.median(valid_acc_f)
        acceleration_filtered_stdev = valid_acc_f.std(ddof=1) if valid_acc_f.size > 1 else 0
        accel_filtered_abs = np.abs(valid_acc_f)
        accel_filtered_abs_mean = accel_filtered_abs.mean() if accel_filtered_abs.size > 0 else 0
        accel_filtered_abs_median = np.median(accel_filtered_abs) if accel_filtered_abs.size > 0 else 0
        accel_filtered_abs_stdev = accel_filtered_abs.std(ddof=1) if accel_filtered_abs.size > 1 else 0
    else:
        velocity_filtered_mean = velocity_filtered_median = velocity_filtered_stdev = 0
        acceleration_filtered_mean = acceleration_filtered_median = acceleration_filtered_stdev = 0
        accel_filtered_abs_mean = accel_filtered_abs_median = accel_filtered_abs_stdev = 0

    if parameters['arrest_limit'] == 0:
        velocity_mean_out = velocity_mean
        velocity_median_out = velocity_median
        velocity_stdev_out = velocity_stdev
        acceleration_mean_out = acceleration_mean
        acceleration_median_out = acceleration_median
        acceleration_stdev_out = acceleration_stdev
        abs_acc_mean_out = accel_abs_mean
        abs_acc_median_out = accel_abs_median
        abs_acc_stdev_out = accel_abs_stdev
    else:
        velocity_mean_out = velocity_filtered_mean
        velocity_median_out = velocity_filtered_median
        velocity_stdev_out = velocity_filtered_stdev
        acceleration_mean_out = acceleration_filtered_mean
        acceleration_median_out = acceleration_filtered_median
        acceleration_stdev_out = acceleration_filtered_stdev
        abs_acc_mean_out = accel_filtered_abs_mean
        abs_acc_median_out = accel_filtered_abs_median
        abs_acc_stdev_out = accel_filtered_abs_stdev

    cols_euclidean = [col for col in df_obj_calcs.columns if 'Euclid' in col]
    single_euclidean = {}

    if cols_euclidean:
        number_pattern = re.compile(r"\d+")

        for col in cols_euclidean:
            match = number_pattern.search(str(col))
            if match:
                tau_num = int(match.group())
                euclidean_values = df_obj_calcs[col].values
                valid_euclidean = euclidean_values[(~np.isnan(euclidean_values)) & (euclidean_values != 0)]
                if len(valid_euclidean) > 2:
                    single_euclidean[tau_num] = np.median(valid_euclidean)
                else:
                    single_euclidean[tau_num] = np.nan

    summary_dict = {
        'Object ID': obj,
        'Duration': duration_val,
        'Final Euclidean': final_euclid,
        'Max Euclidean': max_euclid,
        'Path Length': max_path,
        'Straightness': straightness,
        'Displacement Ratio': displacement_ratio,
        'Outreach Ratio': outreach_ratio,
        'Mean Velocity': velocity_mean_out,
        'Median Velocity': velocity_median_out,
        'StDev Velocity': velocity_stdev_out,
        'Mean Acceleration': acceleration_mean_out,
        'Median Acceleration': acceleration_median_out,
        'StDev Acceleration': acceleration_stdev_out,
        'Mean Absolute Acceleration': abs_acc_mean_out,
        'Median Absolute Acceleration': abs_acc_median_out,
        'StDev Absolute Acceleration': abs_acc_stdev_out,
        'Arrest Coefficient': arrest_coefficient,
        'Overall Euclidean Median': None,
        'Convex Hull Volume': convex,
        'Category': category
    }

    summary_tuple = tuple(summary_dict.get(col, None) for col in summary_columns)
    return obj, summary_tuple, single_euclidean

def summary_sheet(arr_segments, df_all_calcs, unique_objects, twodim_mode, parameters, arr_cats, savefile,
                  angle_steps, all_angle_medians, df_removed):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented")

    summary_columns = [
        'Object ID', 'Category', 'Duration', 'Path Length', 'Final Euclidean', 'Max Euclidean',
        'Overall Euclidean Median', 'Maximum MSD', 'Displacement Ratio', 'Outreach Ratio', 'Straightness',
        'Median Turning Angle', 'Mean Velocity', 'Median Velocity', 'StDev Velocity', 'Mean Acceleration',
        'Median Acceleration', 'StDev Acceleration', 'Mean Absolute Acceleration',
        'Median Absolute Acceleration', 'StDev Absolute Acceleration'
    ]
    if parameters['arrest_limit'] != 0:
        idx = summary_columns.index('Median Turning Angle') + 1
        summary_columns.insert(idx, 'Arrest Coefficient')
    if not twodim_mode:
        idx = summary_columns.index('Maximum MSD') + 1
        summary_columns.insert(idx, 'Convex Hull Volume')

    n_summary_cols = len(summary_columns)

    with thread_lock:
        messages.append("Calculating mean square displacements...")
    tic = tempo.time()
    tau = parameters["tau"]
    df_msd = msd_parallel_main(arr_segments, unique_objects, tau)
    toc = tempo.time()
    with thread_lock:
        msg = " MSD calculations done in {:.0f} seconds.".format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append("")
    complete_progress_step("MSD")

    df_helicity = None
    if parameters['helicity']:
        if twodim_mode:
            with thread_lock:
                messages.append('Helicity skipped, 3D dataset required.')
                messages.append('')
            complete_progress_step('Helicity')
        else:
            with thread_lock:
                messages.append('Calculating helicity...')
            tic = tempo.time()

            df_helicity = compute_helicity_analysis(arr_segments, arr_cats, parameters)
            toc = tempo.time()
            with thread_lock:
                msg = " Helicity done in {:.0f} seconds.".format(int(round((toc - tic), 1)))
                messages[-1] += msg
                messages.append('')
            complete_progress_step('Helicity')

    df_all_calcs_by_obj = dict(tuple(df_all_calcs.groupby("Object ID")))

    with thread_lock:
        messages.append("Calculating summary features...")
    tic = tempo.time()

    arr_segments_by_obj = {}
    for obj in unique_objects:
        arr_segments_by_obj[obj] = arr_segments[arr_segments[:, 0] == obj, :]

    category_dict = {}
    if arr_cats.shape[0] > 0:
        arr_cats_int = arr_cats[:, 0].astype(int)
        for i, obj_id in enumerate(arr_cats_int):
            category_dict[int(obj_id)] = str(arr_cats[i, 1])

    worker_data = []
    for obj in unique_objects:
        object_data = arr_segments_by_obj[obj]
        df_obj_calcs = df_all_calcs_by_obj.get(obj, pd.DataFrame())
        if obj in category_dict:
            category = category_dict[obj]
        else:
            category = 'Unknown'
            print(f"Warning: Object {obj} not found in category data")
        worker_data.append((obj, object_data, df_obj_calcs, category, parameters, summary_columns))

    sum_ = {}
    single_euclid_dict = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(mp.cpu_count()-1, len(unique_objects))) as executor:
        try:
            futures = {executor.submit(compute_object_summary, data): data[0] for data in worker_data}
            for future in concurrent.futures.as_completed(futures):
                obj = futures[future]
                try:
                    obj, summary_tuple, single_euclid = future.result()
                    sum_[obj] = summary_tuple
                    single_euclid_dict[obj] = single_euclid
                except Exception:
                    obj_category = next((data[3] for data in worker_data if data[0] == obj), 'Unknown')
                    nan_values = []
                    for i, col in enumerate(summary_columns):
                        if col == 'Object ID':
                            nan_values.append(obj)
                        elif col == 'Category':
                            nan_values.append(obj_category)
                        else:
                            nan_values.append(np.nan)

                    sum_[obj] = tuple(nan_values)
                    single_euclid_dict[obj] = {}
        except Exception:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(compute_object_summary, data): data[0] for data in worker_data}
                for future in concurrent.futures.as_completed(futures):
                    obj = futures[future]
                    try:
                        obj, summary_tuple, single_euclid = future.result()
                        sum_[obj] = summary_tuple
                        single_euclid_dict[obj] = single_euclid
                    except Exception:
                        obj_category = next((data[3] for data in worker_data if data[0] == obj), 'Unknown')
                        nan_values = []
                        for i, col in enumerate(summary_columns):
                            if col == 'Object ID':
                                nan_values.append(obj)
                            elif col == 'Category':
                                nan_values.append(obj_category)
                            else:
                                nan_values.append(np.nan)

                        sum_[obj] = tuple(nan_values)
                        single_euclid_dict[obj] = {}

    for obj in unique_objects:
        if obj not in sum_:
            obj_category = next((data[3] for data in worker_data if data[0] == obj), 'Unknown')
            nan_values = []
            for i, col in enumerate(summary_columns):
                if col == 'Object ID':
                    nan_values.append(obj)
                elif col == 'Category':
                    nan_values.append(obj_category)
                else:
                    nan_values.append(np.nan)

            sum_[obj] = tuple(nan_values)
            single_euclid_dict[obj] = {}

    summary_rows = [sum_[obj] for obj in unique_objects]
    df_sum = pd.DataFrame(summary_rows, columns=summary_columns)
    df_sum["Object ID"] = df_sum["Object ID"].astype(int)

    df_single_euclids = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_euclids.reset_index(inplace=True)
    cols_euclidean_numbers = [
        int(re.search(r"\d+", str(col)).group()) if re.search(r"\d+", str(col)) else col
        for col in df_single_euclids.columns[1:]]
    df_single_euclids.columns = ["Object ID"] + cols_euclidean_numbers
    df_single_euclids = df_single_euclids.sort_values(by="Object ID").reset_index(drop=True)
    df_single_euclids["Object ID"] = df_single_euclids["Object ID"].astype(int)
    df_single_euclids = df_single_euclids.dropna(axis=1, how='all')

    df_single_angles = pd.DataFrame.from_dict(all_angle_medians, orient='index')
    df_single_angles.reset_index(inplace=True)
    df_single_angles.columns = ["Object ID"] + list(angle_steps)
    df_single_angles = df_single_angles.sort_values(by="Object ID").reset_index(drop=True)
    df_single_angles["Object ID"] = df_single_angles["Object ID"].astype(int)
    df_single_angles = df_single_angles.dropna(axis=1, how='all')

    overall_euclidean_median = df_single_euclids.drop("Object ID", axis=1).median(axis=1, skipna=True)
    df_sum["Overall Euclidean Median"] = df_sum["Object ID"].map(
        pd.Series(overall_euclidean_median.values, index=df_single_euclids['Object ID']))

    angle_step_cols = [col for col in df_single_angles.columns if isinstance(col, int)]
    median_max_angle = df_single_angles[angle_step_cols].max(axis=1)

    df_sum['Median Turning Angle'] = df_sum['Object ID'].map(
        pd.Series(median_max_angle.values, index=df_single_angles['Object ID']))
    df_msd = df_msd.dropna(axis=1, how='all')
    existing_cols = [col for col in range(1, tau + 1) if col in df_msd.columns]
    df_msd = df_msd[["Object ID"] + existing_cols]
    df_msd["Object ID"] = df_msd["Object ID"].astype(int)

    msd_vals = df_msd.set_index("Object ID")[existing_cols]
    category_tracks = arr_cats[:, 1]
    if len(category_tracks) == len(msd_vals):
        msd_vals["Category"] = category_tracks
        grouped = msd_vals.groupby("Category")
        df_msd_avg_per_cat = grouped.mean().T
        df_msd_std_per_cat = grouped.std().T
        df_msd_avg_per_cat.index.name = "MSD"
    else:
        df_msd_avg_per_cat = pd.DataFrame()
        df_msd_std_per_cat = pd.DataFrame()

    msd_vals_summary = msd_vals.drop(columns="Category", errors="ignore")
    df_msd_sum_all = pd.DataFrame({
        "Mean": msd_vals_summary.mean(),
        "StDev": msd_vals_summary.std()})
    df_msd_sum_all.index.name = "MSD"

    category_df = pd.DataFrame(arr_cats, columns=["Object ID", "Category"])
    if not category_df.empty:
        category_df["Object ID"] = category_df["Object ID"].astype(int)
        category_df["Category"] = category_df["Category"].astype(str)
        def insert_category(df):
            merged = pd.merge(df, category_df, on="Object ID", how="left")
            merged["Category"] = merged["Category"].astype(str)
            cols = merged.columns.tolist()
            cols.insert(1, cols.pop(cols.index("Category")))
            return merged[cols]
        df_single_euclids = insert_category(df_single_euclids)
        df_single_angles = insert_category(df_single_angles)
        df_msd = insert_category(df_msd)
        df_msd["Category"] = df_msd["Category"].astype(str)

        if 'Category' in df_msd.columns and df_msd['Category'].nunique() > 1:
            msd_vals_with_cats = df_msd.set_index("Object ID")[existing_cols + ["Category"]]
            if len(msd_vals_with_cats) > 0:
                grouped = msd_vals_with_cats.groupby("Category")
                df_msd_avg_per_cat = grouped[existing_cols].mean().T
                df_msd_std_per_cat = grouped[existing_cols].std().T
                df_msd_avg_per_cat.index.name = "MSD"
                df_msd_std_per_cat.index.name = "MSD"

    max_msd_dict = {}
    msd_cols = [col for col in df_msd.columns if isinstance(col, int)]
    for obj in unique_objects:
        obj_val = int(obj)
        msd_row = df_msd[df_msd["Object ID"] == obj_val]
        if not msd_row.empty and msd_cols:
            max_msd = msd_row[msd_cols].max(axis=1, skipna=True).values[0]
        else:
            max_msd = np.nan
        max_msd_dict[obj_val] = max_msd

    df_sum["Maximum MSD"] = df_sum["Object ID"].map(max_msd_dict)

    for col in summary_columns:
        if col not in df_sum.columns:
            df_sum[col] = np.nan
    df_sum = df_sum[summary_columns]

    min_maxeuclid = parameters.get('min_maxeuclid', 0)
    euclidean_filtered_count = 0

    if min_maxeuclid > 0:
        pre_filter_count = len(df_sum)
        filtered_objects = df_sum[df_sum['Max Euclidean'] < min_maxeuclid]['Object ID'].tolist()
        df_sum = df_sum[df_sum['Max Euclidean'] >= min_maxeuclid].copy()
        post_filter_count = len(df_sum)
        euclidean_filtered_count = pre_filter_count - post_filter_count

        if euclidean_filtered_count > 0:
            euclidean_removed = pd.DataFrame({
                'Object ID': filtered_objects,
                'Reason for removal': ['Min. Max. Euclidean'] * len(filtered_objects)
            })
            euclidean_removed['Object ID'] = euclidean_removed['Object ID'].astype(int)
            df_removed = pd.concat([df_removed, euclidean_removed], ignore_index=True)

            filtered_object_ids = df_sum['Object ID'].tolist()
            df_single_euclids = df_single_euclids[df_single_euclids['Object ID'].isin(filtered_object_ids)].copy()
            df_single_angles = df_single_angles[df_single_angles['Object ID'].isin(filtered_object_ids)].copy()
            df_msd = df_msd[df_msd['Object ID'].isin(filtered_object_ids)].copy()

            if 'Category' in df_msd.columns and df_msd['Category'].nunique() > 1:
                msd_vals_with_cats = df_msd.set_index("Object ID")[existing_cols + ["Category"]]
                if len(msd_vals_with_cats) > 0:
                    grouped = msd_vals_with_cats.groupby("Category")
                    df_msd_avg_per_cat = grouped[existing_cols].mean().T
                    df_msd_std_per_cat = grouped[existing_cols].std().T
                    df_msd_avg_per_cat.index.name = "MSD"
                    df_msd_std_per_cat.index.name = "MSD"

            msd_vals = df_msd.set_index("Object ID")[existing_cols]

            msd_vals_summary = msd_vals.drop(columns="Category", errors="ignore")
            df_msd_sum_all = pd.DataFrame({
                "Mean": msd_vals_summary.mean(),
                "StDev": msd_vals_summary.std()})
            df_msd_sum_all.index.name = "MSD"

    df_msd_loglogfits = msd_loglogfits(df_msd)
    df_msd_loglogfits.columns = df_msd_loglogfits.columns.astype(str)

    if df_helicity is not None and not df_helicity.empty:
        helicity_metrics = ['Mean Helicity', 'Median Helicity', 'Mean Curvature', 'Median Curvature']
        helicity_cols = ['Object ID'] + helicity_metrics
        df_helicity_subset = df_helicity[helicity_cols].copy()
        df_sum = df_sum.merge(df_helicity_subset, on='Object ID', how='left')

    toc = tempo.time()
    with thread_lock:
        msg = " Summary features done in {:.0f} seconds.".format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append("")
    complete_progress_step("Summary")

    df_pca = None
    if parameters.get("infile_categories", False):
        try:
            df_pca = ml_analysis(df_sum.copy(), parameters, savefile)
        except XGBAbortException:
            pass

    return (df_sum, df_single_euclids, df_single_angles, df_msd, df_msd_sum_all,
            df_msd_avg_per_cat, df_msd_std_per_cat, df_msd_loglogfits, df_pca, df_removed, euclidean_filtered_count)
