import numpy as np
import pandas as pd
import re
import statistics
import time as tempo
import warnings
from pandas.errors import PerformanceWarning
from scipy.spatial import ConvexHull

from overall_medians import overall_medians
from PCA import pca
from xgb import xgboost
from shared_state import messages, thread_lock

def summary_sheet(arr_segments, df_all_calcs, unique_objects, tau_msd, parameters, arr_tracks, savefile):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented")

    with thread_lock:
        messages.append('Running Summary Sheet...')

    tic = tempo.time()
    sum_ = {}
    single_euclid_dict = {}
    single_angle_dict = {}
    msd_dict = {}
    time_interval = False

    for obj in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        object_id = object_data[:, 0]
        x_val = object_data[:, 2]
        y_val = object_data[:, 3]
        z_val = object_data[:, 4]
        vals_msd = []

        for t_diff in range(1, tau_msd + 1):
            if x_val.size <= t_diff:
                vals_msd.append(None)
                continue
            sq_diff = np.square(x_val[t_diff:] - x_val[:-t_diff]) + \
                      np.square(y_val[t_diff:] - y_val[:-t_diff]) + \
                      np.square(z_val[t_diff:] - z_val[:-t_diff])
            valid = sq_diff > 0
            msd = np.mean(sq_diff[valid]) if np.any(valid) else None
            vals_msd.append(msd)
        msd_dict[obj] = vals_msd

        category = ''
        if arr_tracks.shape[0] > 0:
            obj_id_str = str(int(object_data[0, 0]))
            matching_index = np.where(arr_tracks[:, 0].astype(str) == obj_id_str)[0]
            if matching_index.size > 0:
                category = arr_tracks[matching_index[0], 1]

        convex_coords = np.array([x_val, y_val, z_val]).transpose()
        if convex_coords.shape[0] < 4:
            convex_hull_volume = 0
        else:
            try:
                convex_hull = ConvexHull(convex_coords)
                convex_hull_volume = convex_hull.volume
            except Exception:
                convex_hull_volume = 0

        max_path = df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Path Length'].max()
        duration_list = list(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Time'])
        time_interval = abs(duration_list[1] - duration_list[0])
        duration_val = len(duration_list) * time_interval

        final_euclid = list(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Total Displacement'])[-1]
        max_euclid = df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Total Displacement'].max()

        straightness = final_euclid / max_path
        tc_straightness = straightness * np.sqrt(duration_val)
        displacement_ratio = final_euclid / max_euclid if max_euclid != 0 else 0
        tc_convex = convex_hull_volume / np.sqrt(duration_val)
        outreach_ratio = max_euclid / max_path if max_path != 0 else 0

        velocity_all = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Instantaneous Velocity'])
        velocity = velocity_all[1:]
        valid_velocity = velocity[~np.isnan(velocity) & (velocity != 0)]
        velocity_mean = statistics.mean(valid_velocity) if valid_velocity.size > 0 else 0
        velocity_median = statistics.median(valid_velocity) if valid_velocity.size > 0 else 0

        acceleration_all = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Instantaneous Acceleration'])
        if acceleration_all.size >= parameters['moving']:
            valid_acc = acceleration_all[~np.isnan(acceleration_all) & (acceleration_all != 0)]
            acceleration_mean = statistics.mean(valid_acc) if valid_acc.size > 0 else 0
            acceleration_median = statistics.median(valid_acc) if valid_acc.size > 0 else 0
            accel_abs = np.abs(valid_acc)
            accel_abs_mean = statistics.mean(accel_abs) if accel_abs.size > 0 else 0
            accel_abs_median = statistics.median(accel_abs) if accel_abs.size > 0 else 0
        else:
            acceleration_mean = acceleration_median = accel_abs_mean = accel_abs_median = 0

        acceleration_filtered_all = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Instantaneous Acceleration Filtered'])
        valid_acc_f = acceleration_filtered_all[~np.isnan(acceleration_filtered_all) & (acceleration_filtered_all != 0)]
        if valid_acc_f.size >= parameters['moving']:
            velocity_filtered_all = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Instantaneous Velocity Filtered'])
            valid_velocity_f = velocity_filtered_all[~np.isnan(velocity_filtered_all) & (velocity_filtered_all != 0)]
            velocity_filtered_mean = statistics.mean(valid_velocity_f) if valid_velocity_f.size > 0 else 0
            velocity_filtered_median = statistics.median(valid_velocity_f) if valid_velocity_f.size > 0 else 0
            acceleration_filtered_mean = statistics.mean(valid_acc_f)
            acceleration_filtered_median = statistics.median(valid_acc_f)
            acceleration_filtered_stdev = statistics.stdev(valid_acc_f) if valid_acc_f.size > 1 else 0
            accel_filtered_abs = np.abs(valid_acc_f)
            accel_filtered_abs_mean = statistics.mean(accel_filtered_abs) if accel_filtered_abs.size > 0 else 0
            accel_filtered_abs_median = statistics.median(accel_filtered_abs) if accel_filtered_abs.size > 0 else 0
            accel_filtered_abs_stdev = statistics.stdev(accel_filtered_abs) if accel_filtered_abs.size > 1 else 0
            velocity_filtered_stdev = statistics.stdev(valid_velocity_f) if valid_velocity_f.size > 1 else 0
        else:
            velocity_filtered_mean = velocity_filtered_median = 0
            acceleration_filtered_mean = acceleration_filtered_median = acceleration_filtered_stdev = 0
            accel_filtered_abs_mean = accel_filtered_abs_median = accel_filtered_abs_stdev = 0
            velocity_filtered_stdev = 0

        cols_angles = list(df_all_calcs.loc[df_all_calcs['Object ID'] == obj,
                           df_all_calcs.columns[df_all_calcs.columns.str.contains('Filtered Angle')]])
        cols_euclidean = list(df_all_calcs.loc[df_all_calcs['Object ID'] == obj,
                              df_all_calcs.columns[df_all_calcs.columns.str.contains('Euclid')]])

        overall_euclidean_median, overall_angle_median, single_euclid, single_angle = overall_medians(obj, df_all_calcs,
                                                                                                      cols_angles,
                                                                                                      cols_euclidean)

        single_euclid_dict[obj] = single_euclid
        single_angle_dict[obj] = single_angle

        inst_disp = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == obj, 'Instantaneous Displacement'])
        valid_disp = inst_disp[~np.isnan(inst_disp) & (inst_disp != 0)]
        time_under = valid_disp[valid_disp < parameters['arrest_limit']]
        arrest_coefficient = (time_under.size * time_interval) / duration_val if duration_val != 0 else 0

        sum_[obj] = (
            obj, duration_val, final_euclid, max_euclid, max_path, straightness,
            tc_straightness, displacement_ratio, outreach_ratio, velocity_mean,
            velocity_median, velocity_filtered_mean, velocity_filtered_median,
            velocity_filtered_stdev, acceleration_mean, acceleration_median,
            accel_abs_mean, accel_abs_median, acceleration_filtered_mean, acceleration_filtered_median,
            acceleration_filtered_stdev, accel_filtered_abs_mean, accel_filtered_abs_median,
            accel_filtered_abs_stdev, arrest_coefficient, overall_angle_median,
            overall_euclidean_median, convex_hull_volume, tc_convex, category
        )

    df_single_euclids_df = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_euclids_df.reset_index(inplace=True)
    cols_euclidean_numbers = [
        re.search(r'\d+', str(col)).group() if re.search(r'\d+', str(col)) else str(col)
        for col in df_single_euclids_df.columns[1:]
    ]
    df_single_euclids_df.columns = ['Object ID'] + cols_euclidean_numbers

    df_single_angles_df = pd.DataFrame.from_dict(single_angle_dict, orient='index')
    df_single_angles_df.reset_index(inplace=True)
    angle_col_numbers = [
        re.search(r'\d+', str(col)).group() if re.search(r'\d+', str(col)) else str(col)
        for col in df_single_angles_df.columns[1:]
    ]
    df_single_angles_df.columns = ['Object ID'] + angle_col_numbers

    df_msd = pd.DataFrame({'Object ID': list(msd_dict.keys()),
                           **pd.DataFrame.from_dict(msd_dict, orient='index').to_dict(orient='list')})
    existing_cols = [col for col in range(1, tau_msd + 1) if col in df_msd.columns]
    df_msd = df_msd[['Object ID'] + existing_cols]

    msd_vals = df_msd.set_index('Object ID')[existing_cols]
    if parameters['infile_tracks']:
        category_tracks = arr_tracks[:, 1]
        if len(category_tracks) == len(msd_vals):
            msd_vals['Category'] = category_tracks
            grouped = msd_vals.groupby('Category')
            df_msd_avg_per_cat = grouped.mean().T
            df_msd_std_per_cat = grouped.std().T
            df_msd_avg_per_cat.index.name = 'MSD'
        else:
            pass
    else:
        df_msd_avg_per_cat = pd.DataFrame()
        df_msd_std_per_cat = pd.DataFrame()

    msd_vals_summary = msd_vals.drop(columns='Category', errors='ignore')
    df_msd_sum_all = pd.DataFrame({
        'Avg': msd_vals_summary.mean(),
        'StDev': msd_vals_summary.std()
    })
    df_msd_sum_all.index.name = 'MSD'

    df_sum = pd.DataFrame.from_dict(sum_, orient='index')
    df_sum.columns = [
        'Object ID', 'Duration', 'Final Euclidean', 'Max Euclidean', 'Path Length',
        'Straightness', 'Time Corrected Straightness', 'Displacement Ratio', 'Outreach Ratio',
        'Velocity Mean', 'Velocity Median', 'Velocity filtered Mean', 'Velocity Filtered Median',
        'Velocity Filtered Standard Deviation', 'Acceleration Mean', 'Acceleration Median',
        'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
        'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
        'Absolute Acceleration Filtered Mean', 'Absolute Acceleration Filtered Median',
        'Absolute Acceleration Filtered Standard Deviation', 'Arrest Coefficient',
        'Overall Angle Median', 'Overall Euclidean Median', 'Convex Hull Volume',
        'Time Corrected Convex Hull Volume', 'Category'
    ]

    toc = tempo.time()

    with thread_lock:
        messages.append('...Summary sheet done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
        messages.append('')

    if parameters['infile_tracks']:
        with thread_lock:
            messages.append('Object category input found! Running PCA and XGBoost...')
        df_pca = pca(df_sum, parameters, savefile)
        xgboost(df_sum, parameters, savefile)
    else:
        with thread_lock:
            messages.append('Object category input not found. Skipping PCA and XGBoost.')
        df_pca = None

    return df_sum, time_interval, df_single_euclids_df, df_single_angles_df, df_msd, df_msd_sum_all, df_msd_avg_per_cat, df_msd_std_per_cat, df_pca
