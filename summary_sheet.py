import numpy as np
import pandas as pd
import statistics
import time as tempo
import warnings
from scipy.spatial import ConvexHull
from overall_medians import overall_medians
from PCA import pca
from xgb import xgboost

def summary_sheet(arr_segments, df_all_calcs, unique_objects, tau_msd, parameters, arr_tracks, savefile):
    """
    Generates a summary sheet with various migration parameters and statistics for each object.
    Args:
        arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
        df_all_calcs (pandas.DataFrame): DataFrame containing all calculated migration parameters.
        unique_objects (numpy.ndarray): Array of unique object IDs.
        tau_msd (int): The maximum time differential for mean squared displacement (MSD) calculation.
        parameters (dict): Dictionary containing user-defined parameters for the analysis.
        arr_tracks (numpy.ndarray): Array of tracks with columns [object_id, category].
        savefile (str): Path to save the output files.
    Returns:
        tuple: DataFrames containing summary statistics, time interval, single timepoint medians, MSD values,
               MSD summaries for all objects, and MSD summaries per category.
    """
    tic = tempo.time()
    print('Running Summary Sheet...')
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
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
        # Vectorized MSD computation for every t_diff
        for t_diff in range(1, tau_msd + 1):
            if x_val.size <= t_diff:
                vals_msd.append(0)
                continue
            sq_diff = np.square(x_val[t_diff:] - x_val[:-t_diff]) + \
                      np.square(y_val[t_diff:] - y_val[:-t_diff]) + \
                      np.square(z_val[t_diff:] - z_val[:-t_diff])
            valid = sq_diff > 0
            msd = np.mean(sq_diff[valid]) if np.any(valid) else 0
            vals_msd.append(msd)
        msd_dict[obj] = vals_msd

        # Retrieve category from tracks
        category = ''
        if arr_tracks.shape[0] > 0:
            obj_id_str = str(int(object_data[0, 0]))
            matching_index = np.where(arr_tracks[:, 0].astype(str) == obj_id_str)[0]
            if matching_index.size > 0:
                category = arr_tracks[matching_index[0], 1]

        # Calculate convex hull volume
        convex_coords = np.array([x_val, y_val, z_val]).transpose()
        if convex_coords.shape[0] < 4 or parameters['two_dim']:
            convex_hull_volume = 0
        else:
            try:
                convex_hull = ConvexHull(convex_coords)
                convex_hull_volume = convex_hull.volume
            except Exception:
                convex_hull_volume = 0

        # Calculate summary statistics from df_all_calcs
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

        # Vectorized filtering for velocity and acceleration columns
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

    df_sum = pd.DataFrame.from_dict(sum_, orient='index')
    df_msd = pd.DataFrame.from_dict(msd_dict, orient='index')
    df_msd.columns = [f'MSD {i}' for i in range(1, tau_msd + 1)]
    objs = list(single_euclid_dict.keys())
    df_single_euclids = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_euclids.columns = cols_euclidean
    df_single_angles = pd.DataFrame.from_dict(single_angle_dict, orient='index')
    df_single_angles.columns = [f"Filtered Angle {(2 * i) - 1}" for i in range(2, df_single_angles.shape[1] + 2)]
    df_single = pd.concat([df_single_euclids, df_single_angles], axis=1)
    df_single.insert(0, 'Object ID', objs)

    msd_means_stdev_all = {}
    msd_means_stdev_per_cat = {}

    df_msd_sum_cat = pd.DataFrame()
    for msd_col in df_msd.columns:
        column_vals = list(df_msd.loc[:, str(msd_col)])
        if len(column_vals) < 2:
            pass
        else:
            msd_means_stdev_all[msd_col] = [np.nanmean(column_vals), np.nanstd(column_vals)]

    if parameters['infile_tracks']:
        category_tracks = arr_tracks[:,1]
        df_msd["Category"] = category_tracks
        all_cat = list(df_msd.loc[:, 'Category'])

        # Get unique categories
        unique_cat = []
        for x in all_cat:
            if x not in unique_cat:
                unique_cat.append(x)
            else:
                pass

        # Calculate MSD per category
        for cat in unique_cat:
            for msd in df_msd.columns[:-1]:
                per_cat = list(df_msd.loc[df_msd['Category'] == cat, str(msd)])
                msd_means_stdev_per_cat[f"Category {cat}, {str(msd)}"] = [np.nanmean(per_cat), np.nanstd(per_cat)]

    df_msd.insert(0, 'Object ID', objs)
    df_msd_sum_all = pd.DataFrame.from_dict(msd_means_stdev_all)
    df_msd_sum_all.index = ["Average", 'Standard Deviation']
    df_msd_sum_all = df_msd_sum_all.transpose()
    if parameters['infile_tracks']:
        df_msd_sum_cat = pd.DataFrame.from_dict(msd_means_stdev_per_cat)
        df_msd_sum_cat.index = ['Average', 'Standard Deviation']
        df_msd_sum_cat = df_msd_sum_cat.transpose()
    else:
        df_msd_sum_cat = pd.DataFrame()

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
        'Time Corrected Convex Hull Volume', parameters['category_col']
    ]
    toc = tempo.time()
    print('...Summary sheet done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    if parameters['infile_tracks']:
        print('Object category input required for PCA and xgboost found! Running PCA and xgboost...')
        pca(df_sum, parameters, savefile)
        xgboost(df_sum, parameters, savefile)
    else:
        print('Object category input required for PCA and xgboost not found. Skipping PCA and xgboost.')

    return df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_sum_cat