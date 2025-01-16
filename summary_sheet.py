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

    tic = tempo.time()
    print('Running Summary Sheet...')
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    sum_ = {}
    single_euclid_dict = {}
    single_angle_dict = {}
    msd_dict = {}
    time_interval = False
    category = 0

    # Calculate summary statistics for each object
    for object in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == object, :]
        object_id = object_data[:, 0]
        x_val = object_data[:, 2]
        y_val = object_data[:, 3]
        z_val = object_data[:, 4]
        vals_msd = []
        # For every coordinate (if index is less than time differential, pass) calculate msd
        for t_diff in range(1, tau_msd + 1):
            displacement_list = []
            for index, coord in enumerate(x_val):
                if index < t_diff:
                    pass
                else:
                    x_diff = (x_val[index] - x_val[index - t_diff])**2
                    y_diff = (y_val[index] - y_val[index - t_diff])**2
                    z_diff = (z_val[index] - z_val[index - t_diff])**2
                    displacement = np.sqrt(x_diff + y_diff + z_diff)
                    squared_displacement = displacement**2
                    if squared_displacement <= 0:
                        pass
                    else:
                        displacement_list.append(squared_displacement)
            msd = np.mean(displacement_list)
            vals_msd.append(msd)

            if t_diff == tau_msd:  # last tau value adds headers
                msd_dict[object] = vals_msd

        # If categories DataFrame was given, add category to object statistics
        if arr_tracks.shape[0] > 0:
            category = ''
            object_id_str = str(int(object_data[0, 0]))
            matching_index = np.where(arr_tracks[:, 0] == object_id_str)[0]
            if matching_index.size > 0:
                # Pull the category corresponding to the matched ID
                category = arr_tracks[matching_index[0], 1]

        # Calculate convex hull volume
        convex_coords = np.array([x_val, y_val, z_val]).transpose()
        if convex_coords.shape[0] < 4 or parameters['two_dim']:
            convex_hull_volume = 0
        else:
            try:
                convex_hull = ConvexHull(convex_coords)
                convex_hull_volume = convex_hull.volume
            except Exception as e:
                convex_hull_volume = 0

        # Calculate summary statistics
        max_path = df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Path Length'].max()
        final_euclid = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Total Displacement'])
        max_euclid = df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Total Displacement'].max()
        duration = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Time'])
        time_interval = np.abs(duration[1] - duration[0])
        duration = len(duration) * time_interval
        final_euclid = final_euclid[-1]
        straightness = final_euclid / max_path
        tc_straightness = straightness * np.sqrt(duration)
        displacement_ratio = final_euclid / max_euclid
        tc_convex = convex_hull_volume / np.sqrt(duration)
        outreach_ratio = max_euclid / max_path
        velocity = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Instantaneous Velocity'])
        velocity_filtered = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Instantaneous Velocity Filtered'])
        velocity_filtered = [x for x in velocity_filtered if np.isnan(x) == False and x != 0]
        velocity.pop(0)
        velocity_mean = statistics.mean(velocity)
        velocity_median = statistics.median(velocity)
        acceleration = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Instantaneous Acceleration'])
        if len(acceleration) >= parameters['moving']:
            acceleration = [x for x in acceleration if np.isnan(x) == False and x != 0]
            acceleration_mean = statistics.mean(acceleration)
            acceleration_median = statistics.median(acceleration)
            accel_abs = np.absolute(acceleration)  # convert acceleration values to absolute
            accel_abs_mean = statistics.mean(accel_abs)
            accel_abs_median = statistics.median(accel_abs)
        elif len(acceleration) < parameters['moving']:
            acceleration_mean = 0
            acceleration_median = 0
            accel_abs_mean = 0
            accel_abs_median = 0
        acceleration_filtered = np.array(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Instantaneous Acceleration Filtered'])
        acceleration_filtered = [x for x in acceleration_filtered if np.isnan(x) == False and x != 0]
        if len(acceleration_filtered) >= parameters['moving']:
            velocity_filtered_mean = statistics.mean(velocity_filtered)
            velocity_filtered_median = statistics.median(velocity_filtered)
            acceleration_filtered_mean = statistics.mean(acceleration_filtered)
            acceleration_filtered_median = statistics.median(acceleration_filtered)
            acceleration_filtered_stdev = statistics.stdev(acceleration_filtered)
            accel_filtered_abs = np.absolute(acceleration_filtered)  # convert acceleration filtered values to absolute
            accel_filtered_abs_mean = statistics.mean(accel_filtered_abs)
            accel_filtered_abs_median = statistics.median(accel_filtered_abs)
            accel_filtered_abs_stdev = statistics.stdev(accel_filtered_abs)
            velocity_filtered_stdev = statistics.stdev(velocity_filtered)
        else:
            velocity_filtered_mean = 0
            velocity_filtered_median = 0
            acceleration_filtered_mean = 0
            acceleration_filtered_median = 0
            acceleration_filtered_stdev = 0
            accel_filtered_abs_mean = 0
            accel_filtered_abs_median = 0
            accel_filtered_abs_stdev = 0
            velocity_filtered_stdev = 0

        cols_angles = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object,
                           df_all_calcs.columns[df_all_calcs.columns.str.contains('Filtered Angle')]])
        cols_euclidean = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object,
                              df_all_calcs.columns[df_all_calcs.columns.str.contains('Euclid')]])
        # Calculate overall medians
        overall_euclidean_median, overall_angle_median, single_euclid, single_angle = overall_medians(object, df_all_calcs,
                                                                                                      cols_angles,
                                                                                                      cols_euclidean)

        single_euclid_dict[object] = single_euclid
        single_angle_dict[object] = single_angle

        instantaneous_displacement = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, 'Instantaneous Displacement'])
        instantaneous_displacement = [x for x in instantaneous_displacement if not (np.isnan(x) or x == 0)]
        time_under = [x for x in instantaneous_displacement if
                      x < parameters['arrest_limit']]
        arrest_coefficient = (len(time_under) * time_interval) / duration
        # Combine summary statistics for each object and add to dictionary
        sum_[object] = object, duration, final_euclid, max_euclid, max_path, straightness, tc_straightness, \
                     displacement_ratio, outreach_ratio, velocity_mean, velocity_median, velocity_filtered_mean, \
                     velocity_filtered_median, velocity_filtered_stdev, acceleration_mean, acceleration_median, \
                     accel_abs_mean, accel_abs_median, acceleration_filtered_mean, acceleration_filtered_median, \
                     acceleration_filtered_stdev, accel_filtered_abs_mean, accel_filtered_abs_median, \
                     accel_filtered_abs_stdev, arrest_coefficient, overall_angle_median, overall_euclidean_median, \
                     convex_hull_volume, tc_convex, category

    # Create DataFrames of results
    df_sum = pd.DataFrame.from_dict(sum_, orient='index')
    df_msd = pd.DataFrame.from_dict(msd_dict, orient='index')
    df_msd.columns = ['MSD ' + str(x) for x in range(1, tau_msd + 1)]
    objs = list(single_euclid_dict.keys())
    df_single_euclids = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_euclids.columns = cols_euclidean
    df_single_angles = pd.DataFrame.from_dict(single_angle_dict, orient='index')
    df_single_angles.columns = [f"Filtered Angle {(x * 2) - 1}" for x in range(2, df_single_angles.shape[1] + 2)]
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

    df_sum.columns = ['Object ID', 'Duration', 'Final Euclidean', 'Max Euclidean',
                      'Path Length', 'Straightness', 'Time Corrected Straightness',
                      'Displacement Ratio', 'Outreach Ratio', 'Velocity Mean',
                      'Velocity Median', 'Velocity filtered Mean', 'Velocity Filtered Median',
                      'Velocity Filtered Standard Deviation', 'Acceleration Mean', 'Acceleration Median',
                      'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
                      'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
                      'Absolute Acceleration Filtered Mean', 'Absolute Acceleration Filtered Median',
                      'Absolute Acceleration Filtered Standard Deviation',
                      'Arrest Coefficient', 'Overall Angle Median', 'Overall Euclidean Median',
                      'Convex Hull Volume', 'Time Corrected Convex Hull Volume', 'Category']
    toc = tempo.time()
    print('...Summary sheet done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))


    # If categories file is supplied, run PCA
    if parameters['infile_tracks']:
        print('Object category input required for PCA found! Running PCA...')
        # TODO: ADD PCA IN LOL
        # pca(df_sum, parameters, savefile)
        xgboost(df_sum, parameters, savefile)


    else:
        print('Object category input required for PCA not found. Skipping PCA.')


    return df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_sum_cat