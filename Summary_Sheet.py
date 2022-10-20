import numpy as np
import pandas as pd
import statistics
from scipy.spatial import ConvexHull
from Overall_Medians import overall_medians
from PCA import pca


def summary_sheet(df, cell_id, cols_angles, cols_euclidean, parent_id, df_infile, x_for, y_for, z_for, tau_val,
                  parameters, track_df):
    sum_ = {}
    single_euclid_dict = {}
    single_angle_dict = {}
    msd_dict = {}
    print('Running Summary Sheet...')
    time_interval = False
    category = 0
    for cell in cell_id:
        x_values = np.array(df_infile.loc[df_infile[parent_id] == cell, x_for])
        y_values = np.array(df_infile.loc[df_infile[parent_id] == cell, y_for])
        z_values = np.array(df_infile.loc[df_infile[parent_id] == cell, z_for])
        vals = []
        y_val_msd_slope = []

        for t_diff in range(1, tau_val + 1):
            r = np.sqrt(x_values ** 2 + y_values ** 2 + z_values ** 2)
            diff = np.diff(r, n=t_diff)
            diff_sq = diff ** 2
            vals.append(np.mean(diff_sq))
            y_val_msd_slope.append(t_diff)
            if t_diff == tau_val:
                msd_dict[f'MSD for Cell {str(cell)}'] = vals
                y_ = [i for i in range(1, len(vals) + 1)]
                vals = []
                y_ = []
        if track_df.shape[0] > 0:
            category = int(track_df.loc[track_df[parameters['parent_id2']] == cell, parameters['category_col']])
        convex_coords = np.array([x_values, y_values, z_values]).transpose()
        if convex_coords.shape[0] < 4 or parameters['two_dem']:
            convex_hull_volume = 0
        else:
            convex_hull = ConvexHull(convex_coords)
            convex_hull_volume = convex_hull.volume

        max_path = df.loc[df[parent_id] == cell, 'Path length'].max()
        final_euclid = list(df.loc[df[parent_id] == cell, 'Total Displacement'])
        max_euclid = df.loc[df[parent_id] == cell, 'Total Displacement'].max()
        duration = list(df.loc[df[parent_id] == cell, 'Time'])
        time_interval = np.abs(duration[1] - duration[0])
        duration = len(duration) * time_interval
        final_euclid = final_euclid[-1]
        straightness = final_euclid / max_path
        tc_straightness = straightness * np.sqrt(duration)
        displacement_ratio = final_euclid / max_euclid
        tc_convex = convex_hull_volume / np.sqrt(duration)
        outreach_ratio = max_euclid / max_path
        velocity = list(df.loc[df[parent_id] == cell, 'Instantaneous Velocity'])
        velocity_filtered = list(df.loc[df[parent_id] == cell, 'Instantaneous Velocity Filtered'])
        velocity_filtered = [x for x in velocity_filtered if np.isnan(x) == False and x != 0]
        velocity.pop(0)
        velocity_mean = statistics.mean(velocity)
        velocity_median = statistics.median(velocity)
        acceleration = list(df.loc[df[parent_id] == cell, 'Instantaneous Acceleration'])
        if len(acceleration) >= parameters['moving']:
            acceleration = [x for x in acceleration if np.isnan(x) == False and x != 0]
            acceleration_mean = statistics.mean(acceleration)
            acceleration_median = statistics.median(acceleration)
        elif len(acceleration) < parameters['moving']:
            acceleration_mean = 0
            acceleration_median = 0

        acceleration_filtered = list(df.loc[df[parent_id] == cell, 'Instantaneous Acceleration Filtered'])
        acceleration_filtered = [x for x in acceleration_filtered if np.isnan(x) == False and x != 0]
        if len(acceleration_filtered) >= parameters['moving']:
            velocity_filtered_mean = statistics.mean(velocity_filtered)
            velocity_filtered_median = statistics.median(velocity_filtered)
            acceleration_filtered_mean = statistics.mean(acceleration_filtered)
            acceleration_filtered_median = statistics.median(acceleration_filtered)
            acceleration_filtered_stdev = statistics.stdev(acceleration_filtered)
            velocity_filtered_stdev = statistics.stdev(velocity_filtered)
        else:
            velocity_filtered_mean = 0
            velocity_filtered_median = 0
            acceleration_filtered_mean = 0
            acceleration_filtered_median = 0
            acceleration_filtered_stdev = 0
            velocity_filtered_stdev = 0

        overall_euclidean_median, overall_angle_median, single_euclid, single_angle = overall_medians(cell, df,
                                                                                                      cols_angles,
                                                                                                      cols_euclidean,
                                                                                                      parent_id)

        single_euclid_dict[cell] = single_euclid
        single_angle_dict[cell] = single_angle

        instantaneous_displacement = list(df.loc[df[parent_id] == cell, 'Instantaneous Displacement'])
        instantaneous_displacement = [x for x in instantaneous_displacement if np.isnan(x) == False and x != 0]
        time_under = [x for x in instantaneous_displacement if
                      x < parameters['arrest_displacement']]
        arrest_coefficient = (len(time_under) * time_interval) / duration

        sum_[cell] = cell, duration, final_euclid, max_euclid, max_path, straightness, tc_straightness, \
                     displacement_ratio, outreach_ratio, velocity_mean, velocity_median, velocity_filtered_mean, \
                     velocity_filtered_median, velocity_filtered_stdev, acceleration_mean, acceleration_median, \
                     acceleration_filtered_mean, \
                     acceleration_filtered_median, acceleration_filtered_stdev, \
                     arrest_coefficient, overall_angle_median, overall_euclidean_median, convex_hull_volume, \
                     tc_convex, category

    df_sum = pd.DataFrame.from_dict(sum_, orient='index')
    df_msd = pd.DataFrame.from_dict(msd_dict, orient='index')
    all_cols = (cols_euclidean + cols_angles)
    df_msd.columns = ['MSD ' + str(x) for x in range(1, tau_val + 1)]
    cells = list(single_euclid_dict.keys())
    df_single_euclids = pd.DataFrame.from_dict(single_euclid_dict, orient='index')
    df_single_angles = pd.DataFrame.from_dict(single_angle_dict, orient='index')
    df_single = pd.concat([df_single_euclids, df_single_angles], axis=1)
    df_single.columns = all_cols
    df_single.insert(0, 'Cell ID', cells)
    df_msd.insert(0, 'Cell ID', cells)

    df_sum.columns = ['Cell ID', 'Duration', 'Final Euclidean', 'Max Euclidean',
                      'Path Length', 'Straightness', 'Time Corrected Straightness',
                      'Displacement Ratio', 'Outreach Ratio', 'Velocity Mean',
                      'Velocity Median', 'Velocity filtered Mean',
                      'Velocity Filtered Median', 'Velocity Filtered Standard Deviation', 'Acceleration Mean',
                      'Acceleration Median', 'Acceleration Filtered Mean',
                      'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
                      'Arrest Coefficient',
                      'Overall Angle Median', 'Overall Euclidean Median', 'Convex Hull Volume',
                      'Time Corrected Convex Hull Volume', 'Cell Type']

    print('pca check')
    if track_df.shape[0] > 0:
        print('running PCA')
        pca(df_sum, parameters)

    return df_sum, time_interval, df_single, df_msd
