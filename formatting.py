import pandas as pd
import numpy as np
import statistics

"""
Needs to be able to sort by cell id, then time.
account for multi-tracked datapoints
"""


def multi_tracking(df, unique_cell_ids, parent_col, time_col, x_col, y_col, z_col, infile_):
    list_of_df = []
    multi_tracked_spotted = False
    multi_before = 0
    for i, cell in enumerate(unique_cell_ids):
        print(cell)
        x_val = list(df.loc[df[parent_col] == cell, x_col])
        y_val = list(df.loc[df[parent_col] == cell, y_col])
        z_val = list(df.loc[df[parent_col] == cell, z_col])
        tracked_times = list(df.loc[df[parent_col] == cell, time_col])
        cell_list = list(df.loc[df[parent_col] == cell, parent_col])
        multi_tracked_occurences = 1
        for index, time in enumerate(tracked_times):
            if index == 0:
                pass
            else:
                if tracked_times[index] == tracked_times[index - 1]:  # identification of a multi-tracked cell
                    multi_tracked_spotted = True
                    multi_tracked_occurences += 1
                    print(f"INDEX {index}")

                elif tracked_times[index] != tracked_times[index - 1] and multi_tracked_spotted:
                    # multi_track done subject to change how to capture area
                    multi_tracked_spotted = False
                    time_end = tracked_times[index - 1]
                    x_multi_tracked_mean = statistics.mean(
                        x_val[index - (multi_tracked_occurences + multi_before):index - multi_before])
                    y_multi_tracked_mean = statistics.mean(
                        y_val[index - (multi_tracked_occurences + multi_before):index - multi_before])
                    z_multi_tracked_mean = statistics.mean(
                        z_val[index - (multi_tracked_occurences + multi_before):index - multi_before])

                    del (x_val[index - (multi_tracked_occurences + multi_before):index - multi_before],
                         y_val[index - (multi_tracked_occurences + multi_before):index - multi_before],
                         z_val[index - (multi_tracked_occurences + multi_before):index - multi_before],
                         tracked_times[index - (multi_tracked_occurences + multi_before):index - multi_before],
                         cell_list[index - (multi_tracked_occurences + multi_before):index - multi_before])

                    x_val.insert(index - (multi_tracked_occurences + multi_before), x_multi_tracked_mean)
                    y_val.insert(index - (multi_tracked_occurences + multi_before), y_multi_tracked_mean)
                    z_val.insert(index - (multi_tracked_occurences + multi_before), z_multi_tracked_mean)
                    tracked_times.insert(index - (multi_tracked_occurences + multi_before), time_end)
                    cell_list.insert(index - (multi_tracked_occurences + multi_before), cell)

                    multi_before += multi_tracked_occurences - 1  # always inserting back mean
                    multi_tracked_occurences = 1

        df_formatted = pd.DataFrame()
        df_formatted[parent_col] = cell_list
        df_formatted[time_col] = tracked_times
        df_formatted[x_col] = x_val
        df_formatted[y_col] = y_val
        df_formatted[z_col] = z_val
        list_of_df.append(df_formatted)

    df_formatted_return = pd.concat(list_of_df, ignore_index=True)
    infile_ = infile_[:-4]
    df_formatted_return.to_csv(f"{infile_}_multitracked.csv", index=False)

    return df_formatted_return


def adjust_2D(df, infile_):
    df['Z Coordinate'] = 0
    infile_ = infile_[:-4]
    df.to_csv(f"{infile_}_2D.csv", index=False)

    return df


def interpolate_lazy(df, unique_cell_ids, parent_col, time_col, x_col, y_col, z_col, infile_, time_between):
    list_of_df = []
    for i, cell in enumerate(unique_cell_ids):
        print(cell)
        num_insertions = 0
        x_val = list(df.loc[df[parent_col] == cell, x_col])
        y_val = list(df.loc[df[parent_col] == cell, y_col])
        z_val = list(df.loc[df[parent_col] == cell, z_col])
        tracked_times = list(df.loc[df[parent_col] == cell, time_col])
        cell_list = list(df.loc[df[parent_col] == cell, parent_col])
        dict_to_insert = {}
        for index, time in enumerate(tracked_times):
            if index == 0:
                pass
            else:
                if (tracked_times[index] - tracked_times[index - 1]) != time_between and\
                        (tracked_times[index] - tracked_times[index - 1]) != 0:
                    # identification of a missing time point
                    x_interpolated = (x_val[index] - x_val[index - 1]) / 2
                    x_interpolated = x_val[index - 1] + x_interpolated
                    print(x_val[index - 1], x_interpolated, tracked_times[index], tracked_times[index-1])
                    print(tracked_times[index] - tracked_times[index - 1])
                    y_interpolated = (y_val[index] - y_val[index - 1]) / 2
                    y_interpolated = y_val[index - 1] + y_interpolated
                    z_interpolated = (z_val[index] - z_val[index - 1]) / 2
                    z_interpolated = z_val[index - 1] + z_interpolated
                    time_interpolated = tracked_times[index] - time_between
                    dict_to_insert[index] = [index + num_insertions, time_interpolated, x_interpolated, y_interpolated,
                                             z_interpolated]
                    num_insertions += 1

        for key in dict_to_insert.keys():
            values_ = dict_to_insert[key]
            cell_list.insert(values_[0], cell)
            tracked_times.insert(values_[0], values_[1])
            x_val.insert(values_[0], values_[2])
            y_val.insert(values_[0], values_[3])
            z_val.insert(values_[0], values_[4])

        df_formatted = pd.DataFrame()
        df_formatted[parent_col] = cell_list
        df_formatted[time_col] = tracked_times
        df_formatted[x_col] = x_val
        df_formatted[y_col] = y_val
        df_formatted[z_col] = z_val
        list_of_df.append(df_formatted)

    df_formatted_return = pd.concat(list_of_df, ignore_index=True)
    infile_ = infile_[:-4]
    df_formatted_return.to_csv(f"{infile_}_interpolated.csv", index=False)

    return df_formatted_return

