import pandas as pd
import numpy as np
import statistics

"""
Needs to be able to sort by cell id, then time.
account for multi-tracked datapoints
"""
def multi_tracking(df, unique_cell_ids, parent_col, time_col, x_col, y_col, z_col, infile_):
    multi_tracked_spotted = False
    list_of_df = []
    df_1 = df.copy()
    for i, cell in enumerate(unique_cell_ids):
        tracked_times = list(df.loc[df[parent_col] == cell, time_col])
        for index, time in enumerate(tracked_times):
            if index == 0:
                pass
            else:
                if tracked_times[index] == tracked_times[index - 1]:  # identification of a multitracked cell
                    multi_tracked_spotted = True

                elif tracked_times[index] != tracked_times[index - 1] and multi_tracked_spotted:
                    # multi_track done subject to change how to capture area
                    df_corrected = pd.DataFrame()
                    multi_tracked_spotted = False
                    time_end = tracked_times[index - 1]
                    df_by_cell_and_time = df.loc[(df[parent_col] == cell) & (df[time_col] == time_end), [parent_col, time_col, x_col, y_col, z_col]]
                    df_1 = df_1.drop(df_by_cell_and_time.index)
                    x_adjusted = statistics.mean(list(df_by_cell_and_time.loc[:, x_col]))
                    y_adjusted = statistics.mean(list(df_by_cell_and_time.loc[:, y_col]))
                    z_adjusted = statistics.mean(list(df_by_cell_and_time.loc[:, z_col]))
                    df_corrected[parent_col] = df_by_cell_and_time[parent_col]
                    df_corrected[time_col] = df_by_cell_and_time[time_col]
                    df_corrected[x_col] = x_adjusted
                    df_corrected[y_col] = y_adjusted
                    df_corrected[z_col] = z_adjusted
                    df_corrected = df_corrected.iloc[:1, :]
                    list_of_df.append(df_corrected)

    list_of_df.append(df_1)
    df_formatted_return = pd.concat(list_of_df)
    df_formatted_return = df_formatted_return.sort_values(by=[parent_col, time_col], ascending=True)

    df_formatted_return.to_csv(f"{os.path.basename(infile_[:-4])}_multitracked.csv", index=False)
    return df_formatted_return



def adjust_2D(df, infile_):
    df['Z Coordinate'] = 0
    infile_ = infile_[:-4]
    df.to_csv(f"{os.path.basename(infile_[:-4])}_2D.csv", index=False)

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
    df_formatted_return.to_csv(f"{os.path.basename(infile_[:-4])}_interpolated.csv", index=False)

    return df_formatted_return

