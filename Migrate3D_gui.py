import statistics
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import time as tempo
import re
from scipy.spatial.qhull import ConvexHull


"""
- Suppress error when no Categories file is loaded
- dem_2_chck variable (line 53) seems to not be used anywhere
- Check to make sure that angle median and euclid median are properly zeroing out on tracks below the moving limit
- Is there a way to determine "ideal" number of timepoints to calculate?
- Any other parameters we should be calculating? Check recent literature
- Any parameters we should drop or remove from Summary sheet because they are useless?
- Consider pre-filtering dataset for contacts down to alive/moving cells only to speed it up?
- do mean squared displacement and then apply random walk etc based on that value 
"""

dpg.create_context()
parameters = {'Interval': 15, 'arrest_displacement': 3.0, 'contact_length': 10.0, 'arrested': 0.95, 'moving': 4,
              'timelapse': 4, 'savefile': 'Migrate3D_Results.xlsx', 'parent_id': 'Parent ID', 'time_col': "Time",
              'x_for': 'X Coordinate', 'y_for': 'Y Coordinate', 'z_for': 'Z Coordinate', 'parent_id2': 'Parent ID', 'category_col': 'Category',
              'Contact': False, 'Tau_val': 6}


def migrate3D(param):
    intervals = parameters['Interval']
    arrest_displacement = parameters['arrest_displacement']
    contact_length = parameters['contact_length']
    arrested = parameters['arrested']
    num_of_tp_moving = parameters['moving']
    timelapse_interval = parameters['timelapse']
    format_check = 'n'
    contact_parameter = parameters['Contact']


    def main():
        try:
            p_bar_increase = 0.10
            while p_bar_increase < 1:
                dpg.set_value('pbar', p_bar_increase)
                infile = parameters['Infile_segs']
                savefile = parameters['savefile']
                infile = pd.read_csv(infile, sep=',')

                df_infile = pd.DataFrame(infile)

                if format_check == 'y':
                    sort_1 = int(input("What column number would you like to sort by first? "))
                    sort_2 = int(input("What column number would you like to sort by second? "))
                    dem_2_chck = input("Replace Z dimension data with zeroes (i.e. convert 3D to 2D data)? (y/n) ")
                    df_infile = df_infile.sort_values(by=[df_infile.columns[sort_1 - 1], df_infile.columns[sort_2 - 1]], ascending=True)

                parent_id = parameters['parent_id']
                time_col = parameters['time_col']
                x_for = parameters['x_for']
                y_for = parameters['y_for']
                z_for = parameters['z_for']

                cell_ids = list(df_infile.iloc[:, 0])
                cell_id = []
                for cells in cell_ids:  # get out repetitive cell id
                    if cells in cell_id:
                        pass
                    else:
                        cell_id.append(cells)
                        cell_ids.remove(cells)

                list_of_df = []
                tic = tempo.time()
                for cell in cell_id:
                    x_values = list(df_infile.loc[df_infile[parent_id] == cell, x_for])
                    y_values = list(df_infile.loc[df_infile[parent_id] == cell, y_for])
                    z_values = list(df_infile.loc[df_infile[parent_id] == cell, z_for])
                    time = list(df_infile.loc[df_infile[parent_id] == cell, time_col])
                    print('Analyzing cell', cell)
                    calculations(cell, x_values, y_values, z_values, list_of_df, intervals, time, parent_id)
                toc = tempo.time()
                print('Calculations done in {:.4f} seconds'.format(toc - tic))
                df_calc = pd.concat(list_of_df)
                columns = list(df_calc.columns)
                angle_columns = []
                df_filtered = pd.DataFrame()
                filter_ = 3
                for i in range(9 + intervals, len(columns), 2):
                    angle_columns.append(columns[i])
                    df_filtered['Angle Filtered ' + str(filter_) + ' TP'] = df_calc.iloc[:, i]
                    filter_ += 2

                for col in angle_columns:
                    df_calc = df_calc.drop(col, axis=1)
                df_calc = pd.concat([df_calc, df_filtered], axis=1)  # calculations sheet done

                time_points_odd = 0  # start summary sheet
                for i in range(3, intervals + 1):
                    if i % 2 != 0:
                        time_points_odd += 1

                cols_angles = list(df_calc.loc[:, [True if re.search('Angle Filtered+', column) else False for column in
                                                   df_calc.columns]])
                cols_euclidean = list(
                    df_calc.loc[:, [True if re.search('Euclidean+', column) else False for column in df_calc.columns]])

                cell_ids = list(df_calc.iloc[:, 0])
                cell_id = []

                for cell in cell_ids:  # get out repetitive cell id
                    if cell in cell_id:
                        pass
                    else:
                        cell_id.append(cell)
                        cell_ids.remove(cell)
                mapping = {0: None}

                p_bar_increase += 0.25
                dpg.set_value('pbar', p_bar_increase)
                df_sum, time_interval, df_single, df_msd = summary_sheet(df_calc, cell_id, cols_angles, cols_euclidean,
                                                                 parent_id, df_infile, x_for, y_for, z_for,
                                                                 parameters['Tau_val'])
                p_bar_increase += 0.25
                dpg.set_value('pbar', p_bar_increase)

                tic = tempo.time()
                df_cont = contacts(cell_id, df_infile, parent_id, x_for, y_for, z_for, time_col)
                toc = tempo.time()

                if contact_parameter is False:
                    pass
                else:
                    if len(df_cont) == 0:
                        pass
                    else:
                        print('Contacts done in {:.4f} seconds'.format(toc - tic))
                        df_contacts = pd.concat(df_cont, ignore_index=True)
                        df_no_daughter_func = no_daughter_contacts(cell_id, df_contacts, parent_id)
                        df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True)
                        df_alive, df_contact_sum = contacts_alive(df_arrest=df_sum, df_no_mitosis=df_no_daughter,
                                                                  parent_id=parent_id,
                                                                  arrested=arrested,
                                                                  time_interval=time_interval)
                    df_no_dead_ = pd.concat(df_alive, ignore_index=True)
                    df_contact_summary = pd.concat(df_contact_sum, ignore_index=True)
                    df_contact_summary = df_contact_summary.replace(mapping)
                    df_contact_summary = df_contact_summary.dropna()
                df_calc = df_calc.replace(mapping)
                df_sum = df_sum.replace(mapping)
                df_sum.loc[:, 'Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))
                p_bar_increase += 0.30
                dpg.set_value('pbar', p_bar_increase)
                with pd.ExcelWriter(savefile) as workbook:
                    df_calc.to_excel(workbook, sheet_name='Calculations', index=False)
                    df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
                    df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
                    df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
                    if contact_parameter is False:
                        pass
                    else:
                        if len(df_cont) == 0:
                            pass
                        else:
                            print('Saving to .XLSX...')
                            df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                            df_no_daughter.to_excel(workbook, sheet_name='Contacts no Mitosis', index=False)
                            df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                            df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

                p_bar_increase += 0.10
                dpg.set_value('pbar', p_bar_increase)
                print("Done")
                dpg.destroy_context()
        except KeyError:
            with dpg.window(label='ERROR', width=400, height=200) as err_win:
                dpg.add_input_text(default_value='ERROR, no file selected', width=200)
                dpg.set_value('pbar', 0)

    def calculations(cell, x, y, z, list_, num_euclid_spaces, time, parent_id):
        instantaneous_displacement = [0]
        total_displacement = [0]
        path_length = [0]
        instantaneous_velocity = [0]
        instantaneous_acceleration = [0]
        instantaneous_acceleration_filtered = [0]
        instantaneous_velocity_filtered = [0]

        for index, e in enumerate(x):
            if index == 0:
                pass
            else:
                pl = path_length[index - 1]
                sum_x = (x[index] - x[index - 1]) ** 2  # start instantaneous displacement
                sum_y = (y[index] - y[index - 1]) ** 2
                sum_z = (z[index] - z[index - 1]) ** 2
                sum_all = sum_x + sum_y + sum_z
                inst_displacement = np.sqrt(sum_all)
                instantaneous_displacement.append(inst_displacement)  # the sqrt of the sum of final x,y,z dim - initial

                tot_x = (x[index] - x[0]) ** 2  # start total displacement
                tot_y = (y[index] - y[0]) ** 2
                tot_z = (z[index] - z[0]) ** 2
                sum_tot = tot_x + tot_y + tot_z
                total_displacement.append(np.sqrt(sum_tot))
                path_length.append(pl + np.sqrt(sum_all))

                instantaneous_velocity.append(instantaneous_displacement[index] / timelapse_interval)
                instantaneous_acceleration.append(
                    (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / timelapse_interval)
                if inst_displacement > arrest_displacement:
                    instantaneous_velocity_filtered.append(instantaneous_displacement[index] / timelapse_interval)
                    instantaneous_acceleration_filtered.append(
                        (instantaneous_velocity[index] - instantaneous_velocity[index - 1]) / timelapse_interval)
                else:
                    instantaneous_velocity_filtered.append(0)
                    instantaneous_acceleration_filtered.append(0)

        df_to_append = pd.DataFrame({parent_id: cell,
                                     'Time': time,
                                     'Instantaneous Displacement': instantaneous_displacement,
                                     'Total Displacement': total_displacement,
                                     'Path length': path_length,
                                     'Instantaneous Velocity': instantaneous_velocity,
                                     'Instantaneous Acceleration': instantaneous_acceleration,
                                     'Instantaneous Velocity Filtered': instantaneous_velocity_filtered,
                                     'Instantaneous Acceleration Filtered': instantaneous_acceleration_filtered,
                                     })

        for back in range(2, num_euclid_spaces + 1):  # makes euclidean distance calculations
            euclid = []
            for index, element in enumerate(x):
                if back > index:
                    euclid.append(0)
                else:
                    x_val = (x[index] - x[index - back]) ** 2
                    y_val = (y[index] - y[index - back]) ** 2
                    z_val = (z[index] - z[index - back]) ** 2
                    euclid.append(np.sqrt(x_val + y_val + z_val))
            df_to_append['Euclidean ' + str(back) + ' TP'] = euclid

        mod = 3
        arrest_multiplyer = 1
        space = [s for s in range(num_euclid_spaces + 1) if s % 2 != 0]
        for back_angle in range(1, num_euclid_spaces - len(space) + 1):
            angle = []
            angle_filtered = []
            for index, element in enumerate(x):
                try:
                    if back_angle * 2 > index:
                        angle.append(0)
                        angle_filtered.append(0)
                    else:
                        cell = cell
                        x_magnitude0 = x[index] - x[index - back_angle]
                        y_magnitude0 = y[index] - y[index - back_angle]
                        z_magnitude0 = z[index] - z[index - back_angle]
                        x_magnitude1 = x[index - back_angle] - x[index - (back_angle * 2)]
                        y_magnitude1 = y[index - back_angle] - y[index - (back_angle * 2)]
                        z_magnitude1 = z[index - back_angle] - z[index - (back_angle * 2)]
                        np.seterr(invalid='ignore')
                        vec_0 = [x_magnitude0, y_magnitude0, z_magnitude0]
                        vec_1 = [x_magnitude1, y_magnitude1, z_magnitude1]
                        vec_0 = vec_0 / np.linalg.norm(vec_0)
                        vec_1 = vec_1 / np.linalg.norm(vec_1)
                        angle_ = np.arccos(np.clip(np.dot(vec_0, vec_1), -1.0, 1.0))
                        angle_ = angle_ * 180 / np.pi
                        angle.append(angle_)  # unfiltered angle calculations

                        x_val0 = (x[index] - x[index - back_angle]) ** 2  # start filtered angle calculations
                        y_val0 = (y[index] - y[index - back_angle]) ** 2
                        z_val0 = (z[index] - z[index - back_angle]) ** 2

                        x_val1 = (x[index - back_angle] - x[index - (back_angle * 2)]) ** 2
                        y_val1 = (y[index - back_angle] - y[index - (back_angle * 2)]) ** 2
                        z_val1 = (z[index - back_angle] - z[index - (back_angle * 2)]) ** 2
                        euclid_current = np.sqrt(x_val0 + y_val0 + z_val0)
                        euclid_previous = np.sqrt(x_val1 + y_val1 + z_val1)
                        if euclid_current > arrest_displacement * arrest_multiplyer and euclid_previous > arrest_displacement * arrest_multiplyer:
                            angle_filtered.append(np.absolute(angle_))
                        else:
                            angle_filtered.append(0)
                except RuntimeWarning:
                    pass

            arrest_multiplyer += 1
            df_to_append['Angle ' + str(mod) + ' TP'] = angle
            df_to_append['Filtered Angle ' + str(mod) + ' TP'] = angle_filtered
            mod += 2
        list_.append(df_to_append)

    def summary_sheet(df, cell_id, cols_angles, cols_euclidean, parent_id, df_infile, x_for, y_for, z_for, tau_val):
        sum_ = {}
        single_euclid_dict = {}
        single_angle_dict = {}
        msd_dict = {}
        print('Running Summary Sheet...')
        for cell in cell_id:
            x_values = np.array(df_infile.loc[df_infile[parent_id] == cell, x_for])
            y_values = np.array(df_infile.loc[df_infile[parent_id] == cell, y_for])
            z_values = np.array(df_infile.loc[df_infile[parent_id] == cell, z_for])
            vals = []

            for t_diff in range(1, tau_val+1):
                r = np.sqrt(x_values ** 2 + y_values ** 2 + z_values ** 2)
                diff = np.diff(r, n=t_diff)
                diff_sq = diff ** 2
                vals.append(np.mean(diff_sq))
                print(vals)
                if t_diff == tau_val:
                    msd_dict['MSD for Cell ' + str(cell)] = vals
                    vals = []

            x_values = np.array(df_infile.loc[df_infile[parent_id] == cell, x_for])
            y_values = np.array(df_infile.loc[df_infile[parent_id] == cell, y_for])
            z_values = np.array(df_infile.loc[df_infile[parent_id] == cell, z_for])

            convex_coords = np.array([x_values, y_values, z_values]).transpose()
            print(convex_coords)
            if convex_coords.shape[0] < 4:
                convex_hull_volume = 0
            else:
                convex_hull = ConvexHull(convex_coords)
                convex_hull_volume = convex_hull.volume
                print(convex_hull_volume)

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
            tc_convex = convex_hull_volume * np.sqrt(duration)
            outreach_ratio = max_euclid / max_path
            velocity = list(df.loc[df[parent_id] == cell, 'Instantaneous Velocity'])
            velocity_filtered = list(df.loc[df[parent_id] == cell, 'Instantaneous Velocity Filtered'])
            velocity_filtered = [x for x in velocity_filtered if np.isnan(x) == False and x != 0]
            velocity.pop(0)
            velocity_mean = statistics.mean(velocity)
            velocity_median = statistics.median(velocity)
            acceleration = list(df.loc[df[parent_id] == cell, 'Instantaneous Acceleration'])
            if len(acceleration) >= num_of_tp_moving:
                acceleration = [x for x in acceleration if np.isnan(x) == False and x != 0]
                acceleration_mean = statistics.mean(acceleration)
                acceleration_median = statistics.median(acceleration)
            elif len(acceleration) < num_of_tp_moving:
                acceleration_mean = 0
                acceleration_median = 0

            acceleration_filtered = list(df.loc[df[parent_id] == cell, 'Instantaneous Acceleration Filtered'])
            acceleration_filtered = [x for x in acceleration_filtered if np.isnan(x) == False and x != 0]
            if len(acceleration_filtered) >= num_of_tp_moving:
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
                          x < arrest_displacement]
            arrest_coefficient = (len(time_under) * time_interval) / duration

            sum_[cell] = cell, duration, final_euclid, max_euclid, max_path, straightness, tc_straightness, \
                         displacement_ratio, outreach_ratio, velocity_mean, velocity_median, velocity_filtered_mean, \
                         velocity_filtered_median, velocity_filtered_stdev, acceleration_mean, acceleration_median, \
                         acceleration_filtered_mean, \
                         acceleration_filtered_median, acceleration_filtered_stdev, \
                         arrest_coefficient, overall_angle_median, overall_euclidean_median, convex_hull_volume, \
                         tc_convex

        df_sum = pd.DataFrame.from_dict(sum_, orient='index')
        df_msd = pd.DataFrame.from_dict(msd_dict, orient='index')
        all_cols = (cols_euclidean + cols_angles)
        df_msd.columns = ['MSD ' + str(x) for x in range(1, tau_val+1)]
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
                          'Time Corrected Convex Hull Volume']

        print(msd_dict)
        return df_sum, time_interval, df_single, df_msd

    def overall_medians(cell, df, cols_angles, cols_euclidean, parent_id):
        list_of_angle_medians = []
        list_of_euclidean_medians = []
        single_euclidean = []
        single_angle = []

        for col_ in cols_angles:
            angle_median = list(df.loc[df[parent_id] == cell, col_])
            angle_median = [x for x in angle_median if x is not None and x != 0]
            if len(angle_median) > 2:
                angle_median = statistics.median(angle_median)
                if angle_median == 0:
                    pass
                else:
                    list_of_angle_medians.append(angle_median)
                    single_angle.append(angle_median)
            else:
                pass

        for cols_ in cols_euclidean:
            euclidean_median = df.loc[df[parent_id] == cell, cols_]
            euclidean_median = [x for x in euclidean_median if x is not None and x != 0]
            if len(euclidean_median) > 2:
                euclidean_median = statistics.median(euclidean_median)
                list_of_euclidean_medians.append(euclidean_median)
                single_euclidean.append(euclidean_median)
            else:
                pass
        if len(list_of_euclidean_medians) >= 1:
            overall_euclidean_median = statistics.median(list_of_euclidean_medians)
        else:
            overall_euclidean_median = None
        if len(list_of_angle_medians) >= 1:
            overall_angle_median = statistics.median(list_of_angle_medians)
        else:
            overall_angle_median = None

        return overall_euclidean_median, overall_angle_median, single_euclidean, single_angle

    def contacts(cell_id, df, parent_id, x, y, z, time_):
        if contact_parameter is False:
            pass
        else:
            last_done = cell_id[0]
            df_of_contacts = []
            base_done = []
            print('Running Contacts')
            for cell_base in cell_id:
                try:
                    x_base = list(df.loc[df[parent_id] == cell_base, x])
                    y_base = list(df.loc[df[parent_id] == cell_base, y])
                    z_base = list(df.loc[df[parent_id] == cell_base, z])
                    time_base = list(df.loc[df[parent_id] == cell_base, time_])
                    base_done.append(cell_base)
                    for cell_comp in cell_id:
                        if cell_comp == cell_base or cell_comp in base_done:
                            pass
                        else:
                            x_comp = list(df.loc[df[parent_id] == cell_comp, x])
                            y_comp = list(df.loc[df[parent_id] == cell_comp, y])
                            z_comp = list(df.loc[df[parent_id] == cell_comp, z])
                            time_comp = list(df.loc[df[parent_id] == cell_comp, time_])
                            comp_index_advance = 0
                            base_index_advance = 0
                            while time_base != time_comp:
                                try:
                                    time_base_check = time_base[base_index_advance]
                                    time_comp_check = time_comp[comp_index_advance]
                                    if time_base_check > time_comp_check:
                                        comp_index_advance += 1
                                    elif time_comp_check > time_base_check:
                                        base_index_advance += 1
                                    if time_base_check == time_comp_check:
                                        break
                                    if len(time_base) == 0 or len(time_comp) == 0:
                                        break
                                except TypeError:
                                    break
                            shortest_len = 0
                            len_time_base = len(time_base)
                            len_time_comp = len(time_comp)
                            if len_time_base > len_time_comp:
                                shortest_len = time_comp
                            elif len_time_comp > len_time_base:
                                shortest_len = time_base
                            elif len_time_comp == len_time_base:
                                shortest_len = time_base
                            cell_base_list = []
                            cell_comp_list = []
                            time_of_contact = []
                            for i, e in enumerate(shortest_len):
                                try:
                                    time_b = time_base[i + base_index_advance]
                                    time_c = time_comp[i + comp_index_advance]
                                    x_diff = np.abs(x_base[i + base_index_advance] - x_comp[i + comp_index_advance])
                                    if x_diff <= contact_length:
                                        y_diff = np.abs(y_base[i + base_index_advance] - y_comp[i + comp_index_advance])
                                        if y_diff <= contact_length:
                                            z_diff = np.abs(
                                                z_base[i + base_index_advance] - z_comp[i + comp_index_advance])
                                            if z_diff <= contact_length:
                                                cell_base_list.append(cell_base)
                                                cell_comp_list.append(cell_comp)
                                                time_of_contact.append(time_b)
                                            else:
                                                pass
                                        else:
                                            pass
                                    else:
                                        pass
                                except IndexError:
                                    pass

                            df_c = pd.DataFrame({parent_id: cell_base_list,
                                                 'Cell Compare': cell_comp_list,
                                                 'Time of Contact': time_of_contact})
                            if df_c.empty:
                                pass
                            else:
                                df_of_contacts.append(df_c)
                except IndexError:
                    pass

            for cell_base in reversed(cell_id):
                try:
                    x_base = list(df.loc[df[parent_id] == cell_base, x])
                    y_base = list(df.loc[df[parent_id] == cell_base, y])
                    z_base = list(df.loc[df[parent_id] == cell_base, z])
                    time_base = list(df.loc[df[parent_id] == cell_base, time_])
                    base_done.append(cell_base)
                    for cell_comp in cell_id:
                        if cell_comp == cell_base or cell_comp in base_done:
                            pass
                        else:
                            x_comp = list(df.loc[df[parent_id] == cell_comp, x])
                            y_comp = list(df.loc[df[parent_id] == cell_comp, y])
                            z_comp = list(df.loc[df[parent_id] == cell_comp, z])
                            time_comp = list(df.loc[df[parent_id] == cell_comp, time_])
                            comp_index_advance = 0
                            base_index_advance = 0
                            while time_base != time_comp:
                                try:
                                    time_base_check = time_base[base_index_advance]
                                    time_comp_check = time_comp[comp_index_advance]
                                    if time_base_check > time_comp_check:
                                        comp_index_advance += 1
                                    elif time_comp_check > time_base_check:
                                        base_index_advance += 1
                                    if time_base_check == time_comp_check:
                                        break
                                    if len(time_base) == 0 or len(time_comp) == 0:
                                        break
                                except TypeError:
                                    break
                            shortest_len = 0
                            len_time_base = len(time_base)
                            len_time_comp = len(time_comp)
                            if len_time_base > len_time_comp:
                                shortest_len = time_comp
                            elif len_time_comp > len_time_base:
                                shortest_len = time_base
                            elif len_time_comp == len_time_base:
                                shortest_len = time_base
                            cell_base_list = []
                            cell_comp_list = []
                            time_of_contact = []
                            for i, e in enumerate(shortest_len):
                                try:
                                    time_b = time_base[i + base_index_advance]
                                    time_c = time_comp[i + comp_index_advance]
                                    x_diff = np.abs(x_base[i + base_index_advance] - x_comp[i + comp_index_advance])
                                    if x_diff <= contact_length:
                                        y_diff = np.abs(y_base[i + base_index_advance] - y_comp[i + comp_index_advance])
                                        if y_diff <= contact_length:
                                            z_diff = np.abs(
                                                z_base[i + base_index_advance] - z_comp[i + comp_index_advance])
                                            if z_diff <= contact_length:
                                                cell_base_list.append(cell_base)
                                                cell_comp_list.append(cell_comp)
                                                time_of_contact.append(time_b)
                                            else:
                                                pass
                                        else:
                                            pass
                                    else:
                                        pass
                                except IndexError:
                                    pass

                            df_c = pd.DataFrame({parent_id: cell_base_list,
                                                 'Cell Compare': cell_comp_list,
                                                 'Time of Contact': time_of_contact})
                            if df_c.empty:
                                pass
                            else:
                                df_of_contacts.append(df_c)
                except IndexError:
                    pass

            return df_of_contacts

    def no_daughter_contacts(cell_id, df, parent_id):
        list_of_df = []
        for cell_base in cell_id:
            cell_comp_list = list(df.loc[df[parent_id] == cell_base, 'Cell Compare'])
            time_ = list(df.loc[df[parent_id] == cell_base, 'Time of Contact'])
            updated_cell_comp = []
            for index_, cell_comp in enumerate(cell_comp_list):
                if np.abs(cell_comp - cell_base) == 1 and cell_base != cell_comp:
                    updated_cell_comp.append(0)
                else:
                    updated_cell_comp.append(cell_comp)
            df_no_daughters = pd.DataFrame({parent_id: cell_base,
                                            'Cell Compare': updated_cell_comp,
                                            'Time of Contact': time_})
            if df_no_daughters.empty:
                pass
            else:
                df_no_daughters = df_no_daughters.replace(0, None)
                df_no_daughters = df_no_daughters.dropna()
                list_of_df.append(df_no_daughters)

        return list_of_df

    def contacts_alive(df_arrest, df_no_mitosis, parent_id, arrested, time_interval):
        cells_in_arrest = list(df_arrest.loc[:, 'Cell ID'])
        all_alive = []
        list_of_df_no_dead = []
        list_of_summary_df = []
        for cells in cells_in_arrest:
            arrest_coeffs = float(df_arrest.loc[df_arrest['Cell ID'] == cells, 'Arrest Coefficient'])
            if arrest_coeffs < arrested:
                all_alive.append(cells)
            else:
                pass

        for ind, cell_a in enumerate(all_alive):
            cell_comp = list(df_no_mitosis.loc[df_no_mitosis[parent_id] == cell_a, 'Cell Compare'])
            time__ = list(df_no_mitosis.loc[df_no_mitosis[parent_id] == cell_a, 'Time of Contact'])
            only_1_comp = []
            for cell in cell_comp:
                if cell in only_1_comp:
                    pass
                else:
                    only_1_comp.append(cell)

            list_of_df_no_dead.append(pd.DataFrame({parent_id: cell_a,
                                                    'Cell Compare': cell_comp,
                                                    'Time of Contact': time__}))
            if len(time__) > 2:
                time_actual = [x * time_interval for x in range(len(time__))]
                med_time = statistics.median(time_actual)

            else:
                med_time = None

            sum_df = pd.DataFrame({parent_id: [cell_a],
                                   'Number of Contacts': [len(only_1_comp)],
                                   'Total Time Spent in Contact': [len(time__) * time_interval],
                                   'Median Contact Duration': [med_time]})
            if sum_df.empty:
                pass
            else:
                list_of_summary_df.append(sum_df)

        return list_of_df_no_dead, list_of_summary_df

    main()


def run_contact(sender, app_data):
    if dpg.get_value(sender) is True:
        parameters['Contact'] = True
    else:
        parameters['Contact'] = False


def callback_file_segs(sender, app_data):
    infile = str(app_data['file_path_name'])
    parameters['Infile_segs'] = infile

def callback_file_cats(sender, app_data):
    infile = str(app_data['file_path_name'])
    parameters['Infile_cats'] = infile

def input_return(sender, app_data):
    parameters[sender] = app_data


def float_return(sender, app_data):
    parameters[sender] = app_data


def Start_migrate(sender, app_data):
    migrate3D(parameters)


with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_segs, file_count=3,
                     tag="segs_dialog_tag"):
    dpg.add_file_extension("", color=(255, 150, 150, 255))
    dpg.add_file_extension(".csv", color=(255, 0, 255, 255))
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".xlsx", color=(255, 255, 0, 255))
    dpg.add_file_extension(".py", color=(0, 255, 0, 255))

with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_cats, file_count=3,
                     tag="cats_dialog_tag"):
    dpg.add_file_extension("", color=(255, 150, 150, 255))
    dpg.add_file_extension(".csv", color=(255, 0, 255, 255))
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".xlsx", color=(255, 255, 0, 255))
    dpg.add_file_extension(".py", color=(0, 255, 0, 255))

with dpg.window(label="Migrate3D", width=800, height=600) as Window:
    dpg.add_button(width=140, label="Open Segments File", callback=lambda: dpg.show_item("segs_dialog_tag"))
    dpg.add_button(width=155, label="Open Categories File", callback=lambda: dpg.show_item("cats_dialog_tag"))
    interval = dpg.add_input_int(width=100, label='Intervals to calculate', default_value=15, callback=input_return,
                                 tag='Interval')
    arrest_displacement = dpg.add_input_float(width=100,
                                              label="Arrest limit (assumes same units as XYZ coordinates)",
                                              default_value=3, callback=float_return, tag='arrest_displacement')
    contact_length = dpg.add_input_float(width=100,
                                         label='Maximum distance between cells that would be considered a contact',
                                         default_value=10, callback=float_return, tag='contact_length')
    arrested = dpg.add_input_float(width=100,
                                   label='Maximum arrest coefficient value for a cell to be considered alive',
                                   default_value=0.95, callback=float_return, tag='arrested')
    moving = dpg.add_input_int(width=100,
                               label='Minimum number of timepoints for a track to be included in analysis',
                               default_value=4, callback=input_return, tag='moving')
    tau_val = dpg.add_input_int(width=100, label='Number of time lags to use for Mean Square Displacement',
                                default_value=6, callback=input_return, tag='Tau_val')
    timelapse = dpg.add_input_float(width=100,
                                    label='Timelapse interval in minutes',
                                    default_value=4, callback=input_return, tag='timelapse')
    save_file = dpg.add_input_text(width=300,
                                   label='Output filename (will be in .xlsx format)',
                                   default_value='Migrate3D_Results.xlsx', callback=input_return, tag='savefile')

    parent_id = dpg.add_input_text(width=200, label='What is the name of the Column your cell ID data is in? ('
                                                    'Segments file)',
                                   default_value='Parent ID', callback=input_return, tag='parent_id')
    time_col = dpg.add_input_text(width=200, label='What is the name of the Column your Time data is in?',
                                  default_value='Time', callback=input_return, tag='time_col')
    x_for = dpg.add_input_text(width=200, label='What is the name of the Column your X coordinate data is in?',
                               default_value='X Coordinate', callback=input_return, tag='x_for')
    y_for = dpg.add_input_text(width=200, label='What is the name of the Column your Y coordinate data is in?',
                               default_value='Y Coordinate', callback=input_return, tag='y_for')
    z_for = dpg.add_input_text(width=200, label='What is the name of the Column your Z coordinate data is in?',
                               default_value='Z Coordinate', callback=input_return, tag='z_for')
    parent_id2 = dpg.add_input_text(width=200, label='What is the name of the Column your cell ID data is in? ('
                                                     'Categories file)',
                                    default_value='Parent ID', callback=input_return, tag='parent_id2')
    category_col = dpg.add_input_text(width=200, label="What is the name of the Column the cell's Category is in?",
                                      default_value='Category', callback=input_return, tag='category_col')

    Contact = dpg.add_checkbox(label='Analyze contacts? (note: can significantly increase processing time)',
                               callback=run_contact)

    dpg.add_progress_bar(width=600, height=10, label='Progress Bar', tag='pbar')

    dpg.add_button(width=100, label='Run', callback=Start_migrate)

    dpg.create_viewport(title='Migrate3D', width=900, height=700)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()