import numpy as np
import dearpygui.dearpygui as dpg
import pandas as pd
import time as tempo
import re
from calculations import calculations
from Summary_Sheet import summary_sheet
from Contacts import contacts, contacts_alive, no_daughter_contacts
from formatting import multi_tracking, adjust_2D, interpolate_lazy

"""

confirm MSD --> competition code has raw data and confirm their MSD sup data 6
 
"""

dpg.create_context()
parameters = {'Interval': 15, 'arrest_displacement': 3.0, 'contact_length': None, 'arrested': 0.95, 'moving': 4,
              'timelapse': 4, 'savefile': 'Migrate3D_Results.xlsx', 'parent_id': 'Parent ID', 'time_col': "Time",
              'x_for': 'X Coordinate', 'y_for': 'Y Coordinate', 'z_for': 'Z Coordinate', 'parent_id2': 'Id',
              'category_col': 'Code', 'Contact': False, 'Tau_val': 6, 'infile_tracks': False, 'multi_track': False,
              'two_dem': False, 'interpolate': False}


def migrate3D(param):
    intervals = parameters['Interval']
    arrest_displacement = parameters['arrest_displacement']
    contact_length = parameters['contact_length']
    arrested = parameters['arrested']
    num_of_tp_moving = parameters['moving']
    timelapse_interval = parameters['timelapse']
    contact_parameter = parameters['Contact']
    track_file = parameters['infile_tracks']

    def main():
        try:
            p_bar_increase = 0.10
            while p_bar_increase < 1:
                dpg.set_value('pbar', p_bar_increase)
                infile_segments = parameters['Infile_segs']
                infile_name = parameters['Infile_segs']
                print(infile_name)
                savefile = parameters['savefile']
                infile_segments = pd.read_csv(infile_segments, sep=',')
                df_infile = pd.DataFrame(infile_segments)
                parent_id = parameters['parent_id']
                time_col = parameters['time_col']
                x_for = parameters['x_for']
                y_for = parameters['y_for']
                z_for = parameters['z_for']

                settings = {'Interval': [intervals], 'Arrest Displacement': [arrest_displacement],
                            'Contact Length': [contact_length], 'Arrested': [arrested], 'Moving': [num_of_tp_moving],
                            'Tau Val': [parameters['Tau_val']]}
                df_settings = pd.DataFrame(data=settings)
                df_infile = df_infile.sort_values(by=[parent_id, time_col], ascending=True)

                cell_ids = list(df_infile.loc[:, parent_id])
                cell_id = []
                for cells in cell_ids:  # get out repetitive cell id
                    if cells in cell_id:
                        pass
                    else:
                        cell_id.append(cells)
                        cell_ids.remove(cells)

                if parameters['multi_track']:
                    df_infile = multi_tracking(df_infile, cell_id, parent_id, time_col, x_for, y_for, z_for,
                                               infile_name)
                if parameters['two_dem']:
                    df_infile = adjust_2D(df_infile, infile_name)

                if parameters['interpolate']:
                    df_infile = interpolate_lazy(df_infile, cell_id, parent_id, time_col, x_for, y_for, z_for,
                                                 infile_name, timelapse_interval)

                list_of_df = []
                tic = tempo.time()
                for cell in cell_id:
                    x_values = list(df_infile.loc[df_infile[parent_id] == cell, x_for])
                    y_values = list(df_infile.loc[df_infile[parent_id] == cell, y_for])
                    z_values = list(df_infile.loc[df_infile[parent_id] == cell, z_for])
                    time = list(df_infile.loc[df_infile[parent_id] == cell, time_col])
                    print('Analyzing cell', cell)
                    calculations(cell, x_values, y_values, z_values, list_of_df, intervals, time, parent_id, parameters)
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
                track_df = pd.DataFrame()
                if parameters['infile_tracks']:
                    track_df = pd.DataFrame(pd.read_csv(track_file))
                    print(track_df.shape)

                df_sum, time_interval, df_single, df_msd = summary_sheet(df_calc, cell_id, cols_angles, cols_euclidean,
                                                                         parent_id, df_infile, x_for, y_for, z_for,
                                                                         parameters['Tau_val'], parameters, track_df)
                p_bar_increase += 0.25
                dpg.set_value('pbar', p_bar_increase)

                tic = tempo.time()
                toc = tempo.time()

                if contact_parameter is False:
                    pass
                else:
                    df_cont = contacts(cell_id, df_infile, parent_id, x_for, y_for, z_for, time_col, contact_length)
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
                    df_settings.to_excel(workbook, sheet_name='Settings', index=False)
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
        except ConnectionError:
            with dpg.window(label='ERROR', width=400, height=200) as err_win:
                dpg.add_input_text(default_value='ERROR, no file selected', width=200)
                dpg.set_value('pbar', 0)

    main()

def formatting_check(sender, app_data):
    if dpg.get_value(sender) is True:
        parameters[sender] = True
    else:
        parameters[sender] = False


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
    parameters['infile_tracks'] = infile


def input_return(sender, app_data):
    parameters[sender] = app_data


def float_return(sender, app_data):
    parameters[sender] = app_data


def Start_migrate(sender, app_data):
    migrate3D(parameters)


with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_segs,
                     file_count=3,
                     tag="segs_dialog_tag"):
    dpg.add_file_extension("", color=(255, 150, 150, 255))
    dpg.add_file_extension(".csv", color=(255, 0, 255, 255))
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".xlsx", color=(255, 255, 0, 255))
    dpg.add_file_extension(".py", color=(0, 255, 0, 255))

with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_cats,
                     file_count=3,
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
                                         default_value=0.0, callback=float_return, tag='contact_length')
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
                                    default_value='Id', callback=input_return, tag='parent_id2')
    category_col = dpg.add_input_text(width=200, label="What is the name of the Column the cell's Category is in?",
                                      default_value='Code', callback=input_return, tag='category_col')

    contact = dpg.add_checkbox(label='Analyze contacts? (note: can significantly increase processing time)',
                               callback=run_contact)

    multi_check = dpg.add_checkbox(label='Adjust for Multi-tracked time points?', callback=formatting_check, tag='multi_track')

    two_dem_check = dpg.add_checkbox(label='Adjust for two-dimensional data?', callback=formatting_check, tag='two_dem')

    interpolate_check = dpg.add_checkbox(label='Lazy Interpolation for missing data points?', callback=formatting_check, tag='interpolate')

    dpg.add_progress_bar(width=600, height=10, label='Progress Bar', tag='pbar')

    dpg.add_button(width=100, label='Run', callback=Start_migrate)

    dpg.create_viewport(title='Migrate3D', width=900, height=700)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()
