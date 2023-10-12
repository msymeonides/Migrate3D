import numpy as np
import dearpygui.dearpygui as dpg
import pandas as pd
import time as tempo
import re
from warnings import simplefilter

from calculations import calculations
from Summary_Sheet import summary_sheet
from Contacts import contacts, contacts_alive, no_daughter_contacts
from formatting import multi_tracking, adjust_2D, interpolate_lazy
#Mel's test commit updated !!
# Welcome to Migrate3D version 1.1 (posted January 9, 2023)
# Please see README.md before running this package
# Migrate3D was developed by Matthew Kinahan and Menelaos Symeonides at the University of Vermont, funded by NIH R21-AI152816 and NIH R56-AI172486 (PI: Markus Thali)
# For more information, see https://github.com/msymeonides/Migrate3D/


dpg.create_context()

parameters = {'timelapse': 4, 'arrest_limit': 3.0, 'moving': 4, 'contact_length': 12, 'arrested': 0.95,
              'tau_msd': 50, 'tau_euclid': 25, 'savefile': 'Migrate3D_Results', 'parent_id': 'Parent ID',
              'time_col': "Time", 'x_for': 'X Coordinate', 'y_for': 'Y Coordinate', 'z_for': 'Z Coordinate',
              'parent_id2': 'Id', 'category_col': 'Code', 'interpolate': False, 'multi_track': False,
              'two_dim': False, 'Contact': False, 'pca_filter': None, 'infile_tracks': False}


def migrate3D(param):
    timelapse_interval = parameters['timelapse']
    arrest_limit = parameters['arrest_limit']
    num_of_tp_moving = parameters['moving']
    contact_length = parameters['contact_length']
    arrested = parameters['arrested']
    tau_euclid = parameters['tau_euclid']
    contact_parameter = parameters['Contact']
    track_file = parameters['infile_tracks']

    def main():
        bigtic = tempo.time()
        try:
            p_bar_increase = 0.10
            while p_bar_increase < 1:
                dpg.set_value('pbar', p_bar_increase)
                infile_segments = parameters['Infile_segs']
                infile_name = parameters['Infile_segs']
                savefile = parameters['savefile']
                infile_segments = pd.read_csv(infile_segments, sep=',')
                df_infile = pd.DataFrame(infile_segments)
                parent_id = parameters['parent_id']
                time_col = parameters['time_col']
                x_for = parameters['x_for']
                y_for = parameters['y_for']
                z_for = parameters['z_for']

                settings = {'Timelapse Interval': [timelapse_interval], 'Arrest Limit': [arrest_limit],
                            'Min. TP Moving': [num_of_tp_moving], 'Max. Contact Length': [contact_length],
                            'Arrested': [arrested], 'Tau (MSD)': [parameters['tau_msd']], 'Tau (Euclid/Angle)': [tau_euclid]}
                df_settings = pd.DataFrame(data=settings)
                df_infile = df_infile.sort_values(by=[parent_id, time_col], ascending=True)
                tic = tempo.time()
                print('Formatting input dataset ' + infile_name + '...')
                cell_ids = list(df_infile.loc[:, parent_id])
                cell_id = []
                for cells in cell_ids:
                    if cells in cell_id:
                        pass
                    else:
                        cell_id.append(cells)
                        cell_ids.remove(cells)

                if parameters['multi_track']:
                    df_infile = multi_tracking(df_infile, cell_id, parent_id, time_col, x_for, y_for, z_for,
                                               infile_name)
                if parameters['two_dim']:
                    df_infile = adjust_2D(df_infile, infile_name)

                if parameters['interpolate']:
                    df_infile = interpolate_lazy(df_infile, cell_id, parent_id, time_col, x_for, y_for, z_for,
                                                 infile_name, timelapse_interval)
                toc = tempo.time()
                print('...Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

                list_of_df = []
                tic = tempo.time()
                print('Calculating migration parameters...')
                for cell in cell_id:
                    x_values = list(df_infile.loc[df_infile[parent_id] == cell, x_for])
                    y_values = list(df_infile.loc[df_infile[parent_id] == cell, y_for])
                    z_values = list(df_infile.loc[df_infile[parent_id] == cell, z_for])
                    time = list(df_infile.loc[df_infile[parent_id] == cell, time_col])
                    calculations(cell, x_values, y_values, z_values, list_of_df, tau_euclid, time, parent_id, parameters)

                df_calc = pd.concat(list_of_df)
                columns = list(df_calc.columns)
                angle_columns = []
                df_filtered = pd.DataFrame()
                filter_ = 3
                for i in range(9 + tau_euclid, len(columns), 2):
                    angle_columns.append(columns[i])
                    df_filtered['Angle Filtered ' + str(filter_) + ' TP'] = df_calc.iloc[:, i]
                    filter_ += 2

                for col in angle_columns:
                    df_calc = df_calc.drop(col, axis=1)
                df_calc = pd.concat([df_calc, df_filtered], axis=1)

                p_bar_increase += 0.1
                dpg.set_value('pbar', p_bar_increase)

                time_points_odd = 0
                for i in range(3, tau_euclid + 1):
                    if i % 2 != 0:
                        time_points_odd += 1

                cols_angles = list(df_calc.loc[:, [True if re.search('Angle Filtered+', column) else False for column in
                                                   df_calc.columns]])
                cols_euclidean = list(
                    df_calc.loc[:, [True if re.search('Euclidean+', column) else False for column in df_calc.columns]])

                cell_ids = list(df_calc.iloc[:, 0])
                cell_id = []

                for cell in cell_ids:
                    if cell in cell_id:
                        pass
                    else:
                        cell_id.append(cell)
                        cell_ids.remove(cell)
                mapping = {0: None}
                toc = tempo.time()
                print('...Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)
                track_df = pd.DataFrame()
                if parameters['infile_tracks']:
                    track_df = pd.DataFrame(pd.read_csv(track_file))

                df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_sum_cat = summary_sheet(df_calc, cell_id, cols_angles, cols_euclidean,
                                                                         parent_id, df_infile, x_for, y_for, z_for,
                                                                         parameters['tau_msd'], parameters, track_df, savefile)
                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)



                if contact_parameter is False:
                    pass
                else:
                    tic = tempo.time()
                    print('Detecting contacts...')
                    df_cont = contacts(cell_id, df_infile, parent_id, x_for, y_for, z_for, time_col, contact_length)
                    if len(df_cont) == 0:
                        pass
                    else:
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
                    toc = tempo.time()
                    print('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
                df_calc = df_calc.replace(mapping)
                df_sum = df_sum.replace(mapping)
                if track_df.shape[0] > 0:
                    df_sum['Cell Type'] = df_sum['Cell Type'].replace(np.nan, 0)
                df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))
                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)
                savepath = savefile + '.xlsx'
                print('Saving main output to ' + savepath + '...')
                savecontacts = savefile + '_Contacts.xlsx'
                try:
                    with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
                        df_settings.to_excel(workbook, sheet_name='Settings', index=False)
                        df_calc.to_excel(workbook, sheet_name='Calculations', index=False)
                        df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
                        df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
                        df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
                        df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summaries All', index=True)
                        df_msd_sum_cat.to_excel(workbook, sheet_name='MSD Per Category', index=True)
                except:
                    print('ExcelWriter has thrown an exception due to the output file being too large. Outputs will be in .CSV format.')
                    df_settings.to_csv(f'{savefile}_Settings.csv', index=False)
                    df_calc.to_csv(f'{savefile}_Calculations.csv', index=False)
                    df_sum.to_csv(f'{savefile}_Summary_Statistics.csv', index=False)
                    df_single.to_csv(f'{savefile}_Single_Timepoint_Medians.csv', index=False)
                    df_msd.to_csv(f'{savefile}_Mean_Squared_Displacements.csv', index=False)
                    df_msd_sum_all.to_csv(f'{savefile}_MSD_Summaries_All', index=True)
                    df_msd_sum_cat.to_csv(f'{savefile}_MSD_Per_Category', index=True)
                finally:
                    if contact_parameter is False:
                        pass
                    else:
                        if len(df_cont) == 0:
                            pass
                        else:
                            print('Saving Contacts output to ' + savecontacts + '...')
                            with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
                                df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                                df_no_daughter.to_excel(workbook, sheet_name='Contacts no Mitosis', index=False)
                                df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                                df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)
                print("Migrate3D done!")
                bigtoc = tempo.time()
                total_time_sec = (int(round((bigtoc - bigtic), 1)))
                total_time_min = round((total_time_sec / 60), 1)
                if total_time_sec < 180:
                    print('Total time taken = {:.0f} seconds.'.format(total_time_sec))
                else:
                    print('Total time taken = {:.1f} minutes.'.format(total_time_min))
                dpg.destroy_context()
        except ValueError:
            with dpg.window(label='ERROR', width=400, height=200) as err_win:
                dpg.add_input_text(default_value='Input error, please ensure all inputs are correct', width=200)
                dpg.set_value('pbar', 0)

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
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

with dpg.window(label="Migrate3D", width=900, height=660) as Window:
    dpg.add_button(width=140, label="Open Segments File", callback=lambda: dpg.show_item("segs_dialog_tag"))
    dpg.add_button(width=155, label="Open Categories File", callback=lambda: dpg.show_item("cats_dialog_tag"))

    timelapse = dpg.add_input_float(width=100, label='Timelapse interval (in same units as Time data in input dataset)',
                                    default_value=4, callback=input_return, tag='timelapse')
    arrest_limit = dpg.add_input_float(width=100, label="Arrest limit (in same units as XYZ coordinates in input dataset)",
                                              default_value=3, callback=float_return, tag='arrest_limit')
    moving = dpg.add_input_int(width=100, label='Minimum timepoints moving (number of recorded timepoints required for a track to be included in analysis)',
                               default_value=4, callback=input_return, tag='moving')
    contact_length = dpg.add_input_float(width=100, label='Maximum contact length (maximum distance between cells that would be considered a contact)',
                                         default_value=12, callback=float_return, tag='contact_length')
    arrested = dpg.add_input_float(width=100, label='Arrested/Dead: Cells with an Arrest Coefficient above this value will be considered to be "dead" during Contact detection',
                                   default_value=0.95, callback=float_return, tag='arrested')
    tau_msd = dpg.add_input_int(width=100, label='Maximum tau value (time lags) for mean square displacement (MSD) calculations',
                                default_value=50, callback=input_return, tag='tau_msd')
    tau_euclid = dpg.add_input_int(width=100, label='Maximum tau value (time lags) for Euclidean Distance and Turning Angle calculations',
                                 default_value=25, callback=input_return, tag='tau_euclid')

    save_file = dpg.add_input_text(width=250, label='Output filename (.xlsx extension will be added, do not include!)',
                                   default_value='Migrate3D_Results', callback=input_return, tag='savefile')
    parent_id = dpg.add_input_text(width=150, label='Column header name in input Segments file for cell identifiers',
                                   default_value='Parent ID', callback=input_return, tag='parent_id')
    time_col = dpg.add_input_text(width=150, label='Column header name in input Segments file for Time data',
                                  default_value='Time', callback=input_return, tag='time_col')
    x_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for X coordinates',
                               default_value='X Coordinate', callback=input_return, tag='x_for')
    y_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for Y coordinates',
                               default_value='Y Coordinate', callback=input_return, tag='y_for')
    z_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for Z coordinates',
                               default_value='Z Coordinate', callback=input_return, tag='z_for')
    parent_id2 = dpg.add_input_text(width=150, label='Column header name in input Categories file for cell identifiers',
                                    default_value='Id', callback=input_return, tag='parent_id2')
    category_col = dpg.add_input_text(width=150, label="Column header name in input Categories file for cell category",
                                      default_value='Code', callback=input_return, tag='category_col')

    interpolate_check = dpg.add_checkbox(label='Perform lazy interpolation for missing data points?',
                                         callback=formatting_check, tag='interpolate')
    multi_check = dpg.add_checkbox(label='Average out any multi-tracked time points?',
                                   callback=formatting_check, tag='multi_track')
    two_dim_check = dpg.add_checkbox(label='Adjust for 2D data? Check if input dataset is 2D, or to convert 3D data to 2D by '
                                           'ignoring Z coordinates.', callback=formatting_check, tag='two_dim')
    contact = dpg.add_checkbox(label='Detect cell-cell contacts? (Warning: can significantly increase processing time!)',
                               callback=run_contact)
    pca_filter = dpg.add_input_text(width=250, label='To limit PCA analysis to certain cell types, enter them here '
                                    'separated by a comma.', callback=input_return, tag='pca_filter')

    dpg.add_progress_bar(width=600, height=10, label='Progress Bar', tag='pbar')
    dpg.add_button(width=100, label='Run', callback=Start_migrate)
    dpg.create_viewport(title='Migrate3D', width=900, height=700)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()