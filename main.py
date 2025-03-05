import numpy as np
import dearpygui.dearpygui as dpg
import pandas as pd
import time as tempo
import os
import parallel_contacts
import asyncio
from warnings import simplefilter
from datetime import date
#from xgboost import plot_importance
from matplotlib import pyplot

from calculations import calculations
from summary_sheet import summary_sheet
from contacts import contacts, contacts_moving, no_daughter_contacts
from formatting import multi_tracking, adjust_2D, interpolate_lazy
from attract import attract


# Welcome to Migrate3D version 2.X DEVELOPMENT
# Please see README.md before running this package
# Migrate3D was developed by Matthew Kinahan, Emily Mynar, and Menelaos Symeonides at the University of Vermont,
# funded by NIH R21-AI152816 and NIH R56-AI172486 (PI: Markus Thali)
# For more information, see https://github.com/msymeonides/Migrate3D/


dpg.create_context()

# Default parameters
parameters = {'timelapse': 4, 'arrest_limit': 3.0, 'moving': 4, 'contact_length': 12, 'arrested': 0.95,
              'tau_msd': 50, 'tau_euclid': 25, 'savefile': '{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
              'object_id_col_name': 'Parent ID', 'time_col_name': "Time", 'x_col_name': 'X Coordinate', 'y_col_name': 'Y Coordinate',
              'z_col_name': 'Z Coordinate', 'object_id_2_col': 'ID', 'category_col': 'Category', 'interpolate': False,
              'multi_track': False, 'two_dim': False, 'contact': False, 'pca_filter': None, 'infile_tracks': False}


def migrate3D(param):
    """
        Main function to run the Migrate3D analysis. It sets parameters according to user input and calls the main function.
        Args:
            param (dict): Dictionary containing user-defined parameters for the analysis.
    """

    # Set parameters according to user input
    timelapse_interval = round((parameters['timelapse']), 3)
    arrest_limit = parameters['arrest_limit']
    num_of_tp_moving = parameters['moving']
    contact_length = parameters['contact_length']
    arrested = parameters['arrested']
    tau_euclid = parameters['tau_euclid']
    contact_parameter = parameters['contact']
    track_file = parameters['infile_tracks']

    if parameters['multi_track']:
        multi_track = 'On'
    else:
        multi_track = 'Off'
    if parameters['two_dim']:
        two_dim = 'On'
    else:
        two_dim = 'Off'
    if parameters['interpolate']:
        interpolate = 'On'
    else:
        interpolate = 'Off'
    if parameters['infile_tracks']:
        infile_tracks = parameters['infile_tracks']
    else:
        infile_tracks = 'None'
    if parameters['contact']:
        contact = 'On'
    else:
        contact = 'Off'


    def main():
        """
            Main processing function for Migrate3D. It reads input files, checks column names, processes data, performs calculations,
            and saves the results to an Excel file.
        """
        bigtic = tempo.time()
        try:
            p_bar_increase = 0.10
            while p_bar_increase < 1:
                dpg.set_value('pbar', p_bar_increase)

                # Get parameters
                infile_name = parameters['infile_segments']
                infile_segments = pd.read_csv(infile_name, sep=',')
                savefile = parameters['savefile']
                df_infile = pd.DataFrame(infile_segments)
                parent_id = parameters['object_id_col_name']
                time_for = parameters['time_col_name']
                x_for = parameters['x_col_name']
                y_for = parameters['y_col_name']
                z_for = parameters['z_col_name']

                # Check if the segements file column names match
                expected_columns = [parent_id, time_for, x_for, y_for, z_for]
                for col in expected_columns:
                    if col not in df_infile.columns:
                        print(f"Error: Column '{col}' not found in Segments input file. Please fix the column names.")
                        return

                # Check if the column names match in infile_tracks
                if parameters['infile_tracks']:
                    df_tracks = pd.read_csv(track_file, sep=',')
                    object_id_2 = parameters['object_id_2_col']
                    category_col_name = parameters['category_col']
                    expected_track_columns = [object_id_2, category_col_name]
                    for col in expected_track_columns:
                        if col not in df_tracks.columns:
                            print(f"Error: Column '{col}' not found in Tracks input file. Please fix the column names.")
                            return

                # Add a blank 'Z' column if it doesn't exist
                if z_for not in df_infile.columns:
                    df_infile[z_for] = 0

                # Get data for each object from segments file and add to list
                input_data_list = []
                for row in df_infile.index:
                    object_id = df_infile[parent_id][row]
                    time_col = df_infile[time_for][row]
                    x_col = df_infile[x_for][row]
                    y_col = df_infile[y_for][row]
                    z_col = df_infile[z_for][row]
                    input_data_list.append([object_id, time_col, x_col, y_col, z_col])

                # Create array of all objects, timepoints, and coordinates
                arr_segments = np.array(input_data_list)

                # Create settings DF for Excel output
                settings = {'Segments file': [os.path.basename(infile_name).split('/')[-1]],
                            'Categories file': [os.path.basename(infile_tracks).split('/')[-1]],
                            'Timelapse Interval': [timelapse_interval], 'Arrest Limit': [arrest_limit],
                            'Min. TP Moving': [num_of_tp_moving], 'Max. Contact Length': [contact_length],
                            'Arrested': [arrested], 'Tau (MSD)': [parameters['tau_msd']], 'Tau (Euclid/Angle)': [tau_euclid],
                            'Interpolation': [interpolate], 'Multitracking': [multi_track], 'Adjust to 2D': [two_dim],
                            'Contacts': [contact]}
                df_settings = pd.DataFrame(data=settings)

                # Sort segments array by time and object ID
                arr_segments = arr_segments[arr_segments[:, 1].argsort()]
                arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]

                # Create array containing unique object IDs
                unique_objects = np.unique(arr_segments[:, 0])

                tic = tempo.time()

                # Format dataset
                print('Formatting input dataset:\n' + infile_name + '...')
                formatting_dfs = {}
                if parameters['multi_track']:
                    arr_segments = multi_tracking(arr_segments)
                if parameters['two_dim']:
                    arr_segments = adjust_2D(arr_segments)
                if parameters['interpolate']:
                    arr_segments = interpolate_lazy(arr_segments, timelapse_interval, unique_objects)

                # Create dataframe of formatted segments for later export to Excel file
                df_segments = pd.DataFrame(arr_segments, columns=['Object ID', 'Time', 'X', 'Y', z_for])

                toc = tempo.time()
                print('...Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

                tic = tempo.time()
                print('Calculating migration parameters...')

                # Perform calculations on each unique object
                all_calcs = []
                for object in unique_objects:
                    object_data = arr_segments[arr_segments[:, 0] == object, :]
                    object_id = object_data[0, 0]
                    df_calcs = calculations(object, object_data, tau_euclid, object_id, parameters)
                    p_bar_increase += 0.0001
                    dpg.set_value('pbar', p_bar_increase)
                    all_calcs.append(df_calcs)
                df_all_calcs = pd.concat(all_calcs)
                mapping = {0: None}

                toc = tempo.time()
                print('...Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)

                # Create categories dataframe
                track_df = pd.DataFrame()
                track_input_list = []
                object_id_2 = parameters['object_id_2_col']
                category_col_name = parameters['category_col']

                if parameters['infile_tracks']:
                    track_df = pd.DataFrame(pd.read_csv(track_file))
                    for row in track_df.index:
                        object_id2 = track_df[object_id_2][row]
                        category = track_df[category_col_name][row]
                        track_input_list.append([object_id2, category])
                    arr_tracks = np.array(track_input_list)
                else:
                    arr_tracks = np.zeros_like(arr_segments)

                # Create summary sheet of calculations
                df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_sum_cat = summary_sheet(arr_segments,
                                                                                                         df_all_calcs,
                                                                                                         unique_objects,
                                                                                                         parameters['tau_msd'],
                                                                                                         parameters,
                                                                                                         arr_tracks, savefile)

                tic = tempo.time()
                print('Detecting attractors...')
                # Create a mapping from object IDs to cell types
                cell_types = dict(zip(track_df[parameters['object_id_2_col']], track_df[parameters['category_col']]))
                attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile)
                toc = tempo.time()
                print('...Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)

                # Check if contacts parameter is true and if so call contacts functions
                if contact_parameter is False:
                    pass
                else:
                    tic = tempo.time()
                    print('Detecting contacts...')

                    df_contacts, df_no_daughter, df_no_dead_, df_contact_summary = asyncio.run(
                        parallel_contacts.main(unique_objects, arr_segments, parameters['contact_length'], df_sum,
                                               parameters['arrested'], timelapse_interval)
                    )
                    toc = tempo.time()
                    print('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
                # Check if contacts parameter is true and if so call contacts functions
                """if contact_parameter is False:
                    pass
                else:
                    tic = tempo.time()
                    print('Detecting contacts...')
                    df_cont = contacts(unique_objects, arr_segments, contact_length)
                    if len(df_cont) == 0:
                        pass
                    else:
                        df_contacts = pd.concat(df_cont, ignore_index=True)
                        df_no_daughter_func = no_daughter_contacts(unique_objects, df_contacts)
                        df_no_daughter = pd.concat(df_no_daughter_func, ignore_index=True)
                        df_alive, df_contact_sum = contacts_moving(df_sum, df_no_daughter,
                                                                   arrested, time_interval)
                    df_no_dead_ = pd.concat(df_alive, ignore_index=True)
                    with_contacts = []
                    for df in df_contact_sum:
                        if df['Median Contact Duration'].notna().any():
                            with_contacts.append(df)
                    df_contact_summary = pd.concat(with_contacts, ignore_index=True)
                    df_contact_summary = df_contact_summary.replace(mapping)
                    df_contact_summary = df_contact_summary.dropna()
                    toc = tempo.time()
                    print('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))"""

                # Replace zero with None
                df_all_calcs = df_all_calcs.replace(mapping)
                df_sum = df_sum.replace(mapping)

                # If categories are present, restore zeroes for category
                if track_df.shape[0] > 0:
                    df_sum[category_col_name] = df_sum[category_col_name].replace(np.nan, 0)

                # restore zeros for Arrest Coefficient
                df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))

                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)

                # Create file path
                savepath = savefile + '.xlsx'
                print('Saving main output to ' + savepath + '...')
                savecontacts = savefile + '_Contacts.xlsx'

                # Save results to Excel file
                try:
                    with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
                        df_settings.to_excel(workbook, sheet_name='Settings', index=False)
                        df_segments.to_excel(workbook, sheet_name='Object Data', index=False)
                        df_all_calcs.to_excel(workbook, sheet_name='Calculations', index=False)
                        df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
                        df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
                        df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
                        df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summaries All', index=True)
                        if parameters['infile_tracks']:
                            df_msd_sum_cat.to_excel(workbook, sheet_name='MSD Per Category', index=True)
                        else:
                            pass
                # if Excel export fails, save results to CSV
                except:
                    print('ExcelWriter has thrown an exception due to the output file being too large. Outputs will be in .CSV format.')
                    df_settings.to_csv(f'{savefile}_Settings.csv', index=False)
                    df_segments.to_csv(f'{savefile}_Object_Data.csv', index=False)
                    df_all_calcs.to_csv(f'{savefile}_Calculations.csv', index=False)
                    df_sum.to_csv(f'{savefile}_Summary_Statistics.csv', index=False)
                    df_single.to_csv(f'{savefile}_Single_Timepoint_Medians.csv', index=False)
                    df_msd.to_csv(f'{savefile}_Mean_Squared_Displacements.csv', index=False)
                    df_msd_sum_all.to_csv(f'{savefile}_MSD_Summaries_All', index=True)
                    if parameters['infile_tracks']:
                        df_msd_sum_cat.to_csv(f'{savefile}_MSD_Per_Category', index=True)
                    else:
                        pass

                # If contacts were detected, save contacts to separate Excel file
                finally:
                    if contact_parameter is False:
                        pass
                        """else:
                            if len(df_cont) == 0:
                                pass"""
                    else:
                        print('Saving contacts output to ' + savecontacts + '...')
                        with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
                            df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                            df_no_daughter.to_excel(workbook, sheet_name='Contacts no Division', index=False)
                            df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                            df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

                p_bar_increase += 0.20
                dpg.set_value('pbar', p_bar_increase)

                print("Migrate3D done!")
                bigtoc = tempo.time()

                # Display total runtime
                total_time_sec = (int(round((bigtoc - bigtic), 1)))
                total_time_min = round((total_time_sec / 60), 1)
                if total_time_sec < 180:
                    print('Total time taken = {:.0f} seconds.'.format(total_time_sec))
                else:
                    print('Total time taken = {:.1f} minutes.'.format(total_time_min))

                dpg.destroy_context()
        except IndentationError:
            with dpg.window(label='ERROR', width=400, height=600) as err_win:
                dpg.add_input_text(default_value='Input error, please ensure all inputs are correct', width=400)
                dpg.set_value('pbar', 0)

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    main()


def formatting_check(sender, app_data):
    """
        Callback function to handle formatting check based on user input in the GUI.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
    """
    if dpg.get_value(sender) is True:
        parameters[sender] = True
    else:
        parameters[sender] = False


def run_contact(sender, app_data):
    """
        Callback function to handle contact detection based on user input in the GUI.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
    """
    if dpg.get_value(sender) is True:
        parameters['contact'] = True
    else:
        parameters['contact'] = False


def callback_file_segs(sender, app_data):
    """
        Callback function to handle file selection for the segments file.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
    """
    infile = str(app_data['file_path_name'])
    parameters['infile_segments'] = infile


def callback_file_cats(sender, app_data):
    """
        Callback function to handle file selection for the categories file.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
    """
    infile = str(app_data['file_path_name'])
    parameters['infile_tracks'] = infile


def input_return(sender, app_data):
    """
       Callback function to handle text input return in the GUI.
       Args:
           sender: The sender of the event.
           app_data: Additional data associated with the event.
   """
    parameters[sender] = app_data


def float_return(sender, app_data):
    """
        Callback function to handle float input return in the GUI.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
        """
    parameters[sender] = app_data


def start_migrate(sender, app_data):
    """
        Callback function to start the Migrate3D analysis when the 'Run' button is clicked in the GUI.
        Args:
            sender: The sender of the event.
            app_data: Additional data associated with the event.
        """
    migrate3D(parameters)


with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_segs,
                     file_count=3,
                     tag="segs_dialog_tag"):
    dpg.add_file_extension("", color=(255, 99, 71, 255))
    dpg.add_file_extension(".csv", color=(0, 150, 255, 255))
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".xlsx", color=(255, 219, 88, 255))
    dpg.add_file_extension(".py", color=(147, 197, 114, 255))

with dpg.file_dialog(width=700, height=550, directory_selector=False, show=False, callback=callback_file_cats,
                     file_count=3,
                     tag="cats_dialog_tag"):
    dpg.add_file_extension("", color=(255, 99, 71, 255))
    dpg.add_file_extension(".csv", color=(0, 150, 255, 255))
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".xlsx", color=(255, 219, 88, 255))
    dpg.add_file_extension(".py", color=(147, 197, 114, 255))

with dpg.window(label="Migrate3D", width=900, height=660) as Window:
    dpg.add_button(width=140, label="Open Segments File", callback=lambda: dpg.show_item("segs_dialog_tag"))
    dpg.add_button(width=155, label="Open Categories File", callback=lambda: dpg.show_item("cats_dialog_tag"))

    timelapse = dpg.add_input_float(width=100, label='Timelapse interval (in same units as Time data in input dataset)',
                                    default_value=parameters['timelapse'], callback=input_return, tag='timelapse')
    arrest_limit = dpg.add_input_float(width=100, label="Arrest limit (in same units as XYZ coordinates in input dataset)",
                                              default_value=parameters['arrest_limit'], callback=float_return, tag='arrest_limit')
    moving = dpg.add_input_int(width=100, label='Minimum timepoints moving (number of recorded timepoints required for a track to be included in analysis)',
                               default_value=parameters['moving'], callback=input_return, tag='moving')
    contact_length = dpg.add_input_float(width=100, label='Maximum contact length (maximum distance between objects that would be considered a contact)',
                                         default_value=parameters['contact_length'], callback=float_return, tag='contact_length')
    arrested = dpg.add_input_float(width=100, label='Arrested/Dead: Objects with an Arrest Coefficient above this value will be considered to be "dead" during Contact detection',
                                   default_value=parameters['arrested'], callback=float_return, tag='arrested')
    tau_msd = dpg.add_input_int(width=100, label='Maximum tau value (time lags) for mean square displacement (MSD) calculations',
                                default_value=parameters['tau_msd'], callback=input_return, tag='tau_msd')
    tau_euclid = dpg.add_input_int(width=100, label='Maximum tau value (time lags) for Euclidean Distance and Turning Angle calculations',
                                 default_value=parameters['tau_euclid'], callback=input_return, tag='tau_euclid')
    save_file = dpg.add_input_text(width=250, label='Output filename (.xlsx extension will be added, do not include!)',
                                   default_value=('{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results'), callback=input_return, tag='savefile')
    parent_id = dpg.add_input_text(width=150, label='Column header name in input Segments file for object identifiers',
                                   default_value='Parent ID', callback=input_return, tag='object_id_col_name')
    time_col = dpg.add_input_text(width=150, label='Column header name in input Segments file for Time data',
                                  default_value='Time', callback=input_return, tag='time_col_name')
    x_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for X coordinates',
                               default_value='X Coordinate', callback=input_return, tag='x_col_name')
    y_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for Y coordinates',
                               default_value='Y Coordinate', callback=input_return, tag='y_col_name')
    z_for = dpg.add_input_text(width=150, label='Column header name in input Segments file for Z coordinates',
                               default_value='Z Coordinate', callback=input_return, tag='z_col_name')
    parent_id2 = dpg.add_input_text(width=150, label='Column header name in input file for object identifiers',
                                    default_value='ID', callback=input_return, tag='object_id_2')
    category_col = dpg.add_input_text(width=150, label="Column header name in input Categories file for object category",
                                      default_value='Category', callback=input_return, tag='category_col')
    interpolate_check = dpg.add_checkbox(label='Perform lazy interpolation for missing data points?',
                                         callback=formatting_check, tag='interpolate')
    multi_check = dpg.add_checkbox(label='Average out any multi-tracked time points?',
                                   callback=formatting_check, tag='multi_track')
    two_dim_check = dpg.add_checkbox(label='Adjust for 2D data? Check if input dataset is 2D, or to convert 3D data to 2D by '
                                           'ignoring Z coordinates.', callback=formatting_check, tag='two_dim')
    contact = dpg.add_checkbox(label='Detect contacts? (Warning: can significantly increase processing time!)',
                               callback=run_contact)
    pca_filter = dpg.add_input_text(width=250, label='To limit PCA analysis to certain categories, enter them here '
                                    'separated by a comma.', callback=input_return, tag='pca_filter')

    dpg.add_progress_bar(width=600, height=10, label='Progress Bar', tag='pbar')
    dpg.add_button(width=100, label='Run', callback=start_migrate)
    dpg.create_viewport(title='Migrate3D', width=900, height=700)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()