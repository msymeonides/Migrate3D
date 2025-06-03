import pandas as pd
import numpy as np
import os
from formatting import multi_tracking, interpolate_lazy
from calculations import calculations
import time as tempo
from summary_sheet import summary_sheet
from attract import attract
import parallel_contacts
from summarize_contacts import summarize_contacts
pd.set_option('future.no_silent_downcasting', True)


def migrate3D(parent_id, time_for, x_for, y_for, z_for, timelapse_interval, arrest_limit, moving, contact_length,
              arrested, tau_msd, tau_euclid, formatting_options, savefile, segments_file_name, tracks_file, parent_id2,
              category_col_name, parameters, pca_filter, progress_callback = None):
    bigtic = tempo.time()
    # Update parameter dictionary
    parameters['savefile'] = savefile
    parameters['x_col_name'] = x_for
    parameters['object_id_col_name'] = parent_id
    parameters['time_col_name'] = time_for
    parameters['x_col_name'] = x_for
    parameters['y_col_name'] = y_for
    parameters['z_col_name'] = z_for
    parameters['timelapse'] = timelapse_interval
    parameters['arrest_limit'] = arrest_limit
    parameters['moving'] = moving
    parameters['contact_length'] = contact_length
    parameters['arrested'] = arrested
    parameters['tau_msd'] = tau_msd
    parameters['tau_euclid'] = tau_euclid
    parameters['pca_filter'] = pca_filter

    # formatting options for param update
    if formatting_options is None:
        pass
    else:
        if 'Multitrack' in formatting_options:
            parameters['multi_track'] = True
        if 'Interpolate' in formatting_options:
            parameters['interpolate'] = True
        if 'Verbose' in formatting_options:
            parameters['verbose'] = True
        if 'Contacts' in formatting_options:
            parameters['contact'] = True
        if 'Attractors' in formatting_options:
            parameters['attractors'] = True

    # Check if tracks used at all and update accordingly
    if 'Enter your category .csv file here by clicking or dropping (optional):' in tracks_file:
        pass
    else:
        parameters['infile_tracks'] = True
        parameters['object_id_2_col'] = parent_id2
        parameters['category_col'] = category_col_name

    # Build the content folder path
    content_folder = os.path.join(os.getcwd())

    # Specify the file name you want to reference
    segments_file_name = segments_file_name

    # Construct the full file path
    infile_path = os.path.join(content_folder, segments_file_name)

    # Assign the file path to parameters (update key as needed)
    parameters['infile_segments'] = infile_path

    infile_name = parameters['infile_segments']
    infile_segments = pd.read_csv(infile_name, sep=',')
    df_infile = pd.DataFrame(infile_segments)

    if z_for is None:
        df_infile[z_for] = 0

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

    settings = {'Segments file': [os.path.basename(infile_name).split('/')[-1]],
                'Categories file': [os.path.basename(tracks_file).split('/')[-1]],
                'Timelapse Interval': [timelapse_interval], 'Arrest Limit': [arrest_limit],
                'Min. TP Moving': parameters['moving'], 'Max. Contact Length': [contact_length],
                'Arrested': [arrested], 'Tau (MSD)': [parameters['tau_msd']], 'Tau (Euclid/Angle)': [tau_euclid],
                'Interpolation': parameters['interpolate'], 'Multitracking': [parameters['multi_track']],
                'Contacts': parameters['contact']}
    df_settings = pd.DataFrame(data=settings)


    # Sort segments array by time and object ID
    arr_segments = arr_segments[arr_segments[:, 1].argsort()]
    arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]

    # Create array containing unique object IDs
    unique_objects = np.unique(arr_segments[:, 0])
    tic = tempo.time()

    # Format dataset
    print('Formatting input dataset:\n' + infile_name + '...')
    if progress_callback:
        progress_callback(10)

    if parameters['multi_track']:
        arr_segments = multi_tracking(arr_segments)
    if parameters['interpolate']:
        arr_segments = interpolate_lazy(arr_segments, timelapse_interval, unique_objects)

    # Create dataframe of formatted segments for later export to Excel file
    df_segments = pd.DataFrame(arr_segments, columns=['Object ID', 'Time', 'X', 'Y', z_for])
    toc = tempo.time()
    print('...Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
    if progress_callback:
        progress_callback(10)

    tic = tempo.time()
    print('Calculating migration parameters...')
    if progress_callback:
        progress_callback(5)
    # Perform calculations on each unique object
    all_calcs = []

    for object in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == object, :]
        object_id = object_data[0, 0]
        df_calcs = calculations(object_data, tau_euclid, object_id, parameters)
        all_calcs.append(df_calcs)
    df_all_calcs = pd.concat(all_calcs)
    mapping = {0: None}
    toc = tempo.time()
    print('...Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
    if progress_callback:
        progress_callback(25)

    # Create categories dataframe
    track_df = pd.DataFrame()
    track_input_list = []
    object_id_2 = parameters['object_id_2_col']
    category_col_name = parameters['category_col']

    if parameters['infile_tracks']:
        track_df = pd.DataFrame(pd.read_csv(tracks_file))
        track_df = track_df[[parameters['object_id_2_col'], parameters['category_col']]]
        for row in track_df.index:
            object_id2 = track_df[object_id_2][row]
            category = track_df[category_col_name][row]
            track_input_list.append([object_id2, category])
        track_df.columns = ['Object ID', 'Category']
        arr_tracks = np.array(track_input_list)
    else:
        arr_tracks = np.zeros_like(arr_segments)

    # Create summary sheet of calculations
    print('Running Summary Sheet...')
    if progress_callback:
        progress_callback(5)
    df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_sum_cat, df_pca = summary_sheet(arr_segments,
                                                                                                     df_all_calcs,
                                                                                                     unique_objects,
                                                                                                     parameters[
                                                                                                         'tau_msd'],
                                                                                                     parameters,
                                                                                                     arr_tracks,
                                                                                                     savefile)

    tic = tempo.time()

    if parameters['attractors'] and parameters['infile_tracks']:
        print('Detecting attractors...')
        if progress_callback:
            progress_callback(25)
        # Create a mapping from object IDs to cell types
        cell_types = dict(zip(track_df['Object ID'], track_df['Category']))
        attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile)
        toc = tempo.time()
        print('...Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
        if progress_callback:
            progress_callback(0)

    else:
        pass

    # Check if contacts parameter is true and if so call contacts functions
    if parameters['contact'] is False:
        pass
    else:
        tic = tempo.time()
        print('Detecting contacts...')
        if progress_callback:
            progress_callback(0)
        # Extract the timepoints from arr_segments (column index 1) and get unique values
        unique_timepoints = np.unique(arr_segments[:, 1])
        # Then pass unique_timepoints to parallel_contacts.main instead of unique_objects
        df_contacts, df_no_daughter, df_no_dead_ = parallel_contacts.main(
            unique_timepoints,  # using timepoints for chunking
            arr_segments,
            parameters['contact_length'],
            df_sum,
            parameters['arrested'],
            timelapse_interval
        )
        if not df_no_dead_.empty:
            df_contact_summary = summarize_contacts(df_no_dead_, timelapse_interval)
            print(f"Contact summary created with {len(df_contact_summary)} rows.")
        else:
            df_contact_summary = pd.DataFrame()
            print("No valid contacts detected; skipping summary processing.")

        toc = tempo.time()
        print('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
        if progress_callback:
            progress_callback(5)

    # Replace zero with None
    df_all_calcs = df_all_calcs.replace(mapping)
    df_sum = df_sum.replace(mapping)

    # If categories are present, restore zeroes for category
    if track_df.shape[0] > 0:
        df_sum['Category'] = df_sum['Category'].replace(np.nan, 0)

    # restore zeros for Arrest Coefficient
    df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))

    # Create file path
    savepath = savefile + '.xlsx'
    print('Saving main output to ' + savepath + '...')
    if progress_callback:
        progress_callback(100)
    savecontacts = savefile + '_Contacts.xlsx'

    # Save results to Excel file
    if parameters['verbose']:  # Save all data to Excel file
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

    else:  # Save only summary data to Excel file
        with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
            df_settings.to_excel(workbook, sheet_name='Settings', index=False)
            df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
            df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
            df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
            df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summaries All', index=True)
            if parameters['infile_tracks']:
                df_msd_sum_cat.to_excel(workbook, sheet_name='MSD Per Category', index=True)
            else:
                pass

        # If contacts were detected, save contacts to separate Excel file
        if parameters['contact'] is False:
            pass
        else:
            print('Saving contacts output to ' + savecontacts + '...')
            with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
                df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                df_no_daughter.to_excel(workbook, sheet_name='Contacts no Division', index=False)
                df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

        if progress_callback:
            progress_callback(10)
        print("Migrate3D done!")
        if progress_callback:
            progress_callback(100)
        bigtoc = tempo.time()

        # Display total runtime
        total_time_sec = (int(round((bigtoc - bigtic), 1)))
        total_time_min = round((total_time_sec / 60), 1)
        if total_time_sec < 180:
            print('Total time taken = {:.0f} seconds.'.format(total_time_sec))
        else:
            print('Total time taken = {:.1f} minutes.'.format(total_time_min))

        if progress_callback:
            progress_callback(100)


    return df_segments, df_sum, df_pca
