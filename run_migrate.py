import pandas as pd
import os
from formatting import multi_tracking, adjust_2D, interpolate_lazy
import time as tempo
from summary_sheet import summary_sheet
from attract import attract



    else:
    else:



    infile_name = parameters['infile_segments']
    infile_segments = pd.read_csv(infile_name, sep=',')
    df_infile = pd.DataFrame(infile_segments)

                # Detect timelapse interval
                if parameters['timelapse'] != 0:
                    timelapse_interval = round((parameters['timelapse']), 3)
                else:
                    time_data = df_infile[time_for].values
                    time_diffs = np.diff(time_data)
                    print(time_diffs)
                    timelapse_interval = mode(time_diffs).mode[0]
                    print(timelapse_interval)

                # Detect 2D data
                if (z_for == '') | (z_for not in df_infile.columns):
                    print('2D dataset detected. If your dataset is 3D, check your Z column name.')

    for col in expected_columns:
        if col not in df_infile.columns:
            print(f"Error: Column '{col}' not found in Segments input file. Please fix the column names.")
            return
            # Check if the column names match in infile_tracks
    if parameters['infile_tracks']:
        object_id_2 = parameters['object_id_2_col']
        category_col_name = parameters['category_col']
        expected_track_columns = [object_id_2, category_col_name]
        for col in expected_track_columns:
            if col not in df_tracks.columns:
                print(f"Error: Column '{col}' not found in Tracks input file. Please fix the column names.")
                return

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
                'Timelapse Interval': [timelapse_interval], 'Arrest Limit': [arrest_limit],
                'Arrested': [arrested], 'Tau (MSD)': [parameters['tau_msd']], 'Tau (Euclid/Angle)': [tau_euclid],
    df_settings = pd.DataFrame(data=settings)

    # Sort segments array by time and object ID
    arr_segments = arr_segments[arr_segments[:, 1].argsort()]
    arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]

    # Create array containing unique object IDs
    unique_objects = np.unique(arr_segments[:, 0])
    tic = tempo.time()

    # Format dataset
    print('Formatting input dataset:\n' + infile_name + '...')

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
        df_calcs = calculations(object_data, tau_euclid, object_id, parameters)
        all_calcs.append(df_calcs)
    df_all_calcs = pd.concat(all_calcs)
    mapping = {0: None}
    toc = tempo.time()
    print('...Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    # Create categories dataframe
    track_df = pd.DataFrame()
    track_input_list = []
    object_id_2 = parameters['object_id_2_col']
    category_col_name = parameters['category_col']

    if parameters['infile_tracks']:
        for row in track_df.index:
            object_id2 = track_df[object_id_2][row]
            category = track_df[category_col_name][row]
            track_input_list.append([object_id2, category])
        arr_tracks = np.array(track_input_list)
    else:
        arr_tracks = np.zeros_like(arr_segments)

    # Create summary sheet of calculations
                                                                                                     df_all_calcs,
                                                                                                     unique_objects,
                                                                                                     parameters,
    tic = tempo.time()
        print('Detecting attractors...')
        # Create a mapping from object IDs to cell types
        cell_types = dict(zip(track_df[parameters['object_id_2_col']], track_df[parameters['category_col']]))
        attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile)
        toc = tempo.time()
        print('...Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))


    # Check if contacts parameter is true and if so call contacts functions
        pass
    else:
        tic = tempo.time()
        print('Detecting contacts...')
        else:
        toc = tempo.time()
        print('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    # Replace zero with None
    df_all_calcs = df_all_calcs.replace(mapping)
    df_sum = df_sum.replace(mapping)

    # If categories are present, restore zeroes for category
    if track_df.shape[0] > 0:
        df_sum[category_col_name] = df_sum[category_col_name].replace(np.nan, 0)

    # restore zeros for Arrest Coefficient
    df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))


    # Create file path
    savepath = savefile + '.xlsx'
    print('Saving main output to ' + savepath + '...')
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
            pass
        else:
            print('Saving contacts output to ' + savecontacts + '...')
            with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
                df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                df_no_daughter.to_excel(workbook, sheet_name='Contacts no Division', index=False)
                df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

        print("Migrate3D done!")
        bigtoc = tempo.time()

        # Display total runtime
        total_time_sec = (int(round((bigtoc - bigtic), 1)))
        total_time_min = round((total_time_sec / 60), 1)
        if total_time_sec < 180:
            print('Total time taken = {:.0f} seconds.'.format(total_time_sec))
        else:
            print('Total time taken = {:.1f} minutes.'.format(total_time_min))


