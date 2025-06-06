import pandas as pd
import numpy as np
import time as tempo
from pathlib import Path

from formatting import multi_tracking, interpolate_lazy
from calculations import calculations
from summary_sheet import summary_sheet
from attract import attract
import parallel_contacts
from summarize_contacts import summarize_contacts
from shared_state import messages, thread_lock, set_progress
pd.set_option('future.no_silent_downcasting', True)


def migrate3D(parent_id, time_for, x_for, y_for, z_for, timelapse_interval, arrest_limit, moving, contact_length,
              arrested, tau_msd, tau_euclid, formatting_options, savefile, segments_file_name, tracks_file, parent_id2,
              category_col_name, parameters, pca_filter):
    bigtic = tempo.time()

    with thread_lock:
        messages.append('Starting Migrate3D...')

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

    if 'Enter your category .csv file here by clicking or dropping (optional):' in tracks_file:
        pass
    else:
        parameters['infile_tracks'] = True
        parameters['object_id_2_col'] = parent_id2
        parameters['category_col'] = category_col_name

    seg_path = Path(segments_file_name).expanduser().resolve()
    parameters['infile_segments'] = str(seg_path)

    infile_segments = pd.read_csv(parameters['infile_segments'], sep=',')
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

    arr_segments = np.array(input_data_list)

    settings = {'Segments file': [seg_path.name],
                'Categories file': [Path(tracks_file).name],
                'Timelapse Interval': [timelapse_interval], 'Arrest Limit': [arrest_limit],
                'Min. TP Moving': parameters['moving'], 'Max. Contact Length': [contact_length],
                'Arrested': [arrested], 'Tau (MSD)': [parameters['tau_msd']], 'Tau (Euclid/Angle)': [tau_euclid],
                'Interpolation': parameters['interpolate'], 'Multitracking': [parameters['multi_track']],
                'Contacts': parameters['contact']}
    df_settings = pd.DataFrame(data=settings)

    arr_segments = arr_segments[arr_segments[:, 1].argsort()]
    arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]

    unique_objects = np.unique(arr_segments[:, 0])
    tic = tempo.time()

    with thread_lock:
        messages.append(f"Formatting input dataset:\n{seg_path}...")

    if parameters['multi_track']:
        arr_segments = multi_tracking(arr_segments)
    if parameters['interpolate']:
        arr_segments = interpolate_lazy(arr_segments, timelapse_interval, unique_objects)

    df_segments = pd.DataFrame(arr_segments, columns=['Object ID', 'Time', 'X', 'Y', z_for])
    toc = tempo.time()

    with thread_lock:
        messages.append('...Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    set_progress(5)

    tic = tempo.time()

    with thread_lock:
        messages.append('Calculating migration parameters...')

    all_calcs = []

    for object in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == object, :]
        object_id = object_data[0, 0]
        df_calcs = calculations(object_data, tau_euclid, object_id, parameters)
        all_calcs.append(df_calcs)
    df_all_calcs = pd.concat(all_calcs)
    mapping = {0: None}
    toc = tempo.time()

    with thread_lock:
        messages.append('...Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    set_progress(25)

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

    with thread_lock:
        messages.append('Running Summary Sheet...')

    tic = tempo.time()
    df_sum, time_interval, df_single, df_msd, df_msd_sum_all, df_msd_avg_per_cat, df_msd_std_per_cat, df_pca = summary_sheet(arr_segments,
                  df_all_calcs, unique_objects, parameters['tau_msd'], parameters, arr_tracks, savefile)
    toc = tempo.time()

    with thread_lock:
        messages.append('...Summary sheet done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

    set_progress(15)

    tic = tempo.time()

    if parameters['attractors'] and parameters['infile_tracks']:
        with thread_lock:
            messages.append('Detecting attractors...')

        cell_types = dict(zip(track_df['Object ID'], track_df['Category']))
        attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile)
        toc = tempo.time()

        with thread_lock:
            messages.append('...Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

        set_progress(10)

    else:
        pass

    if parameters['contact'] is False:
        pass
    else:
        tic = tempo.time()

        with thread_lock:
            messages.append('Detecting contacts...')

        unique_timepoints = np.unique(arr_segments[:, 1])
        df_contacts, df_no_daughter, df_no_dead_ = parallel_contacts.main(
            unique_timepoints,
            arr_segments,
            parameters['contact_length'],
            df_sum,
            parameters['arrested'],
            timelapse_interval
        )
        if not df_no_dead_.empty:
            df_contact_summary = summarize_contacts(df_no_dead_, timelapse_interval)
        else:
            df_contact_summary = pd.DataFrame()
            with thread_lock:
                messages.append("No valid contacts detected; skipping summary processing.")

        toc = tempo.time()

        with thread_lock:
            messages.append('...Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))

        set_progress(20)

    df_all_calcs = df_all_calcs.replace(mapping)
    df_sum = df_sum.replace(mapping)

    if track_df.shape[0] > 0:
        df_sum['Category'] = df_sum['Category'].replace(np.nan, 0)

    df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))

    savepath = savefile + '.xlsx'

    with thread_lock:
        messages.append('Saving main output to ' + savepath + '...')

    savecontacts = savefile + '_Contacts.xlsx'

    if parameters['verbose']:
        with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
            df_settings.to_excel(workbook, sheet_name='Settings', index=False)
            df_segments.to_excel(workbook, sheet_name='Object Data', index=False)
            df_all_calcs.to_excel(workbook, sheet_name='Calculations', index=False)
            df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
            df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
            df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
            df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summaries All', index=True)
            if parameters['infile_tracks']:
                df_msd_avg_per_cat.to_excel(workbook, sheet_name='MSD Avg Per Category', index=True)
                df_msd_std_per_cat.to_excel(workbook, sheet_name='MSD StDev Per Category', index=True)
            else:
                pass

    else:
        with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
            df_settings.to_excel(workbook, sheet_name='Settings', index=False)
            df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
            df_single.to_excel(workbook, sheet_name='Single Timepoint Medians', index=False)
            df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
            df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summary', index=True)
            if parameters['infile_tracks']:
                df_msd_avg_per_cat.to_excel(workbook, sheet_name='MSD Avg Per Category', index=True)
                df_msd_std_per_cat.to_excel(workbook, sheet_name='MSD StDev Per Category', index=True)
            else:
                pass

        if parameters['contact'] is False:
            pass
        else:
            with thread_lock:
                messages.append('Saving contacts output to ' + savecontacts + '...')
            with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
                df_contacts.to_excel(workbook, sheet_name='Contacts', index=False)
                df_no_daughter.to_excel(workbook, sheet_name='Contacts no Division', index=False)
                df_no_dead_.to_excel(workbook, sheet_name='Contacts no Dead', index=False)
                df_contact_summary.to_excel(workbook, sheet_name='Contact Summary', index=False)

        set_progress(100)
        bigtoc = tempo.time()

        total_time_sec = (int(round((bigtoc - bigtic), 1)))
        total_time_min = round((total_time_sec / 60), 1)
        if total_time_sec < 180:
            with thread_lock:
                messages.append('Migrate3D done! Total time taken = {:.0f} seconds.'.format(total_time_sec))
        else:
            with thread_lock:
                messages.append('Migrate3D done! Total time taken = {:.1f} minutes.'.format(total_time_min))

        set_progress(100)

    return df_segments, df_sum, df_pca
