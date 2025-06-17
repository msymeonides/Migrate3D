import pandas as pd
import numpy as np
import time as tempo
import statistics
import base64
import io
from pathlib import Path

from formatting import multi_tracking, interpolate_lazy, remove_tracks_with_gaps
from calculations import calculations
from summary_sheet import summary_sheet
from generate_figures import save_all_figures
from attractors import attract
import contacts_parallel
from shared_state import messages, thread_lock, complete_progress_step
pd.set_option('future.no_silent_downcasting', True)

def migrate3D(parent_id, time_for, x_for, y_for, z_for, timelapse_interval, arrest_limit, moving, contact_length,
              arrested, tau, formatting_options, savefile, segments_file_name, tracks_file, parent_id2,
              category_col_name, parameters, pca_filter, attract_params):
    bigtic = tempo.time()

    with thread_lock:
        messages.append('Starting Migrate3D...')
        messages.append('')
        print()

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
    parameters['tau'] = tau
    parameters['pca_filter'] = pca_filter
    parameters['attract_params'] = attract_params

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
        if 'Generate Figures' in formatting_options:
            parameters['generate_figures'] = True

    if tracks_file is not None and 'Enter your category .csv file here' not in str(tracks_file):
        parameters['infile_tracks'] = True
        parameters['object_id_2_col'] = parent_id2
        parameters['category_col'] = category_col_name
    else:
        parameters['infile_tracks'] = False

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
    arr_segments = arr_segments[arr_segments[:, 1].argsort()]
    arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]

    unique_objects = np.unique(arr_segments[:, 0])
    tic = tempo.time()

    with thread_lock:
        messages.append(f"Formatting input dataset:\n{seg_path}")

    dropped_objects = []
    df_removed = pd.DataFrame()

    if parameters['multi_track']:
        arr_segments = multi_tracking(arr_segments)
    if parameters['interpolate']:
        arr_segments = interpolate_lazy(arr_segments, timelapse_interval)
    if not parameters.get('interpolate', False):
        original_objects = set(unique_objects)
        arr_segments_filtered = remove_tracks_with_gaps(arr_segments, unique_objects, timelapse_interval)
        filtered_objects = set(np.unique(arr_segments_filtered[:, 0]))
        dropped_objects = sorted(float(obj) for obj in (original_objects - filtered_objects))
        arr_segments = arr_segments_filtered
        unique_objects = np.unique(arr_segments[:, 0])
        if dropped_objects:
            df_removed = pd.DataFrame({'Object ID': dropped_objects})
            with thread_lock:
                messages.append(f"Removed {len(dropped_objects)} object(s) with timepoint gaps "
                                f"(object IDs have been recorded in 'Removed Objects' sheet in the main output file).")

    df_segments = pd.DataFrame(arr_segments, columns=['Object ID', 'Time', 'X', 'Y', z_for])
    toc = tempo.time()

    with thread_lock:
        messages.append('...Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1))))
        messages.append('')
    complete_progress_step("Formatting")

    tic = tempo.time()

    with thread_lock:
        messages.append('Calculating migration parameters...')

    all_calcs = []

    for obj in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        object_id = object_data[0, 0]
        df_calcs = calculations(object_data, tau, object_id, parameters)
        all_calcs.append(df_calcs)
    df_all_calcs = pd.concat(all_calcs)
    mapping = {0: None}
    toc = tempo.time()

    with thread_lock:
        msg = ' Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append('')
    complete_progress_step("Calculations")

    track_df = pd.DataFrame()
    track_input_list = []
    object_id_2 = parameters['object_id_2_col']
    category_col_name = parameters['category_col']
    categories_file_name = None

    if parameters['infile_tracks']:
        if isinstance(tracks_file, str) and tracks_file.startswith('data:'):
            categories_file_name = parameters.get('category_file_name', 'Uploaded via base64')
            content_type, content_string = tracks_file.split(',')
            decoded = base64.b64decode(content_string)
            track_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif isinstance(tracks_file, str) and Path(tracks_file).is_file():
            categories_file_name = str(tracks_file)
            track_df = pd.read_csv(tracks_file)
        track_df = track_df[[parameters['object_id_2_col'], parameters['category_col']]]
        for row in track_df.index:
            object_id2 = track_df[object_id_2][row]
            category = track_df[category_col_name][row]
            track_input_list.append([object_id2, category])
        track_df.columns = ['Object ID', 'Category']
        arr_tracks = np.array(track_input_list)
    else:
        arr_tracks = np.zeros((0, 2))
        categories_file_name = 'None'

    settings = [
        ('Segments file', seg_path.name),
        ('Categories file', categories_file_name),
        ('Arrest limit', arrest_limit),
        ('Min. timepoints', parameters['moving']),
        ('Contact length', contact_length),
        ('Max. arrest coeff.', arrested),
        ('Timelapse interval', timelapse_interval),
        ('Max. Tau', parameters['tau']),
        ('Multitracking', parameters['multi_track']),
        ('Interpolation', parameters['interpolate'])
    ]
    df_settings = pd.DataFrame(settings, columns=['Parameter', 'Value'])

    (df_sum, df_single_euclid, df_single_angle, df_msd, df_msd_sum_all, df_msd_avg_per_cat, df_msd_std_per_cat,
     df_msd_loglogfits, df_pca) = summary_sheet(arr_segments, df_all_calcs, unique_objects, parameters['tau'],
                                               parameters, arr_tracks, savefile)

    savepath = savefile + '_Results.xlsx'
    savecontacts = savefile + '_Contacts.xlsx'

    df_contacts_summary = None
    df_contacts_per_category = None

    if parameters['contact'] is False:
        pass
    else:
        tic = tempo.time()

        with thread_lock:
            messages.append('Detecting contacts...')

        unique_timepoints = np.unique(arr_segments[:, 1])
        df_contacts_all, df_contacts_nodividing, df_contacts_nodead = contacts_parallel.main(
            unique_timepoints,
            arr_segments,
            parameters['contact_length'],
            df_sum,
            parameters['arrested'],
        )

        if not df_contacts_nodead.empty:
            summary_list = []
            df_contacts_final = df_contacts_nodead
            for object_id, group in df_contacts_final.groupby("Object ID"):
                unique_contacts = group["Object Compare"].unique()
                num_contacts = len(unique_contacts)
                total_time = len(group) * timelapse_interval
                n = len(group)
                durations = [(i + 1) * timelapse_interval for i in range(n)]
                if n == 1:
                    med_time = durations[0]
                elif n == 2:
                    med_time = sum(durations) / 2
                else:
                    med_time = statistics.median(durations)
                summary_list.append({
                    "Object ID": object_id,
                    "Number of Contacts": num_contacts,
                    "Total Time Spent in Contact": total_time,
                    "Median Contact Duration": med_time
                })
            df_contacts_summary = pd.DataFrame(summary_list)
        else:
            df_contacts_summary = pd.DataFrame()

        if parameters['infile_tracks']:
            if not df_contacts_summary.empty and 'Category' not in df_contacts_summary.columns:
                df_contacts_summary = df_contacts_summary.merge(
                    df_sum[['Object ID', 'Category']],
                    on='Object ID',
                    how='left'
                )
                cols = ['Object ID', 'Category'] + [col for col in df_contacts_summary.columns if
                                                    col not in ['Object ID', 'Category']]
                df_contacts_summary = df_contacts_summary[cols]

            all_objects = df_sum[['Object ID', 'Category']].copy()
            df_contacting = all_objects.merge(
                df_contacts_summary[
                    ['Object ID', 'Number of Contacts', 'Total Time Spent in Contact', 'Median Contact Duration']],
                on='Object ID', how='left'
            ).fillna({
                'Number of Contacts': 0,
                'Total Time Spent in Contact': 0,
                'Median Contact Duration': 0
            })

            per_cat = []
            for cat, group in df_contacting.groupby('Category'):
                total = group.shape[0]
                with_contact = group[group['Number of Contacts'] > 0]
                with_3plus = group[group['Number of Contacts'] >= 3]
                n_with_contact = with_contact.shape[0]
                n_with_3plus = with_3plus.shape[0]
                median_contacts = with_contact['Number of Contacts'].median() if n_with_contact > 0 else np.nan
                median_contact_time = with_contact['Total Time Spent in Contact'].median() if n_with_contact > 0 else np.nan
                median_duration = with_contact['Median Contact Duration'].median() if n_with_contact > 0 else np.nan
                per_cat.append({
                    'Category': cat,
                    'Total Objects': total,
                    'Pct With Contact': 100 * n_with_contact / total if total else np.nan,
                    'Pct With >=3 Contacts': 100 * n_with_3plus / total if total else np.nan,
                    'Median Contacts Per Object': median_contacts,
                    'Median Time Spent in Contact': median_contact_time,
                    'Median Contact Duration': median_duration,
                })
            df_contacts_per_category = pd.DataFrame(per_cat)
        else:
            df_contacts_per_category = pd.DataFrame()

        with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
            df_contacts_all.to_excel(workbook, sheet_name='Contacts (all)', index=False)
            df_contacts_nodividing.to_excel(workbook, sheet_name='Contacts (minus dividing)', index=False)
            df_contacts_nodead.to_excel(workbook, sheet_name='Contacts (minus dead)', index=False)
            df_contacts_summary.to_excel(workbook, sheet_name='Contacts Summary', index=False)
            if not df_contacts_per_category.empty:
                df_contacts_per_category.to_excel(workbook, sheet_name='Contacts Per Category', index=False)

        toc = tempo.time()

        with thread_lock:
            msg = ' Contacts done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
            messages[-1] += msg
            messages.append('')
        complete_progress_step('Contacts')

    if parameters['attractors'] is False:
        pass
    else:
        if parameters['infile_tracks']:
            tic = tempo.time()
            with thread_lock:
                messages.append('Detecting attractors...')

            cell_types = dict(zip(track_df['Object ID'], track_df['Category']))
            attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile, parameters['attract_params'])
            toc = tempo.time()
            with thread_lock:
                msg = ' Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
                messages[-1] += msg
                messages.append('')
            complete_progress_step('Attractors')
        else:
            with thread_lock:
                messages.append('Attractors bypassed; no category file provided.')
                messages.append('')
            complete_progress_step('Attractors')

    if parameters['generate_figures'] is False:
        pass
    else:
        if parameters['infile_tracks']:
            with thread_lock:
                messages.append('Generating figures...')
            tic = tempo.time()
            save_all_figures(df_sum, df_segments, df_pca, df_msd, df_msd_loglogfits, df_contacts_summary,
                             df_contacts_per_category, savefile, parameters['infile_tracks'])
            toc = tempo.time()
            with thread_lock:
                msg = ' Done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
                messages[-1] += msg
                messages.append('')
            complete_progress_step('Generate Figures')
        else:
            with thread_lock:
                messages.append('Figure generation bypassed; no categories file provided.')
                messages.append('')
            complete_progress_step('Generate Figures')

    with thread_lock:
        messages.append('Saving main output to ' + savepath + '...')
        messages.append('')

    df_all_calcs = df_all_calcs.replace(mapping)
    df_sum = df_sum.replace(mapping)
    if track_df.shape[0] > 0:
        df_sum['Category'] = df_sum['Category'].replace(np.nan, 0)
    if parameters['arrest_limit'] != 0:
        df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))

    with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
        df_settings.to_excel(workbook, sheet_name='Settings', index=False)
        if dropped_objects:
            df_removed.to_excel(workbook, sheet_name='Removed Objects', index=False)
        if parameters['verbose']:
            df_segments.to_excel(workbook, sheet_name='Object Data', index=False)
            df_all_calcs.to_excel(workbook, sheet_name='Calculations', index=False)
        df_sum.to_excel(workbook, sheet_name='Summary Statistics', index=False)
        df_single_euclid.to_excel(workbook, sheet_name='Euclidean Medians', index=False)
        df_single_angle.to_excel(workbook, sheet_name='Turning Angles', index=False)
        df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
        df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summary', index=True)
        if parameters['infile_tracks']:
            df_msd_avg_per_cat.to_excel(workbook, sheet_name='MSD Mean Per Category', index=True)
            df_msd_std_per_cat.to_excel(workbook, sheet_name='MSD StDev Per Category', index=True)
            df_msd_loglogfits.to_excel(workbook, sheet_name='MSD Log-Log Fits', index=True)

    complete_progress_step("Final results save")
    bigtoc = tempo.time()

    total_time_sec = (int(round((bigtoc - bigtic), 1)))
    total_time_min = round((total_time_sec / 60), 1)
    if total_time_sec < 180:
        with thread_lock:
            messages.append('------------------------------------------------')
            messages.append('Migrate3D done! Total time taken = {:.0f} seconds.'.format(total_time_sec))
            messages.append('------------------------------------------------')
    else:
        with thread_lock:
            messages.append('------------------------------------------------')
            messages.append('Migrate3D done! Total time taken = {:.1f} minutes.'.format(total_time_min))
            messages.append('------------------------------------------------')

    return df_segments, df_sum, df_pca
