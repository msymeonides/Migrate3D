import pandas as pd
import numpy as np
import time as tempo
import statistics
import re

from formatting import multi_tracking, interpolate_lazy, remove_tracks_with_gaps
from calculations import calculations
from summary_sheet import summary_sheet
from generate_figures import save_all_figures
from attractors import attract
import contacts_parallel
from shared_state import messages, thread_lock, set_abort_state, complete_progress_step
pd.set_option('future.no_silent_downcasting', True)

def migrate3D(parent_id, time_for, x_for, y_for, z_for, timelapse_interval, arrest_limit, moving, contact_length,
              arrested, min_maxeuclid, tau, options, savefile, segments_dataframe, categories_dataframe,
              segments_filename, categories_filename, parameters, pca_filter, attract_params):
    bigtic = tempo.time()

    with thread_lock:
        messages.append('Starting Migrate3D...')
        messages.append('')
        messages.append(f'Segments filename: {segments_filename}')
        messages.append(f'Categories filename: {categories_filename}')
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
    parameters['min_maxeuclid'] = min_maxeuclid
    parameters['tau'] = tau
    parameters['pca_filter'] = pca_filter
    parameters['attract_params'] = attract_params

    if options is None:
        pass
    else:
        if 'Multitrack' in options:
            parameters['multi_track'] = True
        if 'Interpolate' in options:
            parameters['interpolate'] = True
        if 'Verbose' in options:
            parameters['verbose'] = True
        if 'Contacts' in options:
            parameters['contact'] = True
        if 'ContactDivFilter' in options:
            parameters['contact_div_filter'] = True
        if 'Attractors' in options:
            parameters['attractors'] = True
        if 'Helicity' in options:
            parameters['helicity'] = True
        if 'Generate Figures' in options:
            parameters['generate_figures'] = True

    df_infile = segments_dataframe.copy()
    segments_file_display = segments_filename

    if z_for is None:
        df_infile[z_for] = 0

    input_data_list = []
    for row in df_infile.index:
        object_id = int(df_infile[parent_id][row])
        time_col = df_infile[time_for][row]
        x_col = df_infile[x_for][row]
        y_col = df_infile[y_for][row]
        z_col = df_infile[z_for][row]
        input_data_list.append([object_id, time_col, x_col, y_col, z_col])

    arr_segments = np.array(input_data_list, dtype=object)
    arr_segments = arr_segments[arr_segments[:, 1].argsort()]
    arr_segments = arr_segments[arr_segments[:, 0].argsort(kind='mergesort')]
    arr_segments[:, 0] = arr_segments[:, 0].astype(int)

    unique_objects = np.unique(arr_segments[:, 0]).astype(int)
    tic = tempo.time()

    with thread_lock:
        messages.append(f"Formatting input dataset...")

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
        dropped_objects = sorted(int(obj) for obj in (original_objects - filtered_objects))
        arr_segments = arr_segments_filtered
        unique_objects = np.unique(arr_segments[:, 0])
        if dropped_objects:
            df_removed = pd.DataFrame({
                'Object ID': dropped_objects,
                'Reason for removal': ['Gaps'] * len(dropped_objects)
            })
            df_removed['Object ID'] = df_removed['Object ID'].astype(int)

    df_segments = pd.DataFrame(arr_segments, columns=['Object ID', 'Time', 'X', 'Y', z_for])
    df_segments['Object ID'] = df_segments['Object ID'].astype(int)
    toc = tempo.time()

    with thread_lock:
        msg = ' Formatting done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append('')
    complete_progress_step("Formatting")

    if len(unique_objects) == 0:
        with thread_lock:
            messages.append("No data left after data formatting, aborting run.")
            messages.append("")
        set_abort_state()
        return None, None, None

    tic = tempo.time()

    with thread_lock:
        messages.append('Calculating migration parameters...')

    all_calcs = []
    all_angle_steps = None
    all_angle_medians = {}

    for obj in unique_objects:
        object_data = arr_segments[arr_segments[:, 0] == obj, :]
        object_id = object_data[0, 0]
        df_calcs, angle_steps, angle_medians_dict = calculations(object_data, tau, object_id, parameters)
        all_calcs.append(df_calcs)
        all_angle_medians[object_id] = angle_medians_dict
        if all_angle_steps is None:
            all_angle_steps = angle_steps

    df_all_calcs = pd.concat(all_calcs)

    cols_to_check = [col for col in df_all_calcs.columns
                         if re.match(r'(Euclid|Turning Angle) \d+', str(col))]

    for col in cols_to_check:
        if df_all_calcs[col].isna().all() or (df_all_calcs[col].fillna(0) == 0).all():
            df_all_calcs = df_all_calcs.drop(columns=col)

    mapping = {0: None}
    toc = tempo.time()

    with thread_lock:
        msg = ' Calculations done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
        messages[-1] += msg
        messages.append('')
    complete_progress_step("Calculations")

    category_input_list = []

    if categories_dataframe is None:
        unique_ids = np.unique(arr_segments[:, 0])
        cat_df = pd.DataFrame({
            parameters['object_id_2_col']: unique_ids,
            parameters['category_col']: '0'
        })
        cat_df.columns = ['Object ID', 'Category']
        cat_df['Category'] = cat_df['Category'].astype(str)
        arr_cats = np.array([[int(obj_id), '0'] for obj_id in unique_ids], dtype=object)
        categories_file_name = 'None'
    else:
        cat_df = categories_dataframe.copy()
        categories_file_name = categories_filename if categories_filename else 'None'

        cat_df = cat_df[[parameters['object_id_2_col'], parameters['category_col']]]
        for row in cat_df.index:
            object_id2 = int(cat_df[parameters['object_id_2_col']][row])
            category = cat_df[parameters['category_col']][row]
            category_input_list.append([object_id2, category])
        cat_df.columns = ['Object ID', 'Category']
        cat_df['Category'] = cat_df['Category'].astype(str)
        arr_cats = np.array(category_input_list)
        arr_cats[:, 0] = arr_cats[:, 0].astype(int)

    if dropped_objects:
        remaining_object_ids = set(unique_objects)
        arr_cats_filtered = []
        for row in arr_cats:
            if int(row[0]) in remaining_object_ids:
                arr_cats_filtered.append(row)
        arr_cats = np.array(arr_cats_filtered) if arr_cats_filtered else np.array([]).reshape(0, 2)

    settings = [
        ('Segments file', segments_file_display),
        ('Categories file', categories_file_name),
        ('Arrest limit', arrest_limit),
        ('Min. timepoints', parameters['moving']),
        ('Contact length', contact_length),
        ('Max. arrest coeff.', arrested),
        ('Min. Max. Euclidean', min_maxeuclid),
        ('Timelapse interval', timelapse_interval),
        ('Max. Tau', parameters['tau']),
        ('Multitracking', parameters['multi_track']),
        ('Interpolation', parameters['interpolate'])
    ]
    df_settings = pd.DataFrame(settings, columns=['Parameter', 'Value'])

    twodim_mode = False
    if arr_segments.shape[1] < 5 or np.all(arr_segments[:, 4] == 0):
        twodim_mode = True

    df_contacts_summary = None
    df_contacts_per_category = None

    (df_sum, df_single_euclid, df_single_angle, df_msd, df_msd_sum_all, df_msd_avg_per_cat, df_msd_std_per_cat,
     df_msd_loglogfits, df_pca, df_removed, euclidean_filtered_count) = summary_sheet(
        arr_segments, df_all_calcs, unique_objects, twodim_mode, parameters, arr_cats, savefile, all_angle_steps,
        all_angle_medians, df_removed
    )

    df_sum['Category'] = df_sum['Category'].astype(str)
    savepath = savefile + '_Results.xlsx'
    savecontacts = savefile + '_Contacts.xlsx'

    if parameters['contact']:
        with thread_lock:
            messages.append('Detecting contacts...')
        tic = tempo.time()
        df_sum_for_contacts = df_sum.copy()
        if parameters['arrest_limit'] == 0 and 'Arrest Coefficient' not in df_sum_for_contacts.columns:
            df_sum_for_contacts['Arrest Coefficient'] = 0.0
        unique_timepoints = np.unique(arr_segments[:, 1])
        df_contacts_all, df_contacts_nodividing, df_contacts_nodead = contacts_parallel.main(
            unique_timepoints,
            arr_segments,
            parameters['contact_length'],
            df_sum_for_contacts,
            parameters['arrested'],
            parameters['contact_div_filter'],
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

        if not df_contacts_summary.empty and 'Category' not in df_contacts_summary.columns:
            df_contacts_summary = df_contacts_summary.merge(
                df_sum_for_contacts[['Object ID', 'Category']],
                on='Object ID',
                how='left'
            )
            cols = ['Object ID', 'Category'] + [col for col in df_contacts_summary.columns if
                                                col not in ['Object ID', 'Category']]
            df_contacts_summary = df_contacts_summary[cols]
            df_contacts_summary['Category'] = df_contacts_summary['Category'].astype(str)

        all_objects = df_sum_for_contacts[['Object ID', 'Category']].copy()
        if df_contacts_summary.empty:
            df_contacting = all_objects.copy()
            df_contacting['Number of Contacts'] = 0
            df_contacting['Total Time Spent in Contact'] = 0
            df_contacting['Median Contact Duration'] = 0
        else:
            for col in ['Number of Contacts', 'Total Time Spent in Contact', 'Median Contact Duration']:
                if col not in df_contacts_summary.columns:
                    df_contacts_summary[col] = 0

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

        with pd.ExcelWriter(savecontacts, engine='xlsxwriter') as workbook:
            df_contacts_all.to_excel(workbook, sheet_name='Contacts (all)', index=False)
            if parameters['contact_div_filter']:
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
    else:
        pass

    if parameters['attractors']:
        tic = tempo.time()
        with thread_lock:
            messages.append('Detecting attractors...')
        cell_types = dict(zip(cat_df['Object ID'], cat_df['Category']))
        attract(unique_objects, arr_segments, cell_types, df_all_calcs, savefile, parameters['attract_params'])
        toc = tempo.time()
        with thread_lock:
            msg = ' Attractors done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
            messages[-1] += msg
            messages.append('')
        complete_progress_step('Attractors')
    else:
        pass

    if parameters['generate_figures']:
        with thread_lock:
            messages.append('Generating figures...')
        tic = tempo.time()
        df_sum_for_figs = df_sum.copy()
        save_all_figures(df_sum_for_figs, df_segments, df_pca, df_msd, df_msd_loglogfits,
                         df_contacts_summary, df_contacts_per_category, savefile,
                         parameters['infile_categories'], twodim_mode)
        toc = tempo.time()
        with thread_lock:
            msg = ' Done in {:.0f} seconds.'.format(int(round((toc - tic), 1)))
            messages[-1] += msg
            messages.append('')
        complete_progress_step('Generate Figures')
    else:
        pass

    with thread_lock:
        gaps_count = len(dropped_objects)
        euclidean_count = euclidean_filtered_count

        if gaps_count > 0 and euclidean_count > 0:
            messages.append(f"Removed {gaps_count} object(s) with timepoint gaps and {euclidean_count} object(s) with Max Euclidean < {min_maxeuclid}.")
            messages.append(f"Removed object ID(s) recorded in 'Removed Objects' sheet in the main results file.")
            messages.append('')
        elif gaps_count > 0:
            messages.append(f"Removed {gaps_count} object(s) with timepoint gaps.")
            messages.append(f"Removed object ID(s) recorded in 'Removed Objects' sheet in the main results file.")
            messages.append('')
        elif euclidean_count > 0:
            messages.append(f"Removed {euclidean_count} object(s) with Max Euclidean < {min_maxeuclid}.")
            messages.append(f"Removed object ID(s) recorded in 'Removed Objects' sheet in the main results file.")
            messages.append('')

        if parameters['verbose']:
            messages.append('Saving main output to ' + savepath + '...')
            messages.append('Please wait patiently. Save times are significantly longer when verbose mode is enabled...')
            messages.append('')
        else:
            messages.append('Saving main output to ' + savepath + '...')
            messages.append('')

    df_all_calcs = df_all_calcs.replace(mapping)
    cols_to_replace = [col for col in df_sum.columns if col != 'Category']
    df_sum[cols_to_replace] = df_sum[cols_to_replace].replace(mapping)
    if parameters['arrest_limit'] != 0:
        df_sum['Arrest Coefficient'] = df_sum.loc[:, 'Arrest Coefficient'].replace((np.nan, ' '), (0, 0))

    with pd.ExcelWriter(savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as workbook:
        df_settings.to_excel(workbook, sheet_name='Settings', index=False)
        if not df_removed.empty:
            df_removed.to_excel(workbook, sheet_name='Removed Objects', index=False)
        df_sum.to_excel(workbook, sheet_name='Summary Features', index=False)
        df_single_euclid.to_excel(workbook, sheet_name='Euclidean Medians', index=False)
        df_single_angle.to_excel(workbook, sheet_name='Turning Angles', index=False)
        df_msd.to_excel(workbook, sheet_name='Mean Squared Displacements', index=False)
        df_msd_sum_all.to_excel(workbook, sheet_name='MSD Summary', index=True)
        if 'Category' in df_msd.columns and df_msd['Category'].nunique() > 1:
            df_msd_avg_per_cat.to_excel(workbook, sheet_name='MSD Mean Per Category', index=True)
            df_msd_std_per_cat.to_excel(workbook, sheet_name='MSD StDev Per Category', index=True)
        df_msd_loglogfits.to_excel(workbook, sheet_name='MSD Log-Log Fits', index=True)

    if parameters['verbose']:
        calc_savepath = savefile + '_Calculations.xlsx'
        with pd.ExcelWriter(calc_savepath, engine='xlsxwriter', engine_kwargs={'options': {'zip64': True}}) as calc_workbook:
            df_segments.to_excel(calc_workbook, sheet_name='Object Data', index=False)
            df_all_calcs.to_excel(calc_workbook, sheet_name='Calculations', index=False)

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
