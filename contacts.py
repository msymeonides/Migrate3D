import numpy as np
import pandas as pd
import statistics


def align_time(time_base, time_comp):
    # Align time vectors to find matching timepoints
    base_index_advance = 0
    comp_index_advance = 0
    while base_index_advance < len(time_base) and comp_index_advance < len(time_comp):
        time_base_check = time_base[base_index_advance]
        time_comp_check = time_comp[comp_index_advance]
        if time_base_check < time_comp_check:
            base_index_advance += 1
        elif time_comp_check < time_base_check:
            comp_index_advance += 1
        else:
            break
    return base_index_advance, comp_index_advance


def contacts(unique_cells, arr_segments, contact_length):
    df_of_contacts = []
    base_done = []

    for cell_base in unique_cells:
        try:
            # Get timepoint and coordinates for base cell
            cell_data = arr_segments[arr_segments[:, 0] == cell_base, :]
            x_base = cell_data[:, 2]
            y_base = cell_data[:, 3]
            z_base = cell_data[:, 4]
            time_base = cell_data[:, 1]
            base_done.append(cell_base)

            for cell_comp in unique_cells:
                # Compare base cell to other cells
                if cell_comp == cell_base or cell_comp in base_done:
                    pass
                else:
                    # Get data for comparison cell
                    cell_data_comp = arr_segments[arr_segments[:, 0] == cell_comp, :]
                    x_comp = cell_data_comp[:, 2]
                    y_comp = cell_data_comp[:, 3]
                    z_comp = cell_data_comp[:, 4]
                    time_comp = cell_data_comp[:, 1]

                    # Align time vectors
                    base_index_advance, comp_index_advance = align_time(time_base, time_comp)

                    shortest_len = min(len(time_base), len(time_comp))

                    cell_base_list = []
                    cell_comp_list = []
                    time_of_contact = []

                    # Check for contacts and add contacts to DataFrame
                    for i in range(shortest_len):
                        time_b = time_base[i + base_index_advance]
                        x_diff = np.abs(x_base[i + base_index_advance] - x_comp[i + comp_index_advance])
                        if x_diff <= contact_length:
                            y_diff = np.abs(y_base[i + base_index_advance] - y_comp[i + comp_index_advance])
                            if y_diff <= contact_length:
                                z_diff = np.abs(z_base[i + base_index_advance] - z_comp[i + comp_index_advance])
                                if z_diff <= contact_length:
                                    cell_base_list.append(cell_base)
                                    cell_comp_list.append(cell_comp)
                                    time_of_contact.append(time_b)
                    df_c = pd.DataFrame({'Cell ID': cell_base_list,
                                         'Cell Compare': cell_comp_list,
                                         'Time of Contact': time_of_contact})
                    if df_c.empty:
                        pass
                    else:
                        df_of_contacts.append(df_c)

        except IndexError:
            pass

    # Repeated loop with unique_cells reversed to find contacts in reversed order
    for cell_base in reversed(unique_cells):
        try:
            cell_data = arr_segments[arr_segments[:, 0] == cell_base, :]
            x_base = cell_data[:, 2]
            y_base = cell_data[:, 3]
            z_base = cell_data[:, 4]
            time_base = cell_data[:, 1]
            base_done.append(cell_base)
            for cell_comp in unique_cells:
                if cell_comp == cell_base or cell_comp in base_done:
                    pass
                else:
                    cell_data_comp = arr_segments[arr_segments[:, 0] == cell_comp, :]
                    x_comp = cell_data_comp[:, 2]
                    y_comp = cell_data_comp[:, 3]
                    z_comp = cell_data_comp[:, 4]
                    time_comp = cell_data_comp[:, 1]

                    base_index_advance, comp_index_advance = align_time(time_base, time_comp)

                    shortest_len = min(len(time_base), len(time_comp))

                    cell_base_list = []
                    cell_comp_list = []
                    time_of_contact = []

                    for i in range(shortest_len):
                        time_b = time_base[i + base_index_advance]
                        x_diff = np.abs(x_base[i + base_index_advance] - x_comp[i + comp_index_advance])
                        if x_diff <= contact_length:
                            y_diff = np.abs(y_base[i + base_index_advance] - y_comp[i + comp_index_advance])
                            if y_diff <= contact_length:
                                z_diff = np.abs(z_base[i + base_index_advance] - z_comp[i + comp_index_advance])
                                if z_diff <= contact_length:
                                    cell_base_list.append(cell_base)
                                    cell_comp_list.append(cell_comp)
                                    time_of_contact.append(time_b)
                    df_c = pd.DataFrame({'Cell ID': cell_base_list,
                                         'Cell Compare': cell_comp_list,
                                         'Time of Contact': time_of_contact})
                    if df_c.empty:
                        pass
                    else:
                        df_of_contacts.append(df_c)

        except IndexError:
            pass

    return df_of_contacts


def no_daughter_contacts(cell_id, df):
    # Remove contacts with potential daughter cells
    list_of_df = []
    for cell_base in cell_id:
        cell_comp_list = list(df.loc[df['Cell ID'] == cell_base, 'Cell Compare'])
        time_ = list(df.loc[df['Cell ID'] == cell_base, 'Time of Contact'])
        updated_cell_comp = []
        for index_, cell_comp in enumerate(cell_comp_list):
            # Check for adjacent cells
            if np.abs(cell_comp - cell_base) == 1 and cell_base != cell_comp:
                updated_cell_comp.append(0)
            else:
                updated_cell_comp.append(cell_comp)

        # Create updated DataFrame with adjacent cells removed
        df_no_daughters = pd.DataFrame({'Cell ID': cell_base,
                                        'Cell Compare': updated_cell_comp,
                                        'Time of Contact': time_})
        if df_no_daughters.empty:
            pass
        else:
            df_no_daughters = df_no_daughters.replace(0, None)
            df_no_daughters = df_no_daughters.dropna()
            list_of_df.append(df_no_daughters)

    return list_of_df


def contacts_alive(df_arrest, df_no_mitosis, arrested, time_interval):
    # Filter out dead cells from contacts based on arrest coefficient
    cells_in_arrest = list(df_arrest.loc[:, 'Cell ID'])
    all_alive = []
    list_of_df_no_dead = []
    list_of_summary_df = []
    for cells in cells_in_arrest:
        # Check if cells are alive
        arrest_coeffs = float(df_arrest.loc[df_arrest['Cell ID'] == cells, 'Arrest Coefficient'].iloc[0])
        if arrest_coeffs < arrested:
            all_alive.append(cells)
        else:
            pass

    # Extract data for alive cells
    for ind, cell_a in enumerate(all_alive):
        cell_comp = list(df_no_mitosis.loc[df_no_mitosis['Cell ID'] == cell_a, 'Cell Compare'])
        time__ = list(df_no_mitosis.loc[df_no_mitosis['Cell ID'] == cell_a, 'Time of Contact'])
        only_1_comp = []
        for cell in cell_comp:
            if cell in only_1_comp:
                pass
            else:
                only_1_comp.append(cell)

        list_of_df_no_dead.append(pd.DataFrame({'Cell ID': cell_a,
                                                'Cell Compare': cell_comp,
                                                'Time of Contact': time__}))

        # Calculate median contact time
        if len(time__) > 2:
            time_actual = [x * time_interval for x in range(len(time__))]
            med_time = statistics.median(time_actual)
        else:
            med_time = None

        sum_df = pd.DataFrame({'Cell ID': [cell_a],
                               'Number of Contacts': [len(only_1_comp)],
                               'Total Time Spent in Contact': [len(time__) * time_interval],
                               'Median Contact Duration': [med_time]})
        if sum_df.empty:
            pass
        else:
            list_of_summary_df.append(sum_df)

    return list_of_df_no_dead, list_of_summary_df