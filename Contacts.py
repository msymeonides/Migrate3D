import numpy as np
import pandas as pd
import statistics


def contacts(cell_id, df, parent_id, x, y, z, time_, contact_length):
    df_of_contacts = []
    base_done = []
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