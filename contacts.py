import numpy as np
import pandas as pd

def align_time(time_base, time_comp):
    common_times = np.intersect1d(time_base, time_comp)
    base_indices = np.nonzero(np.in1d(time_base, common_times))[0]
    comp_indices = np.nonzero(np.in1d(time_comp, common_times))[0]
    return common_times, base_indices, comp_indices

def contacts(unique_objects, arr_segments, contact_length):
    df_of_contacts = []
    processed = set()

    for obj_base in unique_objects:
        try:
            data_base = arr_segments[arr_segments[:, 0] == obj_base]
            order = np.argsort(data_base[:, 1])
            data_base = data_base[order]
            x_base = data_base[:, 2]
            y_base = data_base[:, 3]
            z_base = data_base[:, 4]
            time_base = data_base[:, 1]
            processed.add(obj_base)

            for obj_comp in unique_objects:
                if obj_comp == obj_base or obj_comp in processed:
                    continue
                data_comp = arr_segments[arr_segments[:, 0] == obj_comp]
                order_comp = np.argsort(data_comp[:, 1])
                data_comp = data_comp[order_comp]
                x_comp = data_comp[:, 2]
                y_comp = data_comp[:, 3]
                z_comp = data_comp[:, 4]
                time_comp = data_comp[:, 1]

                # Find common timepoints and corresponding indices
                common_times, base_idx, comp_idx = align_time(time_base, time_comp)
                object_base_list = []
                object_comp_list = []
                time_of_contact = []

                for i, t in enumerate(common_times):
                    if (abs(x_base[base_idx[i]] - x_comp[comp_idx[i]]) <= contact_length and
                        abs(y_base[base_idx[i]] - y_comp[comp_idx[i]]) <= contact_length and
                        abs(z_base[base_idx[i]] - z_comp[comp_idx[i]]) <= contact_length):
                        object_base_list.append(obj_base)
                        object_comp_list.append(obj_comp)
                        time_of_contact.append(t)

                if object_base_list:
                    df_c = pd.DataFrame({
                        "Object ID": object_base_list,
                        "Object Compare": object_comp_list,
                        "Time of Contact": time_of_contact
                    })
                    df_of_contacts.append(df_c)

        except Exception:
            continue

    return df_of_contacts

def contacts_notdividing(object_id, df):
    list_of_df = []
    for object_base in object_id:
        object_comp_list = list(df.loc[df['Object ID'] == object_base, 'Object Compare'])
        time_ = list(df.loc[df['Object ID'] == object_base, 'Time of Contact'])
        updated_object_comp = []
        for index_, object_comp in enumerate(object_comp_list):
            if np.abs(object_comp - object_base) == 1 and object_base != object_comp:
                updated_object_comp.append(0)
            else:
                updated_object_comp.append(object_comp)

        df_no_dividing = pd.DataFrame({'Object ID': object_base,
                                        'Object Compare': updated_object_comp,
                                        'Time of Contact': time_})
        if not df_no_dividing.empty:
            df_no_dividing = df_no_dividing.replace(0, None)
            df_no_dividing = df_no_dividing.dropna()
            list_of_df.append(df_no_dividing)

    return list_of_df

def contacts_notdead(df_arrest, df_no_div, arrested):
    objects_in_arrest = list(df_arrest.loc[:, "Object ID"])
    all_moving = []
    list_of_df_no_dead = []

    for obj in objects_in_arrest:
        arrest_coeff = float(df_arrest.loc[df_arrest["Object ID"] == obj, "Arrest Coefficient"].iloc[0])
        if arrest_coeff < arrested:
            all_moving.append(obj)

    for obj in all_moving:
        object_comp = list(df_no_div.loc[df_no_div["Object ID"] == obj, "Object Compare"])
        time_points = list(df_no_div.loc[df_no_div["Object ID"] == obj, "Time of Contact"])
        list_of_df_no_dead.append(
            pd.DataFrame({
                "Object ID": obj,
                "Object Compare": object_comp,
                "Time of Contact": time_points
            })
        )

    return list_of_df_no_dead
