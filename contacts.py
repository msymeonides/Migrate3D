import numpy as np
import pandas as pd


def align_time(time_base, time_comp):
    common_times = np.intersect1d(time_base, time_comp)
    base_indices = np.nonzero(np.isin(time_base, common_times))[0]
    comp_indices = np.nonzero(np.isin(time_comp, common_times))[0]
    return common_times, base_indices, comp_indices


def contacts(unique_objects, arr_segments, contact_length):
    df_of_contacts = []
    processed = set()

    for obj_base in unique_objects:
        data_base = arr_segments[arr_segments[:, 0] == obj_base]
        if data_base.size == 0:
            continue
        order = np.argsort(data_base[:, 1])
        data_base = data_base[order]
        x_base, y_base, z_base, time_base = data_base[:, 2], data_base[:, 3], data_base[:, 4], data_base[:, 1]
        processed.add(obj_base)

        for obj_comp in unique_objects:
            if obj_comp == obj_base or obj_comp in processed:
                continue
            data_comp = arr_segments[arr_segments[:, 0] == obj_comp]
            if data_comp.size == 0:
                continue
            order_comp = np.argsort(data_comp[:, 1])
            data_comp = data_comp[order_comp]
            x_comp, y_comp, z_comp, time_comp = data_comp[:, 2], data_comp[:, 3], data_comp[:, 4], data_comp[:, 1]

            common_times, base_idx, comp_idx = align_time(time_base, time_comp)
            if len(common_times) == 0:
                continue

            in_contact = (
                (np.abs(x_base[base_idx] - x_comp[comp_idx]) <= contact_length) &
                (np.abs(y_base[base_idx] - y_comp[comp_idx]) <= contact_length) &
                (np.abs(z_base[base_idx] - z_comp[comp_idx]) <= contact_length)
            )
            if np.any(in_contact):
                df_c = pd.DataFrame({
                    "Object ID": obj_base,
                    "Object Compare": obj_comp,
                    "Time of Contact": common_times[in_contact]
                })
                df_of_contacts.append(df_c)
    return df_of_contacts


def contacts_notdividing(object_id, df):
    list_of_df = []
    for object_base in object_id:
        mask = df['Object ID'] == object_base
        object_comp_list = df.loc[mask, 'Object Compare'].to_numpy()
        time_list = df.loc[mask, 'Time of Contact'].to_numpy()
        keep = ~((np.abs(object_comp_list - object_base) == 1) & (object_comp_list != object_base))
        if np.any(keep):
            df_no_dividing = pd.DataFrame({
                'Object ID': object_base,
                'Object Compare': object_comp_list[keep],
                'Time of Contact': time_list[keep]
            })
            list_of_df.append(df_no_dividing)
    return list_of_df


def contacts_notdead(df_arrest, df_no_div, arrested):
    moving_mask = df_arrest["Arrest Coefficient"] < arrested
    moving_objects = df_arrest.loc[moving_mask, "Object ID"].to_numpy()
    filtered = df_no_div[df_no_div["Object ID"].isin(moving_objects)]
    if filtered.empty:
        return []
    grouped = filtered.groupby("Object ID")
    return [group for _, group in grouped]
