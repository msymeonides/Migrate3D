import numpy as np
import pandas as pd


def align_time(time_base, time_comp):
    """
    Computes the intersection of time points so that only exactly matching timepoints are compared.
    Returns:
        common_times: numpy.ndarray of timepoints common to both vectors.
        base_indices: indices in time_base corresponding to the common times.
        comp_indices: indices in time_comp corresponding to the common times.
    """
    common_times = np.intersect1d(time_base, time_comp)
    base_indices = np.nonzero(np.in1d(time_base, common_times))[0]
    comp_indices = np.nonzero(np.in1d(time_comp, common_times))[0]
    return common_times, base_indices, comp_indices

def contacts(unique_objects, arr_segments, contact_length):
    """
    Detects contacts between objects by comparing their coordinates at exactly matching timepoints.
    """
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
            # It is advisable to catch specific exceptions or log errors instead of a blanket exception.
            continue

    return df_of_contacts


def no_daughter_contacts(object_id, df):
    """
    Removes contacts with potential daughter objects, e.g., daughter cells after mitosis.
    Args:
        object_id (numpy.ndarray): Array of unique object IDs.
        df (pandas.DataFrame): DataFrame containing contact information.
    Returns:
        list: A list of DataFrames with contacts involving daughter objects removed.
    """
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

        df_no_daughters = pd.DataFrame({'Object ID': object_base,
                                        'Object Compare': updated_object_comp,
                                        'Time of Contact': time_})
        if not df_no_daughters.empty:
            df_no_daughters = df_no_daughters.replace(0, None)
            df_no_daughters = df_no_daughters.dropna()
            list_of_df.append(df_no_daughters)

    return list_of_df


def contacts_moving(df_arrest, df_no_daughter, arrested):
    """
    Filters out non-moving objects (e.g. dead cells) from the contacts data based on the arrest coefficient.
    This function no longer performs duration or summary calculations.
    Args:
        df_arrest (pandas.DataFrame): DataFrame containing arrest coefficients for objects.
        df_no_daughter (pandas.DataFrame): DataFrame containing contact information with daughter objects removed.
        arrested (float): Threshold for the arrest coefficient to consider an object as moving.
    Returns:
        list: A list of DataFrames with contacts for moving objects.
    """
    objects_in_arrest = list(df_arrest.loc[:, "Object ID"])
    all_moving = []
    list_of_df_no_dead = []

    for obj in objects_in_arrest:
        arrest_coeff = float(df_arrest.loc[df_arrest["Object ID"] == obj, "Arrest Coefficient"].iloc[0])
        if arrest_coeff < arrested:
            all_moving.append(obj)

    for obj in all_moving:
        object_comp = list(df_no_daughter.loc[df_no_daughter["Object ID"] == obj, "Object Compare"])
        time_points = list(df_no_daughter.loc[df_no_daughter["Object ID"] == obj, "Time of Contact"])
        list_of_df_no_dead.append(
            pd.DataFrame({
                "Object ID": obj,
                "Object Compare": object_comp,
                "Time of Contact": time_points
            })
        )

    return list_of_df_no_dead
