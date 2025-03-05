import numpy as np
import pandas as pd


def align_time(time_base, time_comp):
    """
    Aligns two time vectors to find matching timepoints.
    Args:
        time_base (numpy.ndarray): Time vector for the base object.
        time_comp (numpy.ndarray): Time vector for the comparison object.
    Returns:
        tuple: Indices to advance in the base and comparison time vectors to align them.
    """
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


def contacts(unique_objects, arr_segments, contact_length):
    """
        Detects contacts between objects based on their coordinates and a specified contact length.
        Args:
            unique_objects (numpy.ndarray): Array of unique object IDs.
            arr_segments (numpy.ndarray): Array of segments with columns [object_id, timepoint, x, y, z].
            contact_length (float): The maximum distance between objects to be considered in contact.
        Returns:
            list: A list of DataFrames, each containing the contacts detected for a pair of objects.
        """
    df_of_contacts = []
    base_done = []

    for object_base in unique_objects:
        try:
            # Get timepoint and coordinates for base object
            object_data = arr_segments[arr_segments[:, 0] == object_base, :]
            x_base = object_data[:, 2]
            y_base = object_data[:, 3]
            z_base = object_data[:, 4]
            time_base = object_data[:, 1]
            base_done.append(object_base)

            for object_comp in unique_objects:
                # Compare base object to other objects
                if object_comp == object_base or object_comp in base_done:
                    pass
                else:
                    # Get data for comparison object
                    object_data_comp = arr_segments[arr_segments[:, 0] == object_comp, :]
                    x_comp = object_data_comp[:, 2]
                    y_comp = object_data_comp[:, 3]
                    z_comp = object_data_comp[:, 4]
                    time_comp = object_data_comp[:, 1]

                    # Align time vectors
                    base_index_advance, comp_index_advance = align_time(time_base, time_comp)

                    shortest_len = min(len(time_base), len(time_comp))

                    object_base_list = []
                    object_comp_list = []
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
                                    object_base_list.append(object_base)
                                    object_comp_list.append(object_comp)
                                    time_of_contact.append(time_b)
                    df_c = pd.DataFrame({'Object ID': object_base_list,
                                         'Object Compare': object_comp_list,
                                         'Time of Contact': time_of_contact})
                    if df_c.empty:
                        pass
                    else:
                        df_of_contacts.append(df_c)

        except IndexError:
            pass

    # Repeated loop with unique_objects reversed to find contacts in reversed order
    for object_base in reversed(unique_objects):
        try:
            object_data = arr_segments[arr_segments[:, 0] == object_base, :]
            x_base = object_data[:, 2]
            y_base = object_data[:, 3]
            z_base = object_data[:, 4]
            time_base = object_data[:, 1]
            base_done.append(object_base)
            for object_comp in unique_objects:
                if object_comp == object_base or object_comp in base_done:
                    pass
                else:
                    object_data_comp = arr_segments[arr_segments[:, 0] == object_comp, :]
                    x_comp = object_data_comp[:, 2]
                    y_comp = object_data_comp[:, 3]
                    z_comp = object_data_comp[:, 4]
                    time_comp = object_data_comp[:, 1]

                    base_index_advance, comp_index_advance = align_time(time_base, time_comp)

                    shortest_len = min(len(time_base), len(time_comp))

                    object_base_list = []
                    object_comp_list = []
                    time_of_contact = []

                    for i in range(shortest_len):
                        time_b = time_base[i + base_index_advance]
                        x_diff = np.abs(x_base[i + base_index_advance] - x_comp[i + comp_index_advance])
                        if x_diff <= contact_length:
                            y_diff = np.abs(y_base[i + base_index_advance] - y_comp[i + comp_index_advance])
                            if y_diff <= contact_length:
                                z_diff = np.abs(z_base[i + base_index_advance] - z_comp[i + comp_index_advance])
                                if z_diff <= contact_length:
                                    object_base_list.append(object_base)
                                    object_comp_list.append(object_comp)
                                    time_of_contact.append(time_b)
                    df_c = pd.DataFrame({'Object ID': object_base_list,
                                         'Object Compare': object_comp_list,
                                         'Time of Contact': time_of_contact})
                    if df_c.empty:
                        pass
                    else:
                        df_of_contacts.append(df_c)

        except IndexError:
            pass

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
            # Check for adjacent objects
            if np.abs(object_comp - object_base) == 1 and object_base != object_comp:
                updated_object_comp.append(0)
            else:
                updated_object_comp.append(object_comp)

        # Create updated DataFrame with adjacent objects removed
        df_no_daughters = pd.DataFrame({'Object ID': object_base,
                                        'Object Compare': updated_object_comp,
                                        'Time of Contact': time_})
        if df_no_daughters.empty:
            pass
        else:
            df_no_daughters = df_no_daughters.replace(0, None)
            df_no_daughters = df_no_daughters.dropna()
            list_of_df.append(df_no_daughters)

    return list_of_df


# language: python
def contacts_moving(df_arrest, df_no_daughter, arrested, time_interval):
    """
    Filters out non-moving objects (e.g. dead cells) from the contacts data based on the arrest coefficient.
    This function no longer performs duration or summary calculations.

    Args:
        df_arrest (pandas.DataFrame): DataFrame containing arrest coefficients for objects.
        df_no_daughter (pandas.DataFrame): DataFrame containing contact information with daughter objects removed.
        arrested (float): Threshold for the arrest coefficient to consider an object as moving.
        time_interval (float): Time interval between timepoints (retained for potential downstream use).

    Returns:
        list: A list of DataFrames with contacts for moving objects.
    """
    objects_in_arrest = list(df_arrest.loc[:, "Object ID"])
    all_moving = []
    list_of_df_no_dead = []

    # Determine moving objects based on the arrest coefficient.
    for obj in objects_in_arrest:
        arrest_coeff = float(df_arrest.loc[df_arrest["Object ID"] == obj, "Arrest Coefficient"].iloc[0])
        if arrest_coeff < arrested:
            all_moving.append(obj)

    # Extract contact data for each moving object.
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