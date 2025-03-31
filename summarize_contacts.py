import pandas as pd


def summarize_contacts(df_contacts_final, time_interval):
    summary_list = []
    for object_id, group in df_contacts_final.groupby("Object ID"):
        unique_contacts = group["Object Compare"].unique()
        num_contacts = len(unique_contacts)
        total_time = len(group) * time_interval
        n = len(group)
        # Use 1-indexed durations to represent duration of each contact.
        durations = [(i + 1) * time_interval for i in range(n)]
        if n == 1:
            med_time = durations[0]
        elif n == 2:
            med_time = sum(durations) / 2
        else:
            import statistics
            med_time = statistics.median(durations)

        summary_list.append({
            "Object ID": object_id,
            "Number of Contacts": num_contacts,
            "Total Time Spent in Contact": total_time,
            "Median Contact Duration": med_time
        })
    return pd.DataFrame(summary_list)
