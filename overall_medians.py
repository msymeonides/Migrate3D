import statistics

def overall_medians(object, df_all_calcs, cols_angles, cols_euclidean):
    list_of_angle_medians = []
    list_of_euclidean_medians = []
    single_euclidean = []
    single_angle = []

    for col_ in cols_angles:
        angle_median = list(df_all_calcs.loc[df_all_calcs['Object ID'] == object, col_])
        angle_median = [x for x in angle_median if x is not None and x != 0]
        if len(angle_median) > 2:
            angle_median = statistics.median(angle_median)
            if angle_median == 0:
                pass
            else:
                list_of_angle_medians.append(angle_median)
                single_angle.append(angle_median)
        else:
            pass

    for cols_ in cols_euclidean:
        euclidean_median = df_all_calcs.loc[df_all_calcs['Object ID'] == object, cols_]
        euclidean_median = [x for x in euclidean_median if x is not None and x != 0]
        if len(euclidean_median) > 2:
            euclidean_median = statistics.median(euclidean_median)
            list_of_euclidean_medians.append(euclidean_median)
            single_euclidean.append(euclidean_median)
        else:
            pass
    if len(list_of_euclidean_medians) >= 1:
        overall_euclidean_median = statistics.median(list_of_euclidean_medians)
    else:
        overall_euclidean_median = None
    if len(list_of_angle_medians) >= 1:
        overall_angle_median = statistics.median(list_of_angle_medians)
    else:
        overall_angle_median = None

    return overall_euclidean_median, overall_angle_median, single_euclidean, single_angle