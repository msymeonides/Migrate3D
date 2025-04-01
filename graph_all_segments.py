import pandas as pd
import plotly.graph_objects as go

all_colors = ['Blue', 'Red', 'Black', 'Orange', 'Purple', 'Green', 'Pink', 'Brown', 'Gray', 'teal', 'Navy']
def graph_sorted_segments(df, df_sum, cat_provided):
    """
    Takes in sorted segments from Object Data sheet in results, then graphs the 3D tracks
    :param df: pandas dataframe
    :return: 3D scatter plot
    """

    # get all unique object ID
    all_ids = list(df.loc[:, 'Object ID'])
    unique_ids = []
    [unique_ids.append(x) for x in all_ids if x not in unique_ids]
    print(unique_ids)

    # iterate through and add to scatter plot
    traces_ = []
    for object_ in unique_ids:
        df_object = df.loc[df['Object ID'] == object_]
        if cat_provided:
            cat = int(df_sum.loc[df_sum['Object ID'] == object_, 'Category'])
        else:
            cat = 0
        time_data = list(df_object.loc[:, 'Time'])
        time_data = [f'Time point {x} Category {cat}' for x in time_data]
        x_data = list(df_object.loc[:, 'X'])
        y_data = list(df_object.loc[:, 'Y'])
        z_data = list(df_object.loc[:, 'Z Coordinate'])
        traces_.append(go.Scatter3d(x=x_data, y=y_data, z=z_data, hovertext=time_data, name=object_, mode='lines',
                                    marker=dict(size=12), marker_color=all_colors[cat]))

    fig = go.Figure(traces_)
    return fig
