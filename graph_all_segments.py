import pandas as pd
import plotly.graph_objects as go


def graph_sorted_segments(df, df_sum, cat_provided, save_file):
    """
    Takes in sorted segments from Object Data sheet in results, then graphs the 3D tracks
    :param df: pandas dataframe
    :return: 3D scatter plot
    """
    all_colors =['Black', 'Blue', 'Red', 'Purple', 'Orange', 'Green', 'Pink', 'Navy', 'Grey', 'Cyan',
              'darkgray', 'aqua', 'crimson', 'darkviolet', 'orangered', 'darkolivegreen', 'darksalmon', 'Blue', 'Black',
              'lightseagreen']
    # get all unique object ID
    all_ids = list(df.loc[:, 'Object ID'])
    unique_ids = []
    [unique_ids.append(x) for x in all_ids if x not in unique_ids]

    # iterate through and add to scatter plot
    traces_ = []
    for object_ in unique_ids:
        df_object = df.loc[df['Object ID'] == object_]
        if cat_provided:
            cat = int(df_sum.loc[df_sum['Object ID'] == object_, 'Category'].iloc[0])
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
    fig.update_layout(title=f'{save_file} Tracks',
                      plot_bgcolor='white')
    return fig
