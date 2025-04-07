import pandas as pd
import plotly.graph_objects as go


def generate_PCA(df):
    """
    Takes in PCA Scores sheet in PCA output and then returns a 3D scatter
    :param df:
    :return: 3D scatter plot of PCA
    """

    all_colors = ['Black', 'Blue', 'Red', 'Purple', 'Orange', 'Green', 'Pink', 'Navy', 'Grey', 'Cyan',
              'darkgray', 'aqua', 'crimson', 'darkviolet', 'orangered', 'darkolivegreen', 'darksalmon', 'Blue', 'Black',
              'lightseagreen']

    # get all rows
    all_objects = list(df.loc[:, 'Object ID'])

    # begin populating scatter plot
    traces_ = []
    for i in all_objects:
        pc1 = float(df.loc[df['Object ID'] == i, 'PC1'])
        pc2 = float(df.loc[df['Object ID'] == i, 'PC2'])
        pc3 = float(df.loc[df['Object ID'] == i, 'PC3'])
        cat = int(df.loc[df['Object ID'] == i, 'Category'])
        traces_.append(go.Scatter3d(x=[pc1], y=[pc2], z=[pc3], name=i, marker_color=all_colors[cat],
                                    hovertext=f'Category {cat}'))
    fig = go.Figure(data=traces_)
    fig.update_layout(title='PCA', plot_bgcolor='white')

    return fig



