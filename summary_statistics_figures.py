import pandas as pd
import plotly.graph_objects as go
import numpy as np


def generate_figures(df):
    colors = ['Black', 'Blue', 'Red', 'Purple', 'Orange', 'Green', 'Pink', 'Navy', 'Grey', 'Cyan',
              'darkgray', 'aqua', 'crimson', 'darkviolet', 'orangered', 'darkolivegreen', 'darksalmon', 'Blue', 'Black',
              'lightseagreen']
    # generates all figures based on summary statistics
    columns = list(df.columns)
    all_ids = list(df.loc[:, 'Object ID'])
    categories = list(df.loc[:, 'Category'])
    unique_cat = []
    [unique_cat.append(x) for x in categories if x not in unique_cat]
    unique_cat.sort()
    # generate figures by iterating through all columns standard violin... can be changed
    all_figures = []
    for i, col in enumerate(columns):
        if i == 0 or col == 'Category':
            pass
        else:
            df_subset = df[['Object ID', col, 'Category']]
            fig_traces = []
            for cat in unique_cat:
                df_cat_subset = df_subset.loc[df_subset['Category'] == cat]

                fig_traces.append(go.Violin(x=list(df_cat_subset.loc[:, 'Category']), y=list(df_cat_subset.loc[:, col]),
                                            marker_color=colors[cat], showlegend=False))

            fig_ = go.Figure(data=fig_traces)
            fig_.update_layout(title=f'Violin plot of {col}', plot_bgcolor='white')
            all_figures.append(fig_)

    return all_figures


