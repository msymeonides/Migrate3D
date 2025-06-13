import plotly.graph_objects as go
import pandas as pd

colors = ['Black', 'Blue', 'Red', 'Purple', 'Orange', 'Green', 'Pink', 'Navy', 'Grey', 'Cyan',
          'darkgray', 'aqua', 'crimson', 'darkviolet', 'orangered', 'darkolivegreen', 'darksalmon', 'Blue', 'Black',
          'lightseagreen']

def get_category_color_map(categories):
    categories = [int(cat) for cat in categories if pd.notnull(cat)]
    categories = sorted(set(categories))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

def summary_figures(df):
    columns = list(df.columns)
    categories = list(df.loc[:, 'Category'])
    unique_cat = []
    [unique_cat.append(x) for x in categories if x not in unique_cat]
    unique_cat.sort()
    color_map = get_category_color_map(unique_cat)
    all_figures = []
    for i, col in enumerate(columns):
        if i == 0 or col == 'Category':
            pass
        else:
            df_subset = df[['Object ID', col, 'Category']]
            fig_traces = []
            for cat in unique_cat:
                df_cat_subset = df_subset.loc[df_subset['Category'] == cat]
                fig_traces.append(go.Violin(
                    x=list(df_cat_subset.loc[:, 'Category']),
                    y=list(df_cat_subset.loc[:, col]),
                    marker_color=color_map.get(cat, colors[0]),
                    showlegend=False
                ))
            fig_ = go.Figure(data=fig_traces)
            fig_.update_layout(title=f'Violin plot of {col}', plot_bgcolor='white')
            all_figures.append(fig_)
    return all_figures

def tracks_figure(df, df_sum, cat_provided, save_file):
    all_ids = list(df.loc[:, 'Object ID'])
    unique_ids = []
    [unique_ids.append(x) for x in all_ids if x not in unique_ids]
    traces_ = []

    if cat_provided:
        categories = df_sum['Category'].unique()
        color_map = get_category_color_map(categories)
    else:
        color_map = {0: colors[0]}

    for object_ in unique_ids:
        df_object = df.loc[df['Object ID'] == object_]
        if cat_provided:
            cat = int(df_sum.loc[df_sum['Object ID'] == object_, 'Category'].iloc[0])
        else:
            cat = 0
        time_data = list(df_object.loc[:, 'Time'])
        time_data = [f'Time point {x} Category {cat}' for x in time_data]
        x_data = list(df_object.iloc[:, 2])
        y_data = list(df_object.iloc[:, 3])
        z_data = list(df_object.iloc[:, 4])
        traces_.append(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, hovertext=time_data, name=object_, mode='lines',
            marker=dict(size=12), marker_color=color_map.get(cat, colors[0])
        ))
    tracks_fig = go.Figure(traces_)
    tracks_fig.update_layout(title=f'{save_file} Tracks', plot_bgcolor='white')
    return tracks_fig

def pca_figure(df):
    categories = df['Category'].unique()
    color_map = get_category_color_map(categories)
    all_objects = list(df.loc[:, 'Object ID'])
    pcscores = []
    for i in all_objects:
        pc1 = float(df.loc[df['Object ID'] == i, 'PC1'].iloc[0])
        pc2 = float(df.loc[df['Object ID'] == i, 'PC2'].iloc[0])
        pc3 = float(df.loc[df['Object ID'] == i, 'PC3'].iloc[0])
        cat = int(df.loc[df['Object ID'] == i, 'Category'].iloc[0])
        pcscores.append(go.Scatter3d(
            x=[pc1], y=[pc2], z=[pc3], name=i,
            marker_color=color_map.get(cat, colors[0]),
            hovertext=f'Category {cat}'
        ))
    pca_fig = go.Figure(data=pcscores)
    pca_fig.update_layout(title='PCA', plot_bgcolor='white')
    return pca_fig