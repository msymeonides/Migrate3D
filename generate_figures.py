import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


colors = ['Black', 'Blue', 'Red', 'Purple', 'Orange', 'Green', 'Pink', 'Navy', 'Grey', 'Cyan',
          'darkgray', 'aqua', 'crimson', 'darkviolet', 'orangered', 'darkolivegreen', 'darksalmon', 'Blue', 'Black',
          'lightseagreen']

def get_category_color_map(categories):
    categories = [int(cat) for cat in categories if pd.notnull(cat)]
    categories = sorted(set(categories))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

def summary_figures(df, color_map=None):
    columns = list(df.columns)
    categories = list(df.loc[:, 'Category'])
    unique_cat = []
    [unique_cat.append(x) for x in categories if x not in unique_cat]
    unique_cat.sort()
    if color_map is None:
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

def tracks_figure(df, df_sum, cat_provided, save_file, color_map=None):
    all_ids = list(df.loc[:, 'Object ID'])
    unique_ids = []
    [unique_ids.append(x) for x in all_ids if x not in unique_ids]
    traces_ = []

    if cat_provided:
        if color_map is None:
            categories = df_sum['Category'].unique()
            color_map = get_category_color_map(categories)
    else:
        color_map = {0: colors[0]}

    for object_ in unique_ids:
        cat_row = df_sum.loc[df_sum['Object ID'] == object_, 'Category']
        if cat_row.empty:
            continue
        cat = int(cat_row.iloc[0])
        df_object = df.loc[df['Object ID'] == object_]
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

def pca_figure(df, color_map=None):
    categories = sorted(df['Category'].dropna().unique(), key=lambda x: str(x))
    if color_map is None:
        color_map = get_category_color_map(categories)
    traces = []
    for cat in categories:
        df_cat = df[df['Category'] == cat]
        traces.append(go.Scatter3d(
            x=df_cat['PC1'],
            y=df_cat['PC2'],
            z=df_cat['PC3'],
            mode='markers',
            marker=dict(color=color_map.get(cat, colors[0])),
            name=f'Category {cat}',
            showlegend=True,
            legendgroup=str(cat)
        ))
    pca_fig = go.Figure(data=traces)
    pca_fig.update_layout(title='PCA', plot_bgcolor='white')
    return pca_fig

def save_all_figures(df_sum, df_segments, df_pca, savefile, cat_provided, thread_lock=None, messages=None):
    all_figures = summary_figures(df_sum, color_map=get_category_color_map(df_sum['Category'].unique()))
    color_map = get_category_color_map(df_sum['Category'].unique())
    tracks_fig = tracks_figure(df_segments, df_sum, cat_provided, savefile, color_map=color_map)
    pca_fig = None
    if df_pca is not None and not df_pca.empty:
        pca_fig = pca_figure(df_pca, color_map=color_map)
        tracks_html = tracks_fig.to_html(full_html=True, include_plotlyjs='cdn')
        pca_html = pca_fig.to_html(full_html=True, include_plotlyjs='cdn')
        with open(f'{savefile}_Tracks Figure.html', 'w') as f:
            f.write(tracks_html)
        with open(f'{savefile}_PCA Figure.html', 'w') as f:
            f.write(pca_html)
    else:
        if thread_lock and messages is not None:
            with thread_lock:
                messages.append("No valid PCA data found for figure.")

    with open(f'{savefile}_Summary Figures.html', 'w') as f:
        for fig in all_figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    if thread_lock and messages is not None:
        with thread_lock:
            msg = "Figures generated."
            messages.append(msg)
            messages.append('')
