import pandas as pd
import plotly.graph_objects as go
from msd_graphs import run_msd_graphs

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

def pca123_figure(df, color_map=None):
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
    pca123_fig = go.Figure(data=traces)
    pca123_fig.update_layout(
        title='PCA',
        plot_bgcolor='white',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )
    return pca123_fig

def pca124_figure(df, color_map=None):
    categories = sorted(df['Category'].dropna().unique(), key=lambda x: str(x))
    if color_map is None:
        color_map = get_category_color_map(categories)
    traces = []
    for cat in categories:
        df_cat = df[df['Category'] == cat]
        traces.append(go.Scatter3d(
            x=df_cat['PC1'],
            y=df_cat['PC2'],
            z=df_cat['PC4'],
            mode='markers',
            marker=dict(color=color_map.get(cat, colors[0])),
            name=f'Category {cat}',
            showlegend=True,
            legendgroup=str(cat)
        ))
    pca124_fig = go.Figure(data=traces)
    pca124_fig.update_layout(
        title='PCA (PC1, PC2, PC4)',
        plot_bgcolor='white',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC4'
        )
    )
    return pca124_fig

def save_all_figures(df_sum, df_segments, df_pca, df_msd, savefile, cat_provided):
    all_figures = summary_figures(df_sum, color_map=get_category_color_map(df_sum['Category'].unique()))
    color_map = get_category_color_map(df_sum['Category'].unique())
    tracks_fig = tracks_figure(df_segments, df_sum, cat_provided, savefile, color_map=color_map)
    if df_pca is not None and not df_pca.empty:
        pca123_fig = pca123_figure(df_pca, color_map=color_map)
        pca124_fig = pca124_figure(df_pca, color_map=color_map)
        tracks_html = tracks_fig.to_html(full_html=True, include_plotlyjs='cdn')
        pca123_html = pca123_fig.to_html(full_html=True, include_plotlyjs='cdn')
        pca124_html = pca124_fig.to_html(full_html=True, include_plotlyjs='cdn')
        with open(f'{savefile}_Figure_Tracks.html', 'w') as f:
            f.write(tracks_html)
        with open(f'{savefile}_Figure_PCA123.html', 'w') as f:
            f.write(pca123_html)
        with open(f'{savefile}_Figure_PCA124.html', 'w') as f:
            f.write(pca124_html)
    else:
        pass

    df_msd_loglogfits, msd_fig_all, msd_category_figs = run_msd_graphs(df_msd, color_map)

    with open(f'{savefile}_Figure_Summary-Stats.html', 'w') as f:
        for fig in all_figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    with open(f'{savefile}_Figure_MSD.html', 'w', encoding='utf-8') as f:
        f.write(msd_fig_all.to_html(full_html=True, include_plotlyjs='cdn'))
        for fig in msd_category_figs.values():
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    return df_msd_loglogfits