import math
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from msd_graphs import run_msd_graphs

colors = plotly.colors.qualitative.Plotly

def get_category_color_map(categories):
    categories = [int(cat) for cat in categories if pd.notnull(cat)]
    categories = sorted(set(categories))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

def summary_figures(df, color_map=None):
    columns = [col for col in df.columns if col not in ('Object ID', 'Category')]
    categories = sorted(df['Category'].dropna().unique())
    if color_map is None:
        color_map = get_category_color_map(categories)
    n_plots = len(columns)
    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=columns,
        vertical_spacing=0.05, horizontal_spacing=0.05
    )
    fig.update_layout(
        height=400 * n_rows,
        width=None,
        autosize=True,
        title={
            'text': 'Summary Statistics',
            'x': 0.5,
            'font': {'size': 28}
        },
        plot_bgcolor='white',
        violinmode='group'
    )
    for i, col in enumerate(columns):
        row, col_idx = divmod(i, n_cols)
        for cat in categories:
            df_cat = df[df['Category'] == cat]
            fig.add_trace(
                go.Violin(
                    x=[cat] * len(df_cat),
                    y=df_cat[col],
                    marker_color=color_map.get(cat, colors[0]),
                    showlegend=False,
                    scalegroup=f'{col}',
                    scalemode='count',
                    width=0.9,
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(
            tickmode='array',
            tickvals=categories,
            ticktext=[str(cat) for cat in categories],
            row=row + 1, col=col_idx + 1
        )
    return [fig]

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

def pca_figures(df_pca, color_map=None):
    pcs = ['PC1', 'PC2', 'PC3', 'PC4']
    categories = sorted(df_pca['Category'].dropna().unique(), key=lambda x: str(x))
    if color_map is None:
        color_map = get_category_color_map(categories)

    # 1D violin plots
    pcafig_1d = make_subplots(rows=2, cols=2, subplot_titles=pcs)
    for i, pc in enumerate(pcs):
        row, col = divmod(i, 2)
        for cat in categories:
            y = df_pca[df_pca['Category'] == cat][pc]
            pcafig_1d.add_trace(
                go.Violin(
                    y=y,
                    name=f'Cat {cat}',
                    line_color=color_map[cat],
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                ),
                row=row + 1, col=col + 1
            )

    # 2D scatter plots
    pc_pairs = list(combinations(pcs, 2))
    pcafig_2d = make_subplots(rows=2, cols=3, subplot_titles=[f'{x} vs {y}' for x, y in pc_pairs])
    for i, (x, y) in enumerate(pc_pairs):
        row, col = divmod(i, 3)
        for cat in categories:
            df_cat = df_pca[df_pca['Category'] == cat]
            pcafig_2d.add_trace(
                go.Scatter(
                    x=df_cat[x],
                    y=df_cat[y],
                    mode='markers',
                    marker=dict(color=color_map[cat]),
                    name=f'Cat {cat}',
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                ),
                row=row + 1, col=col + 1
            )
        pcafig_2d.update_xaxes(title_text=x, row=row + 1, col=col + 1)
        pcafig_2d.update_yaxes(title_text=y, row=row + 1, col=col + 1)
    for i in range(1, 7):
        pcafig_2d.update_xaxes(scaleanchor=f'y{i}', scaleratio=1, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
        pcafig_2d.update_yaxes(scaleanchor=f'x{i}', scaleratio=1, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
    pcafig_2d.update_layout(
        width=None,
        autosize=True,
    )

    # 3D scatter plots
    pc_triples = list(combinations(pcs, 3))
    n_rows = 2
    pcafig_3d = make_subplots(
        rows=n_rows, cols=2, specs=[[{'type': 'scene'}] * 2] * n_rows,
        subplot_titles=[f'{x}, {y}, {z}' for x, y, z in pc_triples],
        vertical_spacing=0.05, horizontal_spacing=0.05
    )
    pcafig_3d.update_layout(
        height=1000 * n_rows,
        width=None,
        autosize=True
    )
    for i, (x, y, z) in enumerate(pc_triples):
        row, col = divmod(i, 2)
        for cat in categories:
            df_cat = df_pca[df_pca['Category'] == cat]
            pcafig_3d.add_trace(
                go.Scatter3d(
                    x=df_cat[x], y=df_cat[y], z=df_cat[z],
                    mode='markers',
                    marker=dict(color=color_map[cat]),
                    name=f'Cat {cat}',
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0)
                ),
                row=row + 1, col=col + 1
            )
        scene_id = f'scene{1 + i}'
        pcafig_3d.update_layout({
            scene_id: dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
                aspectmode='cube'
            )
        })

    return pcafig_1d, pcafig_2d, pcafig_3d

def save_all_figures(df_sum, df_segments, df_pca, df_msd, savefile, cat_provided):
    color_map = get_category_color_map(df_sum['Category'].unique())

    sumstat_figs = summary_figures(df_sum, color_map=color_map)
    with open(f'{savefile}_Figure_Summary-Stats.html', 'w') as f:
        for fig in sumstat_figs:
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig_html}</div>")

    tracks_fig = tracks_figure(df_segments, df_sum, cat_provided, savefile, color_map=color_map)
    tracks_html = tracks_fig.to_html(full_html=True, include_plotlyjs='cdn')
    with open(f'{savefile}_Figure_Tracks.html', 'w') as f:
        f.write(tracks_html)

    if df_pca is not None and not df_pca.empty:
        pca_figs = pca_figures(df_pca, color_map=color_map)
        fig_1d, fig_2d, fig_3d = pca_figs
        with open(f'{savefile}_Figure_PCA.html', 'w', encoding='utf-8') as f:
            f.write(fig_1d.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
            f.write(fig_2d.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))
            fig3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig3d_html}</div>")
    else:
        pass

    df_msd_loglogfits, msd_fig_all, msd_category_figs = run_msd_graphs(df_msd, color_map)
    with open(f'{savefile}_Figure_MSD.html', 'w', encoding='utf-8') as f:
        f.write(msd_fig_all.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
        for fig in msd_category_figs.values():
            f.write(fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))

    return df_msd_loglogfits
