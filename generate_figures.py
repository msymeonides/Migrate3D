import math
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

colors = plotly.colors.qualitative.Plotly

def get_category_color_map(categories):
    categories = [int(cat) for cat in categories if pd.notnull(cat)]
    categories = sorted(set(categories))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

def summary_figures(df, fit_stats, color_map=None):
    columns = [col for col in df.columns if col not in ('Object ID', 'Category')]
    categories = sorted(df['Category'].dropna().unique())
    if color_map is None:
        color_map = get_category_color_map(categories)
    n_plots = len(columns)
    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=columns + ['MSD log-log fit slope'],
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
                    marker_color=color_map.get(cat, 'black'),
                    showlegend=False,
                    scalegroup=f'{col}',
                    scalemode='count',
                    width=0.8,
                    box_visible=True,
                    hoverinfo='skip'
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(
            tickmode='array',
            tickvals=categories,
            ticktext=[str(cat) for cat in categories],
            row=row + 1, col=col_idx + 1
        )
    if fit_stats is not None:
        i = len(columns)
        row, col_idx = divmod(i, n_cols)
        cats = sorted(fit_stats.keys())
        slopes = [fit_stats[cat]['slope'] for cat in cats]
        ci_low = [fit_stats[cat]['slope'] - fit_stats[cat]['ci_low'] for cat in cats]
        ci_high = [fit_stats[cat]['ci_high'] - fit_stats[cat]['slope'] for cat in cats]
        colors_list = [color_map.get(cat, 'black') for cat in cats]
        fig.add_trace(
            go.Scatter(
                x=cats,
                y=slopes,
                mode='markers',
                marker=dict(
                    color=colors_list,
                    size=11,
                    line=dict(width=1, color=colors_list)
                ),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=ci_high,
                    arrayminus=ci_low,
                    thickness=1,
                    color='black',
                    width=8
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row + 1, col=col_idx + 1
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=cats,
            ticktext=[str(cat) for cat in cats],
            row=row + 1, col=col_idx + 1
        )
        fig.update_yaxes(title_text='Slope', row=row + 1, col=col_idx + 1)
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
            marker=dict(size=12), marker_color=color_map.get(cat, 'black')
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
                    box_visible=True,
                    hoverinfo='skip'
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
                    marker=dict(color=color_map[cat], size=10),
                    name=f'Cat {cat}',
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                    hoverinfo='skip'
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
                    marker=dict(color=color_map[cat], size=6),
                    name=f'Cat {cat}',
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                    hoverinfo='skip'
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

def run_msd_graphs(df_msd, df_msd_loglogfits, color_map):
    fit_stats = {}
    for col in df_msd_loglogfits.columns:
        try:
            cat = int(col)
        except (ValueError, TypeError):
            cat = col
        fit_stats[cat] = {
            'slope': df_msd_loglogfits.at['Slope', col],
            'ci_low': df_msd_loglogfits.at['Lower 95% CI', col],
            'ci_high': df_msd_loglogfits.at['Upper 95% CI', col],
            'r2': df_msd_loglogfits.at['Fit R2', col],
            'max_tau': df_msd_loglogfits.at['Fit Max. Tau', col]
        }
    id_cols = ['Object ID', 'Category']
    tau_cols = [col for col in df_msd.columns if col not in id_cols]
    df_long = df_msd.melt(id_vars=id_cols, value_vars=tau_cols, var_name='tau', value_name='msd')
    df_long['tau'] = df_long['tau'].astype(float)
    df_long = df_long[(df_long['tau'] > 0) & (df_long['msd'] > 0)]
    df_long['log_tau'] = np.log10(df_long['tau'])
    df_long['log_msd'] = np.log10(df_long['msd'])

    x_min, x_max = df_long['log_tau'].min(), df_long['log_tau'].max()
    y_min, y_max = df_long['log_msd'].min(), df_long['log_msd'].max()
    categories = sorted([int(cat) for cat in df_long['Category'].unique() if pd.notnull(cat)])
    msd_figure_categories = {}

    for category in categories:
        cat_df = df_long[df_long['Category'] == category]
        fig = go.Figure()
        for obj_id, group in cat_df.groupby('Object ID'):
            fig.add_trace(go.Scatter(
                x=group['log_tau'],
                y=group['log_msd'],
                mode='lines',
                line=dict(color='lightgrey', width=1),
                showlegend=False
            ))
        mean_log = cat_df.groupby('log_tau')['log_msd'].mean().reset_index()
        x = mean_log['log_tau'].values
        y = mean_log['log_msd'].values

        stats = fit_stats.get(category, {})
        slope = stats.get('slope', np.nan)
        intercept = y[0] - slope * x[0] if not np.isnan(slope) else np.nan
        x_fit_start = x[0]
        x_fit_end = x[-1]
        x_fit = [x_fit_start, x_fit_end]
        y_fit_line = [slope * xi + intercept for xi in x_fit]

        annotation_text = (
            f"Slope: {slope:.3f}<br>"
            f"95% CI: {stats.get('ci_low', np.nan):.3f}, {stats.get('ci_high', np.nan):.3f}<br>"
            f"R2: {stats.get('r2', np.nan):.3f}"
        )

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='black', width=3),
            name='Mean log(MSD)'
        ))
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit_line,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Linear fit (tau {10**x_fit_start:.2g}–{stats.get("max_tau", x[-1]):.2g})'
        ))
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            text=annotation_text,
            showarrow=False,
            align='left',
            font=dict(size=14, color='red'),
            bgcolor='white'
        )
        fig.update_layout(
            title=f'Category {category}',
            xaxis_title='log10(Tau)',
            yaxis_title='log10(MSD)',
            template='simple_white',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )
        msd_figure_categories[category] = fig

    msd_figure_all = go.Figure()
    for category in categories:
        cat_df = df_long[df_long['Category'] == category]
        mean_log = cat_df.groupby('log_tau')['log_msd'].mean().reset_index()
        x = mean_log['log_tau'].values
        y = mean_log['log_msd'].values
        stats = fit_stats.get(category, {})
        slope = stats.get('slope', np.nan)
        intercept = y[0] - slope * x[0] if not np.isnan(slope) else np.nan
        x_fit_start = x[0]
        x_fit_end = x[-1]
        x_fit = [x_fit_start, x_fit_end]
        y_fit_line = [slope * xi + intercept for xi in x_fit]

        color = color_map.get(category, 'blue')
        msd_figure_all.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'Category {category}',
            line=dict(color=color, width=3)
        ))
        msd_figure_all.add_trace(go.Scatter(
            x=x_fit, y=y_fit_line,
            mode='lines',
            name=f'Linear fit (Cat {category}, tau {10**x_fit_start:.2g}–{stats.get("max_tau", x[-1]):.2g})',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=True
        ))
    msd_figure_all.update_layout(
        title='Mean log(MSD) for All Categories',
        xaxis_title='log10(Tau)',
        yaxis_title='log10(MSD)',
        template='simple_white',
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    return msd_figure_all, msd_figure_categories, fit_stats

def save_all_figures(df_sum, df_segments, df_pca, df_msd, df_msd_loglogfits, savefile, cat_provided):
    color_map = get_category_color_map(df_sum['Category'].unique())

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

    msd_fig_all, msd_category_figs, fit_stats = run_msd_graphs(df_msd, df_msd_loglogfits, color_map)
    with open(f'{savefile}_Figure_MSD.html', 'w', encoding='utf-8') as f:
        f.write(msd_fig_all.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
        for fig in msd_category_figs.values():
            f.write(fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))

    sumstat_figs = summary_figures(df_sum, fit_stats, color_map=color_map)
    with open(f'{savefile}_Figure_Summary-Stats.html', 'w') as f:
        for fig in sumstat_figs:
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig_html}</div>")

    return
