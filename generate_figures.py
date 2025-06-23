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
    for i, col in enumerate(columns):
        row, col_idx = divmod(i, n_cols)
        for cat in categories:
            df_cat = df[df['Category'] == cat]
            fig.add_trace(
                go.Violin(
                    x=[str(cat)] * len(df_cat),
                    y=df_cat[col],
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                    scalegroup=f'{col}',
                    scalemode='count',
                    width=0.8,
                    box_visible=True,
                    hoverinfo='skip',
                    name=f'Cat {cat}'
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(
            type='category',
            row=row + 1, col=col_idx + 1
        )

    if fit_stats is not None:
        i = len(columns)
        row, col_idx = divmod(i, n_cols)
        for cat in categories:
            stats = fit_stats.get(cat, {})
            slope = stats.get('slope', None)
            ci_low = stats.get('slope', 0) - stats.get('ci_low', 0)
            ci_high = stats.get('ci_high', 0) - stats.get('slope', 0)
            fig.add_trace(
                go.Scatter(
                    x=[str(cat)],
                    y=[slope],
                    mode='markers',
                    marker=dict(
                        color=color_map.get(cat, 'black'),
                        size=11,
                        line=dict(width=1, color=color_map.get(cat, 'black'))
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[ci_high],
                        arrayminus=[ci_low],
                        thickness=1,
                        color='black',
                        width=8
                    ),
                    legendgroup=f'cat{cat}',
                    showlegend=False,
                    name=f'Cat {cat}',
                    hoverinfo='skip'
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(
            type='category',
            row=row + 1, col=col_idx + 1
        )
        fig.update_yaxes(title_text='Slope', range=[0, None], row=row + 1, col=col_idx + 1)
    fig.update_layout(
        violinmode='group',
        plot_bgcolor='white',
        title={'text': 'Summary Statistics', 'x': 0.5, 'font': {'size': 28}},
        height=400 * n_rows,
        autosize=True
    )
    return [fig]

def tracks_figure(df, df_sum, cat_provided, save_file, twodim_mode, color_map=None):
    all_ids = list(df.loc[:, 'Object ID'])
    unique_ids = []
    [unique_ids.append(x) for x in all_ids if x not in unique_ids]

    if cat_provided:
        if color_map is None:
            categories = df_sum['Category'].unique()
            color_map = get_category_color_map(categories)
        all_categories = sorted(df_sum['Category'].dropna().unique())
    else:
        color_map = {0: colors[0]}
        all_categories = [0]

    if twodim_mode:
        specs = [[{"type": "xy"}, {"type": "xy"}]]
    else:
        specs = [[{"type": "scene"}, {"type": "scene"}]]

    fig_category = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Zeroed Tracks", "Raw Tracks"],
        specs=specs,
        horizontal_spacing=0.05
    )

    fig_objects = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Zeroed Tracks", "Raw Tracks"],
        specs=specs,
        horizontal_spacing=0.05
    )

    # Add category legend entries
    for cat in all_categories:
        if twodim_mode:
            fig_category.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    name=f"Category {cat}",
                    mode='lines',
                    line=dict(color=color_map.get(cat, 'black'), width=4),
                    legendgroup=f"cat_{cat}",
                    showlegend=True
                ),
                row=1, col=1
            )
        else:
            fig_category.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    name=f"Category {cat}",
                    mode='lines',
                    line=dict(color=color_map.get(cat, 'black'), width=4),
                    legendgroup=f"cat_{cat}",
                    showlegend=True
                ),
                row=1, col=1
            )

    # Collect all data points to calculate axis ranges for 3D plots
    if not twodim_mode:
        x_all = []
        y_all = []
        z_all = []
        x_zeroed_all = []
        y_zeroed_all = []
        z_zeroed_all = []

    # Add traces for each object
    for object_ in unique_ids:
        cat_row = df_sum.loc[df_sum['Object ID'] == object_, 'Category']
        if cat_row.empty:
            continue
        cat = cat_row.iloc[0]

        df_object = df.loc[df['Object ID'] == object_]
        time_data = list(df_object.loc[:, 'Time'])
        time_data = [f'Time point {x} Category {cat}' for x in time_data]
        x_data = list(df_object.iloc[:, 2])
        y_data = list(df_object.iloc[:, 3])
        x_start = x_data[0] if x_data else 0
        y_start = y_data[0] if y_data else 0
        x_zeroed = [x - x_start for x in x_data]
        y_zeroed = [y - y_start for y in y_data]

        if twodim_mode:
            # Add 2D traces
            fig_category.add_trace(
                go.Scatter(
                    x=x_zeroed, y=y_zeroed,
                    hovertext=time_data,
                    name=str(object_),
                    mode='lines',
                    line=dict(width=2),
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f"cat_{cat}",
                    showlegend=False
                ),
                row=1, col=1
            )

            fig_category.add_trace(
                go.Scatter(
                    x=x_data, y=y_data,
                    hovertext=time_data,
                    name=str(object_),
                    mode='lines',
                    line=dict(width=2),
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f"cat_{cat}",
                    showlegend=False
                ),
                row=1, col=2
            )

            fig_objects.add_trace(
                go.Scatter(
                    x=x_zeroed, y=y_zeroed,
                    hovertext=time_data,
                    name=f"{object_} (Cat {cat})",
                    mode='lines',
                    line=dict(width=2),
                    marker_color=color_map.get(cat, 'black'),
                    showlegend=True,
                    legendgroup=f"obj_{object_}"
                ),
                row=1, col=1
            )

            fig_objects.add_trace(
                go.Scatter(
                    x=x_data, y=y_data,
                    hovertext=time_data,
                    name=f"{object_} (Cat {cat})",
                    mode='lines',
                    line=dict(width=2),
                    marker_color=color_map.get(cat, 'black'),
                    showlegend=False,
                    legendgroup=f"obj_{object_}"
                ),
                row=1, col=2
            )
        else:
            # 3D mode
            z_data = list(df_object.iloc[:, 4])
            z_start = z_data[0] if z_data else 0
            z_zeroed = [z - z_start for z in z_data]

            # Collect data for axis range calculations
            x_all.extend(x_data)
            y_all.extend(y_data)
            z_all.extend(z_data)
            x_zeroed_all.extend(x_zeroed)
            y_zeroed_all.extend(y_zeroed)
            z_zeroed_all.extend(z_zeroed)

            fig_category.add_trace(
                go.Scatter3d(
                    x=x_zeroed, y=y_zeroed, z=z_zeroed,
                    hovertext=time_data,
                    name=str(object_),
                    mode='lines',
                    line=dict(width=4),
                    marker=dict(size=12),
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f"cat_{cat}",
                    showlegend=False
                ),
                row=1, col=1
            )

            fig_category.add_trace(
                go.Scatter3d(
                    x=x_data, y=y_data, z=z_data,
                    hovertext=time_data,
                    name=str(object_),
                    mode='lines',
                    line=dict(width=4),
                    marker=dict(size=12),
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f"cat_{cat}",
                    showlegend=False
                ),
                row=1, col=2
            )

            fig_objects.add_trace(
                go.Scatter3d(
                    x=x_zeroed, y=y_zeroed, z=z_zeroed,
                    hovertext=time_data,
                    name=f"{object_} (Cat {cat})",
                    mode='lines',
                    line=dict(width=4),
                    marker=dict(size=12),
                    marker_color=color_map.get(cat, 'black'),
                    showlegend=True,
                    legendgroup=f"obj_{object_}"
                ),
                row=1, col=1
            )

            fig_objects.add_trace(
                go.Scatter3d(
                    x=x_data, y=y_data, z=z_data,
                    hovertext=time_data,
                    name=f"{object_} (Cat {cat})",
                    mode='lines',
                    line=dict(width=4),
                    marker=dict(size=12),
                    marker_color=color_map.get(cat, 'black'),
                    showlegend=False,
                    legendgroup=f"obj_{object_}"
                ),
                row=1, col=2
            )

    # Update axes for both figures
    if twodim_mode:
        for fig in [fig_category, fig_objects]:
            fig.update_xaxes(title="X", row=1, col=1)
            fig.update_yaxes(title="Y", row=1, col=1)
            fig.update_xaxes(title="X", row=1, col=2)
            fig.update_yaxes(title="Y", row=1, col=2)
    else:
        # Calculate ranges for proportional 3D axes
        # Raw tracks
        x_range = max(x_all) - min(x_all) if x_all else 1
        y_range = max(y_all) - min(y_all) if y_all else 1
        z_range = max(z_all) - min(z_all) if z_all else 1
        max_range = max(x_range, y_range, z_range)

        x_center = (max(x_all) + min(x_all)) / 2 if x_all else 0
        y_center = (max(y_all) + min(y_all)) / 2 if y_all else 0
        z_center = (max(z_all) + min(z_all)) / 2 if z_all else 0

        x_min, x_max = x_center - max_range/2, x_center + max_range/2
        y_min, y_max = y_center - max_range/2, y_center + max_range/2
        z_min, z_max = z_center - max_range/2, z_center + max_range/2

        # Zeroed tracks
        x_zeroed_range = max(x_zeroed_all) - min(x_zeroed_all) if x_zeroed_all else 1
        y_zeroed_range = max(y_zeroed_all) - min(y_zeroed_all) if y_zeroed_all else 1
        z_zeroed_range = max(z_zeroed_all) - min(z_zeroed_all) if z_zeroed_all else 1
        zeroed_max_range = max(x_zeroed_range, y_zeroed_range, z_zeroed_range)

        x_zeroed_center = (max(x_zeroed_all) + min(x_zeroed_all)) / 2 if x_zeroed_all else 0
        y_zeroed_center = (max(y_zeroed_all) + min(y_zeroed_all)) / 2 if y_zeroed_all else 0
        z_zeroed_center = (max(z_zeroed_all) + min(z_zeroed_all)) / 2 if z_zeroed_all else 0

        x_zeroed_min, x_zeroed_max = x_zeroed_center - zeroed_max_range/2, x_zeroed_center + zeroed_max_range/2
        y_zeroed_min, y_zeroed_max = y_zeroed_center - zeroed_max_range/2, y_zeroed_center + zeroed_max_range/2
        z_zeroed_min, z_zeroed_max = z_zeroed_center - zeroed_max_range/2, z_zeroed_center + zeroed_max_range/2

        # Apply proportional ranges to both figures
        for fig in [fig_category, fig_objects]:
            # Update first scene (zeroed tracks) - note: first scene is just 'scene' not 'scene1'
            fig.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[x_zeroed_min, x_zeroed_max], title="X"),
                    yaxis=dict(range=[y_zeroed_min, y_zeroed_max], title="Y"),
                    zaxis=dict(range=[z_zeroed_min, z_zeroed_max], title="Z")
                ),
                # Update second scene (raw tracks)
                scene2=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[x_min, x_max], title="X"),
                    yaxis=dict(range=[y_min, y_max], title="Y"),
                    zaxis=dict(range=[z_min, z_max], title="Z")
                )
            )

    # Set layout properties
    fig_category.update_layout(
        title=f'{save_file} Tracks (Filter by Category)',
        plot_bgcolor='white',
        legend=dict(
            title="Categories"
        )
    )

    fig_objects.update_layout(
        title=f'{save_file} Tracks (Filter by Object ID)',
        plot_bgcolor='white',
        legend=dict(
            title="Object IDs"
        )
    )

    # Generate HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                height: 100vh;
                overflow-x: hidden;
            }}
            #plot-container {{
                width: 95vw;
                height: 70vh;
                margin: 10vh auto 0;
            }}
        </style>
    </head>
    <body>
        <div id="plot-container"></div>
        <script>
            var figure = {figure_json};
            var config = {{responsive: true}};

            Plotly.newPlot('plot-container', figure.data, figure.layout, config);

            if ({is_twodim}) {{
                Plotly.relayout('plot-container', {{
                    'xaxis.scaleanchor': 'y',
                    'xaxis.scaleratio': 1,
                    'xaxis2.scaleanchor': 'y2',
                    'xaxis2.scaleratio': 1
                }});
            }}

            window.addEventListener('resize', function() {{
                if ({is_twodim}) {{
                    Plotly.relayout('plot-container', {{
                        'xaxis.scaleanchor': 'y',
                        'xaxis.scaleratio': 1,
                        'xaxis2.scaleanchor': 'y2',
                        'xaxis2.scaleratio': 1
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """

    html_category = html_template.format(
        title=f"{save_file} Tracks (Category Filtering)",
        figure_json=fig_category.to_json(),
        is_twodim=str(twodim_mode).lower()
    )

    html_objects = html_template.format(
        title=f"{save_file} Tracks (Object ID Filtering)",
        figure_json=fig_objects.to_json(),
        is_twodim=str(twodim_mode).lower()
    )

    return fig_category, html_category, html_objects

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

def msd_graphs(df_msd, df_msd_loglogfits, color_map):
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

def contacts_figures(df_contacts, df_contpercat, color_map=None):
    violin_metrics = [
        'Number of Contacts',
        'Total Time Spent in Contact',
        'Median Contact Duration'
    ]
    bar_metrics = [
        'Pct With Contact',
        'Pct With >=3 Contacts'
    ]
    subplot_titles = [
        'Number of Contacts',
        'Total Time Spent in Contact',
        'Median Contact Duration',
        'Pct With Contact',
        'Pct With >=3 Contacts'
    ]
    n_cols = 3
    n_rows = 2

    categories = sorted(df_contacts['Category'].dropna().unique())
    if color_map is None:
        color_map = get_category_color_map(categories)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.2, horizontal_spacing=0.05
    )

    # Violin plots (object-level, grouped by category)
    for i, col in enumerate(violin_metrics):
        row, col_idx = divmod(i, n_cols)
        for j, cat in enumerate(categories):
            df_cat = df_contacts[df_contacts['Category'] == cat]
            fig.add_trace(
                go.Violin(
                    x=[str(cat)] * len(df_cat),
                    y=df_cat[col],
                    marker_color=color_map.get(cat, 'black'),
                    legendgroup=f'cat{cat}',
                    showlegend=(i == 0),
                    scalegroup=col,
                    scalemode='count',
                    width=0.8,
                    box_visible=True,
                    name=f'Cat {cat}',
                    line_color=color_map.get(cat, 'black'),
                    meanline_visible=True
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(type='category', row=row + 1, col=col_idx + 1)

    # Bar plots (category-level)
    for i, col in enumerate(bar_metrics, start=3):
        row, col_idx = divmod(i, n_cols)
        for j, cat in enumerate(categories):
            y = df_contpercat[df_contpercat['Category'] == cat][col]
            fig.add_trace(
                go.Bar(
                    x=[str(cat)],
                    y=y,
                    marker_color=color_map.get(cat, 'black'),
                    name=f'Cat {cat}',
                    legendgroup=f'cat{cat}',
                    showlegend=False,
                    width=0.5
                ),
                row=row + 1, col=col_idx + 1
            )
        fig.update_xaxes(type='category', row=row + 1, col=col_idx + 1)

    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)

    fig.update_layout(
        violinmode='group',
        barmode='group',
        plot_bgcolor='white',
        title={'text': 'Contacts', 'x': 0.5, 'font': {'size': 28}},
        height=400 * n_rows,
        autosize=True,
        legend_title_text='Category'
    )
    return [fig]

def save_all_figures(df_sum, df_segments, df_pca, df_msd, df_msd_loglogfits, df_contacts, df_contpercat,
                     savefile, cat_provided, twodim_mode):
    color_map = get_category_color_map(df_sum['Category'].unique())

    tracks_fig, tracks_html_category, tracks_html_objects = tracks_figure(
        df_segments, df_sum, cat_provided, savefile, twodim_mode, color_map=color_map)
    with open(f'{savefile}_Figures_Tracks_byCategory.html', 'w', encoding='utf-8') as f:
        f.write(tracks_html_category)
    with open(f'{savefile}_Figures_Tracks_byObjectID.html', 'w', encoding='utf-8') as f:
        f.write(tracks_html_objects)

    if df_pca is not None and not df_pca.empty:
        pca_figs = pca_figures(df_pca, color_map=color_map)
        fig_1d, fig_2d, fig_3d = pca_figs
        with open(f'{savefile}_Figures_PCA.html', 'w', encoding='utf-8') as f:
            f.write(fig_1d.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
            f.write(fig_2d.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))
            fig3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig3d_html}</div>")
    else:
        pass

    msd_fig_all, msd_category_figs, fit_stats = msd_graphs(df_msd, df_msd_loglogfits, color_map)
    with open(f'{savefile}_Figures_MSD.html', 'w', encoding='utf-8') as f:
        f.write(msd_fig_all.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
        for fig in msd_category_figs.values():
            f.write(fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))

    if df_contacts is not None and df_contpercat is not None and not df_contacts.empty and not df_contpercat.empty:
        contacts_figs = contacts_figures(df_contacts, df_contpercat, color_map=color_map)
        with open(f'{savefile}_Figures_Contacts.html', 'w', encoding='utf-8') as f:
            for fig in contacts_figs:
                f.write(fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))

    sumstat_figs = summary_figures(df_sum, fit_stats, color_map=color_map)
    with open(f'{savefile}_Figures_Summary-Stats.html', 'w', encoding='utf-8') as f:
        for fig in sumstat_figs:
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig_html}</div>")

    return
