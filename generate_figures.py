import math
import multiprocessing as mp
import gc
import psutil
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

colors = plotly.colors.qualitative.Plotly

MEMORY_SAFETY_THRESHOLD = 0.75

def get_category_color_map(cats_or_objs):
    cats_or_objs = [str(cat) for cat in cats_or_objs if pd.notnull(cat)]
    cats_or_objs = sorted(set(cats_or_objs))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(cats_or_objs)}

def estimate_track_processing_memory(df, df_sum, num_batches):
    mem_info = psutil.virtual_memory()
    available_gb = mem_info.available / (1024**3)

    df_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
    df_sum_size_gb = df_sum.memory_usage(deep=True).sum() / (1024**3)
    input_size_gb = df_size_gb + df_sum_size_gb

    total_rows = len(df)
    num_objects = df['Object ID'].nunique()
    avg_points_per_object = total_rows / num_objects if num_objects > 0 else 0

    bytes_per_track = avg_points_per_object * 150
    estimated_output_bytes = num_objects * bytes_per_track
    estimated_output_gb = estimated_output_bytes / (1024**3)

    if estimated_output_gb < 0.1:
        peak_multiplier = 5.5
    elif estimated_output_gb < 1.0:
        peak_multiplier = 5.0
    else:
        peak_multiplier = 4.5

    estimated_peak_gb = (input_size_gb + estimated_output_gb) * peak_multiplier

    use_multiprocessing = estimated_peak_gb <= (available_gb * MEMORY_SAFETY_THRESHOLD)

    return {
        'input_size_gb': input_size_gb,
        'estimated_output_gb': estimated_output_gb,
        'estimated_peak_gb': estimated_peak_gb,
        'available_gb': available_gb,
        'total_gb': mem_info.total / (1024**3),
        'use_multiprocessing': use_multiprocessing,
        'num_objects': num_objects,
        'num_batches': num_batches,
        'total_rows': total_rows
    }

def estimate_pca_processing_memory(df_pca):
    mem_info = psutil.virtual_memory()
    available_gb = mem_info.available / (1024**3)

    df_size_gb = df_pca.memory_usage(deep=True).sum() / (1024**3)

    estimated_peak_gb = df_size_gb * 6.5

    use_multiprocessing = estimated_peak_gb <= (available_gb * MEMORY_SAFETY_THRESHOLD)

    return {
        'input_size_gb': df_size_gb,
        'estimated_peak_gb': estimated_peak_gb,
        'available_gb': available_gb,
        'use_multiprocessing': use_multiprocessing
    }

def process_object_track_batch(batch_args):
    results = []
    df_batch, df_sum_batch, twodim_mode = batch_args

    for obj_id in df_batch['Object ID'].unique():
        obj_data = df_batch[df_batch['Object ID'] == obj_id]
        cat_row = df_sum_batch[df_sum_batch['Object ID'] == obj_id]

        if cat_row.empty:
            continue

        cat = cat_row.iloc[0]['Category']

        time_data = [f'Time point {x} Category {cat}' for x in obj_data['Time']]
        x_data = list(obj_data.iloc[:, 2])
        y_data = list(obj_data.iloc[:, 3])

        if not x_data or not y_data:
            continue

        x_start, y_start = x_data[0], y_data[0]
        x_zeroed = [x - x_start for x in x_data]
        y_zeroed = [y - y_start for y in y_data]

        if not twodim_mode:
            z_data = list(obj_data.iloc[:, 4])
            if not z_data:
                continue
            z_start = z_data[0]
            z_zeroed = [z - z_start for z in z_data]
        else:
            z_data = None
            z_zeroed = None

        results.append({
            'object_id': obj_id,
            'category': cat,
            'time_data': time_data,
            'x_data': x_data,
            'y_data': y_data,
            'z_data': z_data,
            'x_zeroed': x_zeroed,
            'y_zeroed': y_zeroed,
            'z_zeroed': z_zeroed
        })

    return results

def summary_figures(df, fit_stats, color_map=None):
    columns = [col for col in df.columns if col not in ('Object ID', 'Category')]
    categories = sorted(df['Category'].dropna().unique())
    if color_map is None:
        color_map = get_category_color_map(categories)

    if fit_stats is not None and 'Outreach Ratio' in columns:
        outreach_idx = columns.index('Outreach Ratio')
        columns.insert(outreach_idx + 1, 'MSD log-log fit slope')
    elif fit_stats is not None:
        columns.append('MSD log-log fit slope')

    n_plots = len(columns)
    subplot_titles = columns
    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
        vertical_spacing=0.05, horizontal_spacing=0.05
    )

    for i, col in enumerate(columns):
        row, col_idx = divmod(i, n_cols)

        if col == 'MSD log-log fit slope' and fit_stats is not None:
            y_error_tops = []

            for cat in categories:
                stats = fit_stats.get(cat, {})
                slope = stats.get('slope', None)
                if slope is not None:
                    ci_high = stats.get('ci_high', 0) - stats.get('slope', 0)
                    y_error_tops.append(slope + ci_high)

            if y_error_tops:
                y_max = max(y_error_tops)
                y_padding = y_max * 0.1
                y_range = [0, y_max + y_padding]
            else:
                y_range = [0, None]

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
                        name=f'Cat {cat}'
                    ),
                    row=row + 1, col=col_idx + 1
                )
            fig.update_xaxes(
                type='category',
                range=[-0.5, len(categories) - 0.5],
                row=row + 1, col=col_idx + 1
            )
            fig.update_yaxes(
                title_text='Slope',
                range=y_range,
                row=row + 1, col=col_idx + 1
            )
        else:
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
                        name=f'Cat {cat}'
                    ),
                    row=row + 1, col=col_idx + 1
                )
            fig.update_xaxes(
                type='category',
                row=row + 1, col=col_idx + 1
            )

    fig.update_layout(
        violinmode='group',
        plot_bgcolor='white',
        title={'text': 'Summary Features', 'x': 0.5, 'font': {'size': 28}},
        height=400 * n_rows,
        autosize=True
    )
    return [fig]

def tracks_figure(df, df_sum, cat_provided, save_file, twodim_mode, color_map=None, color_by_object=False):
    def prepare_data():
        data_dict = {}
        data_dict['unique_ids'] = list(dict.fromkeys(all_ids))

        data_dict['zeroed_x_data'] = []
        data_dict['zeroed_y_data'] = []
        data_dict['raw_x_data'] = []
        data_dict['raw_y_data'] = []

        if not twodim_mode:
            data_dict['x_all'] = []
            data_dict['y_all'] = []
            data_dict['z_all'] = []
            data_dict['x_zeroed_all'] = []
            data_dict['y_zeroed_all'] = []
            data_dict['z_zeroed_all'] = []

        return data_dict

    def calculate_range_with_padding(min_val, max_val, padding_factor):
        range_size = max_val - min_val
        padding = range_size * padding_factor
        return min_val - padding, max_val + padding

    def add_origin_lines(fig, x_min, x_max, y_min, y_max, col=1):
        if y_min <= 0 <= y_max:
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max], y=[0, 0],
                    mode='lines', line=dict(color='black', width=1),
                    opacity=0.7, showlegend=False
                ),
                row=1, col=col
            )

        if x_min <= 0 <= x_max:
            fig.add_trace(
                go.Scatter(
                    x=[0, 0], y=[y_min, y_max],
                    mode='lines', line=dict(color='black', width=1),
                    opacity=0.7, showlegend=False
                ),
                row=1, col=col
            )

    def add_category_legend(fig, categories):
        for cat in categories:
            color_key = cat if not color_by_object else None
            trace_args = {
                'name': f"Category {cat}",
                'mode': 'lines',
                'line': dict(color=color_map.get(color_key, 'black'), width=4),
                'legendgroup': f"cat_{cat}",
                'showlegend': True
            }

            if twodim_mode:
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], **trace_args),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter3d(x=[None], y=[None], z=[None], **trace_args),
                    row=1, col=1
                )

    def add_track_trace(fig, track_data, zeroed=False, obj_figure=False, color_by_object=False):
        x_data = track_data['x_zeroed'] if zeroed else track_data['x_data']
        y_data = track_data['y_zeroed'] if zeroed else track_data['y_data']
        z_data = track_data['z_zeroed'] if zeroed else track_data['z_data']

        obj_name = str(track_data['object_id'])
        category = str(track_data['category'])
        time_data = track_data['time_data']

        color_key = obj_name if color_by_object else category
        col = 1 if zeroed else 2
        common_args = {
            'name': obj_name,
            'mode': 'lines',
            'marker_color': color_map.get(color_key, 'black'),
        }

        if obj_figure:
            common_args['name'] = f"{obj_name} (Cat {category})"
            common_args['legendgroup'] = f"obj_{obj_name}"
            common_args['showlegend'] = zeroed
        else:
            common_args['legendgroup'] = f"cat_{category}"
            common_args['showlegend'] = False

        if twodim_mode:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=y_data,
                    hovertext=time_data,
                    line=dict(width=2),
                    **common_args
                ),
                row=1, col=col
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=x_data, y=y_data, z=z_data,
                    line=dict(width=4),
                    marker=dict(size=12),
                    **common_args
                ),
                row=1, col=col
            )

    def calculate_3d_axis_range(x_vals, y_vals, z_vals):
        if not (x_vals and y_vals and z_vals):
            return (0, 1), (0, 1), (0, 1)

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        return (
            (x_center - max_range / 2, x_center + max_range / 2),
            (y_center - max_range / 2, y_center + max_range / 2),
            (z_center - max_range / 2, z_center + max_range / 2)
        )

    def setup_axes(fig, all_track_data):
        if twodim_mode:
            for col in [1, 2]:
                fig.update_xaxes(
                    title="X", row=1, col=col,
                    showline=True, linewidth=1, linecolor='black', mirror=True,
                    ticks='outside', ticklen=5, tickwidth=1,
                    showgrid=False, zeroline=False
                )
                fig.update_yaxes(
                    title="Y", row=1, col=col,
                    showline=True, linewidth=1, linecolor='black', mirror=True,
                    ticks='outside', ticklen=5, tickwidth=1,
                    showgrid=False, zeroline=False
                )
        else:
            x_all = []
            y_all = []
            z_all = []
            x_zeroed_all = []
            y_zeroed_all = []
            z_zeroed_all = []

            for track in all_track_data:
                if track['x_data'] and track['y_data'] and track['z_data']:
                    x_all.extend(track['x_data'])
                    y_all.extend(track['y_data'])
                    z_all.extend(track['z_data'])
                    x_zeroed_all.extend(track['x_zeroed'])
                    y_zeroed_all.extend(track['y_zeroed'])
                    z_zeroed_all.extend(track['z_zeroed'])

            if x_all and y_all and z_all:
                (x_min, x_max), (y_min, y_max), (z_min, z_max) = calculate_3d_axis_range(
                    x_all, y_all, z_all
                )

                fig.update_layout(
                    scene2=dict(
                        aspectmode='cube',
                        xaxis=dict(range=[x_min, x_max], title="X"),
                        yaxis=dict(range=[y_min, y_max], title="Y"),
                        zaxis=dict(range=[z_min, z_max], title="Z")
                    )
                )

            if x_zeroed_all and y_zeroed_all and z_zeroed_all:
                (x_zeroed_min, x_zeroed_max), (y_zeroed_min, y_zeroed_max), (z_zeroed_min,
                                                                             z_zeroed_max) = calculate_3d_axis_range(
                    x_zeroed_all, y_zeroed_all, z_zeroed_all
                )

                fig.update_layout(
                    scene=dict(
                        aspectmode='cube',
                        xaxis=dict(range=[x_zeroed_min, x_zeroed_max], title="X"),
                        yaxis=dict(range=[y_zeroed_min, y_zeroed_max], title="Y"),
                        zaxis=dict(range=[z_zeroed_min, z_zeroed_max], title="Z")
                    )
                )

    axis_padding = 0.1
    all_ids = list(df.loc[:, 'Object ID'])

    if cat_provided:
        if color_map is None:
            categories = df_sum['Category'].unique()
            color_map = get_category_color_map(categories)
        all_categories = sorted(df_sum['Category'].dropna().unique())
    else:
        color_map = {0: colors[0]}
        all_categories = [0]

    specs = [[{"type": "xy"}, {"type": "xy"}]] if twodim_mode else [[{"type": "scene"}, {"type": "scene"}]]

    data = prepare_data()

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

    add_category_legend(fig_category, all_categories)

    unique_objects = df['Object ID'].unique()
    batch_size = max(50, len(unique_objects) // (max(1, min(61, mp.cpu_count() - 2)) * 2))

    object_batches = []
    for i in range(0, len(unique_objects), batch_size):
        batch_objects = unique_objects[i:i + batch_size]
        df_batch = df[df['Object ID'].isin(batch_objects)]
        df_sum_batch = df_sum[df_sum['Object ID'].isin(batch_objects)]
        object_batches.append((df_batch, df_sum_batch, twodim_mode))

    mem_est = estimate_track_processing_memory(df, df_sum, len(object_batches))
    use_multiprocessing = mem_est['use_multiprocessing']

    all_track_data = []

    if use_multiprocessing:
        num_workers = max(1, min(61, mp.cpu_count() - 2, len(object_batches)))
        try:
            with mp.Pool(processes=num_workers) as pool:
                batch_results = pool.map(process_object_track_batch, object_batches)
                for results in batch_results:
                    all_track_data.extend(results)
                gc.collect()
        except Exception:
            for batch in object_batches:
                try:
                    batch_results = process_object_track_batch(batch)
                    all_track_data.extend(batch_results)
                except Exception:
                    continue
            gc.collect()
    else:
        for i, batch in enumerate(object_batches):
            try:
                batch_results = process_object_track_batch(batch)
                all_track_data.extend(batch_results)
                if (i + 1) % 10 == 0:
                    gc.collect()
            except Exception:
                continue
        gc.collect()

    if twodim_mode:
        for track in all_track_data:
            data['zeroed_x_data'].extend(track['x_zeroed'])
            data['zeroed_y_data'].extend(track['y_zeroed'])
            data['raw_x_data'].extend(track['x_data'])
            data['raw_y_data'].extend(track['y_data'])

        if data['zeroed_x_data'] and data['zeroed_y_data']:
            x_min, x_max = min(data['zeroed_x_data']), max(data['zeroed_x_data'])
            y_min, y_max = min(data['zeroed_y_data']), max(data['zeroed_y_data'])
            x_min, x_max = calculate_range_with_padding(x_min, x_max, axis_padding)
            y_min, y_max = calculate_range_with_padding(y_min, y_max, axis_padding)

            for fig in [fig_category, fig_objects]:
                add_origin_lines(fig, x_min, x_max, y_min, y_max, col=1)

        if data['raw_x_data'] and data['raw_y_data']:
            x_min, x_max = min(data['raw_x_data']), max(data['raw_x_data'])
            y_min, y_max = min(data['raw_y_data']), max(data['raw_y_data'])
            x_min, x_max = calculate_range_with_padding(x_min, x_max, axis_padding)
            y_min, y_max = calculate_range_with_padding(y_min, y_max, axis_padding)

            for fig in [fig_category, fig_objects]:
                add_origin_lines(fig, x_min, x_max, y_min, y_max, col=2)

    for track_data in all_track_data:
        for fig, is_obj_figure in [(fig_category, False), (fig_objects, True)]:
            add_track_trace(fig, track_data, zeroed=True, obj_figure=is_obj_figure, color_by_object=color_by_object)
            add_track_trace(fig, track_data, zeroed=False, obj_figure=is_obj_figure, color_by_object=color_by_object)

    setup_axes(fig_category, all_track_data)
    setup_axes(fig_objects, all_track_data)

    fig_category.update_layout(
        title=f'{save_file} Tracks (Filter by Category)',
        plot_bgcolor='white',
        legend=dict(title="Categories")
    )

    fig_objects.update_layout(
        title=f'{save_file} Tracks (Filter by Object ID)',
        plot_bgcolor='white',
        legend=dict(title="Object IDs")
    )

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
            var config = {{
                responsive: true,
                toImageButtonOptions: {{ format: 'png', filename: '{title}' }},
                displayModeBar: true,
                displaylogo: false
            }};

            Plotly.newPlot('plot-container', figure.data, figure.layout, config);

            Plotly.relayout('plot-container', {{
                'xaxis.scaleanchor': 'y',
                'xaxis.scaleratio': 1,
                'xaxis2.scaleanchor': 'y2',
                'xaxis2.scaleratio': 1
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

def _create_pca_3d_subplot(args):
    """Helper function to create a single 3D PCA subplot (for multiprocessing)"""
    i, pc_triple, df_pca, color_map = args
    x, y, z = pc_triple
    categories = sorted(df_pca['Category'].dropna().unique(), key=lambda x: str(x))

    traces = []
    for cat in categories:
        df_cat = df_pca[df_pca['Category'] == cat]
        trace = go.Scatter3d(
            x=df_cat[x], y=df_cat[y], z=df_cat[z],
            mode='markers',
            marker=dict(color=color_map[cat], size=6),
            name=f'Cat {cat}',
            legendgroup=f'cat{cat}',
            showlegend=(i == 0)
        )
        traces.append(trace)

    scene_config = dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z,
        aspectmode='cube'
    )

    return (i, traces, scene_config, f'{x}, {y}, {z}')

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
                    box_visible=True
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
                    showlegend=(i == 0)
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

def msd_graphs(df_msd, df_msd_loglogfits_long, color_map):
    fit_stats = {}
    for cat in df_msd_loglogfits_long['Category'].unique():
        cat_rows = df_msd_loglogfits_long[df_msd_loglogfits_long['Category'] == cat]
        fit_stats[cat] = {
            'slope': cat_rows.loc[cat_rows['Statistic'] == 'Slope', 'Value'].values[0],
            'ci_low': cat_rows.loc[cat_rows['Statistic'] == 'Lower 95% CI', 'Value'].values[0],
            'ci_high': cat_rows.loc[cat_rows['Statistic'] == 'Upper 95% CI', 'Value'].values[0],
            'r2': cat_rows.loc[cat_rows['Statistic'] == 'Fit R2', 'Value'].values[0],
            'max_tau': cat_rows.loc[cat_rows['Statistic'] == 'Fit Max. Tau', 'Value'].values[0]
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
    categories = sorted([str(cat) for cat in df_long['Category'].unique() if pd.notnull(cat)])

    mem_est = estimate_msd_processing_memory(df_long, len(categories))
    use_multiprocessing = mem_est['use_multiprocessing']

    msd_figure_categories = {}
    if len(categories) > 1 and use_multiprocessing:
        num_workers = max(1, min(mp.cpu_count() - 1, len(categories)))
        try:
            with mp.Pool(processes=num_workers) as pool:
                args_list = [(category, df_long, fit_stats, x_min, x_max, y_min, y_max)
                             for category in categories]
                results = pool.map(_create_msd_category_figure, args_list)
                msd_figure_categories = dict(results)
                gc.collect()
        except Exception:
            for category in categories:
                fig = _create_msd_category_figure((category, df_long, fit_stats, x_min, x_max, y_min, y_max))
                msd_figure_categories[fig[0]] = fig[1]
            gc.collect()
    else:
        for i, category in enumerate(categories):
            fig = _create_msd_category_figure((category, df_long, fit_stats, x_min, x_max, y_min, y_max))
            msd_figure_categories[fig[0]] = fig[1]
            if (i + 1) % 3 == 0:
                gc.collect()
        gc.collect()

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
            name=f'Linear fit (Cat {category}, tau {int(round(10**x_fit_start))}–{int(round(stats.get("max_tau", x[-1])))} )',
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


def _create_msd_category_figure(args):
    category, df_long, fit_stats, x_min, x_max, y_min, y_max = args

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
        name=f'Linear fit (tau {int(round(10**x_fit_start))}–{int(round(stats.get("max_tau", x[-1])))} )',
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

    return (category, fig)

def estimate_msd_processing_memory(df_long, num_categories):
    mem_info = psutil.virtual_memory()
    available_gb = mem_info.available / (1024**3)

    df_size_gb = df_long.memory_usage(deep=True).sum() / (1024**3)
    num_objects = df_long['Object ID'].nunique()

    estimated_output_bytes = num_objects * num_categories * 500
    estimated_output_gb = estimated_output_bytes / (1024**3)

    if num_objects > 1000:
        peak_multiplier = 0.8
    elif num_objects > 500:
        peak_multiplier = 0.7
    else:
        peak_multiplier = 0.6

    estimated_peak_gb = (df_size_gb + estimated_output_gb) * peak_multiplier
    use_multiprocessing = estimated_peak_gb <= (available_gb * MEMORY_SAFETY_THRESHOLD)

    return {
        'input_size_gb': df_size_gb,
        'estimated_output_gb': estimated_output_gb,
        'estimated_peak_gb': estimated_peak_gb,
        'available_gb': available_gb,
        'use_multiprocessing': use_multiprocessing,
        'num_objects': num_objects,
        'num_categories': num_categories
    }

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
    for df in [df_sum, df_segments, df_pca, df_msd, df_contacts, df_contpercat]:
        if df is not None and not df.empty:
            if 'Object ID' in df.columns:
                df['Object ID'] = df['Object ID'].astype(int)
            if 'Category' in df.columns:
                df['Category'] = df['Category'].astype(str)

    df_msd_loglogfits_long = None
    if df_msd_loglogfits is not None:
        if df_msd_loglogfits.index.name is None:
            df_msd_loglogfits.index.name = 'Statistic'
        df_msd_loglogfits_long = df_msd_loglogfits.reset_index().melt(
            id_vars='Statistic', var_name='Category', value_name='Value'
        )

    categories = df_sum['Category'].unique()
    if len(categories) == 1:
        unique_objects = list(df_segments['Object ID'].unique())
        color_map = {obj: colors[i % len(colors)] for i, obj in enumerate(unique_objects)}
        color_by_object = True
    else:
        color_map = get_category_color_map(categories)
        color_by_object = False

    if cat_provided:
        tracks_fig, tracks_html_category, tracks_html_objects = tracks_figure(
            df_segments, df_sum, True, savefile, twodim_mode,
            color_map=color_map, color_by_object=color_by_object
        )
        with open(f'{savefile}_Figures_Tracks_byCategory.html', 'w', encoding='utf-8') as f:
            f.write(tracks_html_category)
    else:
        tracks_fig, tracks_html_category, tracks_html_objects = tracks_figure(
            df_segments, df_sum, True, savefile, twodim_mode,
            color_map=color_map, color_by_object=color_by_object
        )

    with open(f'{savefile}_Figures_Tracks_byObjectID.html', 'w', encoding='utf-8') as f:
        f.write(tracks_html_objects)

    if cat_provided and df_pca is not None and not df_pca.empty:
        pca_figs = pca_figures(df_pca, color_map=color_map)
        fig_1d, fig_2d, fig_3d = pca_figs
        with open(f'{savefile}_Figures_PCA.html', 'w', encoding='utf-8') as f:
            f.write(fig_1d.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
            f.write(fig_2d.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True}))
            fig3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig3d_html}</div>")
    else:
        pass

    if df_msd_loglogfits_long is not None:
        msd_fig_all, msd_category_figs, fit_stats = msd_graphs(df_msd, df_msd_loglogfits_long, color_map)
        if cat_provided:
            with open(f'{savefile}_Figures_MSD.html', 'w', encoding='utf-8') as f:
                f.write(msd_fig_all.to_html(full_html=True, include_plotlyjs='cdn'))
                for cat, fig in msd_category_figs.items():
                    f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        else:
            single_fig = next(iter(msd_category_figs.values()), msd_fig_all)
            with open(f'{savefile}_Figures_MSD.html', 'w', encoding='utf-8') as f:
                f.write(single_fig.to_html(full_html=True, include_plotlyjs='cdn'))
        fit_stats_for_summary = fit_stats
    else:
        fit_stats_for_summary = None

    if df_contacts is not None and df_contpercat is not None and not df_contacts.empty and not df_contpercat.empty:
        contacts_figs = contacts_figures(df_contacts, df_contpercat, color_map=color_map)
        with open(f'{savefile}_Figures_Contacts.html', 'w', encoding='utf-8') as f:
            for fig in contacts_figs:
                f.write(fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))

    sumstat_figs = summary_figures(df_sum, fit_stats_for_summary, color_map=color_map)
    with open(f'{savefile}_Figures_Summary-Features.html', 'w', encoding='utf-8') as f:
        for fig in sumstat_figs:
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            f.write(f"<div style='width:95vw;'>{fig_html}</div>")

    return

if __name__ == '__main__':
    mp.set_start_method("spawn")
