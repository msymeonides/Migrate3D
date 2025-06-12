import base64
from dash import Dash, dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from datetime import date
import io
import math
import numpy as np
import os
import pandas as pd
import threading
import traceback

from governor import migrate3D
from graph_all_segments import graph_sorted_segments
from generate_PCA import generate_PCA
from summary_statistics_figures import generate_figures
from shared_state import messages, thread_lock, get_progress, complete_progress_step, init_progress_tracker

# Welcome to Migrate3D version 2.0, released June 2025.
# Please see README.md before running this package
# Migrate3D was developed by Menelaos Symeonides, Emily Mynar, Matthew Kinahan and Jonah Harris
# at the University of Vermont, funded by NIH R56-AI172486 and NIH R01-AI172486 (PI: Markus Thali).
# For more information, see https://github.com/msymeonides/Migrate3D/

# Defaults for tunable parameters can be set here
parameters = {'arrest_limit': 3,    # Arrest limit
              'moving': 4,          # Minimum timepoints
              'contact_length': 12, # Contact length
              'arrested': 0.95,     # Maximum arrest coefficient
              'timelapse': 1,       # Timelapse interval
              'tau_msd': 10,        # Maximum MSD Tau value
              'tau_euclid': 5,      # Maximum Euclidean distance Tau value
              'savefile': '{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
              'multi_track': False, 'interpolate': False, 'verbose': False,
              'contact': False, 'attractors': False, 'generate_figures': False, 'pca_filter': None,
              'infile_tracks': False,
              'object_id_col_name': 'Parent ID', 'time_col_name': "Time", 'x_col_name': 'X Coordinate',
              'y_col_name': 'Y Coordinate', 'z_col_name': 'Z Coordinate', 'object_id_2_col': 'ID',
              'category_col': 'Category',
              }

# Defaults for Attractors tunable parameters can be set here
attract_params = {
        'distance_threshold': 100,  # Maximum distance between attractor and attracted objects
        'approach_ratio': 0.5,  # Ratio of end distance to start distance must be less than this value
        'min_proximity': 20,  # Attracted objects must get at least this close to attractors for at least one timepoint
        'time_persistence': 6,  # Minimum number of consecutive timepoints for a chain to be included in results
        'max_gaps': 4,  # Number of consecutive timepoints of increasing distance allowed before chain is broken
        'allowed_attractor_types': '',  # Object categories allowed to be attractors
        'allowed_attracted_types': '',  # Object categories allowed to be attracted
                }

file_storing = {}
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/', external_stylesheets=[dbc.themes.BOOTSTRAP])
with thread_lock:
    messages.append('Waiting for user input. Load data, adjust parameters and options, and click "Run Migrate3D" to start the analysis.')

formatting_option_map = {
    'multi_track': 'Multitrack',
    'interpolate': 'Interpolate',
    'verbose': 'Verbose',
    'contact': 'Contacts',
    'attractors': 'Attractors',
    'generate_figures': 'Generate Figures'
}
default_formatting_options = [
    v for k, v in formatting_option_map.items() if parameters.get(k, False)
]

app.layout = dbc.Container(
    children=[
        dbc.Row(
            [
                dbc.Col([], width=3),
                dbc.Col(
                    html.Div(
                        html.H1(
                            html.Span("Migrate3D", style={
                                "border": "2px solid #333",
                                "padding": "0.25em 0.75em",
                                "borderRadius": "8px",
                                "display": "inline-block"
                            }),
                            style={"margin": 0, "textAlign": "center"}
                        ),
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "height": "70px"
                        }
                    ),
                    width=6,
                    style={"display": "flex", "justifyContent": "center", "alignItems": "center"}
                ),
                dbc.Col(
                    html.Div(
                        html.Img(
                            src="assets/uvm_asset.jpeg",
                            style={
                                "height": "70px",
                                "width": "auto",
                                "objectFit": "contain"
                            }
                        ),
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "flex-end",
                            "height": "70px"
                        }
                    ),
                    width=3,
                    style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}
                ),
            ],
            style={"height": "90px", "alignItems": "center"}
        ),
        html.H4(
            "Comprehensive motion analysis software package",
            style={"fontWeight": "normal", "color": "#555", "textAlign": "center"}
        ),
        html.H6(
            "Version 2.0, published June 2025. Developed at the University of Vermont by Menelaos Symeonides, Emily Mynar, Matt Kinahan, and Jonah Harris.",
            style={"fontWeight": "normal", "color": "#555", "textAlign": "center"}
        ),
        html.Hr(),
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gridTemplateRows': 'auto auto',
                'gap': '1px',
                'alignItems': 'start'
            },
            children=[
                html.Div(
                    "Segments input file should be a .csv with object ID, time, X, Y, and Z coordinates. "
                    "Please ensure that column headers are in the first row of the .csv file input.",
                    style={'marginBottom': '10px', 'gridColumn': '1'}
                ),
                html.Div(
                    'Categories input files should be a .csv with object ID and object category. Please ensure that column headers are in the '
                    'first row of the .csv file input. A Categories file is optional, but required for certain analyses.',
                    style={'marginBottom': '10px', 'gridColumn': '2'}
                ),
                html.Div(
                    dcc.Upload(
                        id='segments_upload',
                        children='Segments .csv file (drag and drop, or click to select a file):',
                        style={
                            'width': '100%',
                            'minWidth': '500px',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '0 auto'
                        }
                    ),
                    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gridColumn': '1'}
                ),
                html.Div(
                    dcc.Upload(
                        id='category_upload',
                        children='Categories .csv file (drag and drop, or click to select a file):',
                        style={
                            'width': '100%',
                            'minWidth': '500px',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '0 auto'
                        }
                    ),
                    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gridColumn': '2'}
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            id='identifier_divs',
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gridTemplateRows': '1fr',
                'gap': '32px',
                'alignItems': 'start'
            },
            children=[
                html.Div(
                    id='column_populate_segments',
                    children=[
                        html.H4('Column Identifiers for Segments file:'),
                        dcc.Dropdown(id='parent_id', placeholder='Select object ID column'),
                        dcc.Dropdown(id='time_formatting', placeholder='Select time column'),
                        dcc.Dropdown(id='x_axis', placeholder='Select X coordinate column'),
                        dcc.Dropdown(id='y_axis', placeholder='Select Y coordinate column'),
                        dcc.Dropdown(id='z_axis', placeholder='Select Z coordinate column (leave blank for 2D data)'),
                    ],
                    style={'width': '70%', 'margin': '0 auto'}
                ),
                html.Div(
                    id='Categories_dropdown',
                    children=[
                        html.H4('Column Identifiers for Categories file (if used):'),
                        dcc.Dropdown(id='parent_id2', placeholder='Select object ID column'),
                        dcc.Dropdown(id='category_col', placeholder='Select Category column'),
                    ],
                    style={'width': '70%', 'margin': '0 auto'}
                ),
            ]
        ),
                html.Hr(),
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': '1fr 1fr',
                        'gridTemplateRows': '1fr',
                        'gap': '32px',
                        'alignItems': 'start'
                    },
                    children=[
                        html.Div(
                            id='Parameters',
                            children=[
                                html.H4('Tunable parameters'),
                                html.Div(html.H6('Set these values as desired (refer to README file). You can change the default values by editing the "parameters" section at the top of main.py.'),style={'marginLeft': '2%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Arrest limit (displacements below this value will be considered noise and not real movement)']),
                                dcc.Input(id='arrest_limit', value=parameters['arrest_limit']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Minimum timepoints (objects must be moving for at least this many timepoints to be fully analyzed)']),
                                dcc.Input(id='moving', value=parameters['moving']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Contact length (if the distance between two objects is less than this, they will be considered to be in contact)']),
                                dcc.Input(id='contact_length', value=parameters['contact_length']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Maximum arrest coefficient (objects with arrest coefficient above this value will be considered arrested)']),
                                dcc.Input(id='arrested', value=parameters['arrested']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.H4('Autodetected parameters'),
                                html.Div(html.H6('The timelapse interval, maximum MSD Tau value, and maximum Euclidean distance Tau value will be autodetected from the segments file immediately after it is loaded. Any value can be manually entered in case these are not detected correctly.'),style={'marginLeft': '2%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Timelapse interval']),
                                dcc.Input(id='Timelapse', value=parameters['timelapse'], placeholder='Enter timelapse interval'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Maximum MSD Tau value (Default: The modal number of timepoints in the dataset)']),
                                dcc.Input(id='tau_msd', value=parameters['tau_msd'], placeholder='Enter MSD Tau'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Maximum Euclidean distance Tau value (Default: Half of the Maximum MSD Tau Value)']),
                                dcc.Input(id='tau_euclid', value=parameters['tau_euclid'], placeholder='Enter Euclidean Tau'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.H4(children=['Formatting options']),
                                dcc.Checklist(
                                    id='formatting_options',
                                    options=[
                                        {'label': ' Multitracking (if an object ID is represented by multiple segments at a given timepoint, they will be spatially averaged into one segment)',
                                            'value': 'Multitrack'},
                                        {'label': ' Interpolation (if an object ID is missing a timepoint, that timepoint will be inferred by simple linear interpolation and inserted)',
                                            'value': 'Interpolate'},
                                        {'label': ' Verbose (includes the results of all calculations in the output file)',
                                            'value': 'Verbose'},
                                        {'label': ' Contacts (identifies contacts between objects)',
                                            'value': 'Contacts'},
                                        {'label': ' Attractors (identifies instances where an object is attracting other objects towards it)',
                                            'value': 'Attractors'},
                                        {'label': ' Generate Figures (creates figures for summary statistics and PCA)',
                                            'value': 'Generate Figures'}
                                    ],
                                    value=default_formatting_options,
                                    inputStyle={'width': '30px', 'height': '30px', 'marginRight': '5px',
                                                'marginBottom': '5px', 'marginTop': '5px'},
                                    labelStyle={'display': 'block', 'alignItems': 'center', 'marginLeft': '5%'}
                                ),
                                html.Br(),
                                html.Button('Open Attractors Parameters', id='toggle-attractor-settings', n_clicks=0,
                                                style={
                                                    'margin': '0 auto',
                                                    'display': 'block'
                                                }
                                            ),
                                html.Hr(),
                                html.H6(
                                    children='Enter subset of categories to be used during PCA and XGBoost analysis (separate category IDs with a space). Leave blank to use all categories.'
                                ),
                                dcc.Input(id='PCA_filter', placeholder='e.g. 4 5 6', style={'width': '250px'}),
                                html.Hr(),
                                html.H6(children=['Save results as:']),
                                dcc.Input(
                                    id='save_file',
                                    placeholder=parameters['savefile'],
                                    value=parameters['savefile'],
                                    style={'width': '250px'}
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'flexDirection': 'column'
                            },
                            children=[
                                html.Div([
                                    dbc.Progress(
                                        id="progress-bar",
                                        value=0,
                                        striped=True,
                                        animated=True,
                                        className="mb-3",
                                        color='success'
                                    ),
                                    html.H4("Current status", style={'marginBottom': '10px'}),
                                    html.Div(
                                        id='alert_box',
                                        style={
                                            'minWidth': '200px',
                                            'width': '100%',
                                            'maxHeight': '1000px',
                                            'overflowY': 'auto',
                                            'boxShadow': '0 2px 8px rgba(0,0,0,0.2)',
                                            'borderRadius': '8px',
                                            'padding': '16px',
                                            'zIndex': 2000,
                                            'border': '1px solid #ccc',
                                            'flex': '1'
                                        }
                                    )
                                ]),
                                html.Button(
                                    'Run Migrate3D',
                                    id='Run_migrate',
                                    n_clicks=0,
                                    disabled=False,
                                    style={
                                        'fontSize': '2rem',
                                        'padding': '20px 40px',
                                        'width': '40%',
                                        'alignSelf': 'center',
                                        'marginTop': '50px'
                                    }
                                ),
                                html.Div(
                                    id='attractor-settings-div',
                                    style={'display': 'none'},
                                    children=[
                                        html.Div([
                                            html.H4(children=['Attractors parameters']),
                                            html.H6(
                                                'Distance threshold (Maximum distance between attractor and attracted objects to be considered)'),
                                            dcc.Input(id='dist_thr', type='number', step=0.1,
                                                      value=attract_params['distance_threshold']),
                                            html.Hr(),
                                            html.H6(
                                                'Approach ratio (Ratio of end distance to start distance must be less than this value)'),
                                            dcc.Input(id='approach_ratio', type='number', step=0.1,
                                                      value=attract_params['approach_ratio']),
                                            html.Hr(),
                                            html.H6(
                                                'Min. proximity (Attracted objects must get at least this close to attractors for at least one timepoint)'),
                                            dcc.Input(id='min_proximity', type='number', step=0.1,
                                                      value=attract_params['min_proximity']),
                                            html.Hr(),
                                            html.H6(
                                                'Time persistence (Minimum number of consecutive timepoints for a chain to be included in results)'),
                                            dcc.Input(id='time_persistence', type='number', step=1,
                                                      value=attract_params['time_persistence']),
                                            html.Hr(),
                                            html.H6(
                                                'Max. gaps (Number of consecutive timepoints of increasing distance allowed before chain is broken)'),
                                            dcc.Input(id='max_gaps', type='number', step=1,
                                                      value=attract_params['max_gaps']),
                                            html.Hr(),
                                            html.H6(
                                                'Object categories allowed to be attractors (space-separated list of category IDs)'),
                                            dcc.Input(id='allowed_attractors', type='text', value=attract_params['allowed_attractor_types']),
                                            html.Hr(),
                                            html.H6(
                                                'Object categories allowed to be attracted (space-separated list of category IDs)'),
                                            dcc.Input(id='allowed_attracted', type='text', value=attract_params['allowed_attracted_types']),
                                        ], style={'padding': '10px', 'border': '1px solid #ccc','marginTop': '400px', 'width': '70%'})
                                    ]
                                )
                            ]
                        ),
                    dcc.Interval(id='progress-interval', interval=1000, n_intervals=0),
                    ]
                ),
        html.Br(),
        html.Br(),
        html.Div(id='dummy', style={'display': 'none'})
    ],
    className="body",
    fluid=True
)

def run_migrate_thread(args):
    (parent_id, time_for, x_for, y_for, z_for, timelapse, arrest_limit, moving,
     contact_length, arrested, tau_msd, tau_euclid, formatting_options, savefile,
     segments_file_name, tracks_file, parent_id2, category_col_name,
     parameters, pca_filter, attract_params) = args

    if tracks_file is not None and 'Enter your category .csv file here' not in str(tracks_file):
         parameters['infile_tracks'] = True
    else:
         parameters['infile_tracks'] = False

    optional_flags = {
        "pca_xgb": True if parameters.get("infile_tracks", False) else False,
        "contacts": True if formatting_options and "Contacts" in formatting_options else False,
        "attractors": True if formatting_options and "Attractors" in formatting_options else False,
        "generate_figures": True if formatting_options and "Generate Figures" in formatting_options else False
    }
    init_progress_tracker(optional_flags)

    try:
        df_segments, df_sum, df_pca = migrate3D(
            parent_id, time_for, x_for, y_for, z_for, float(timelapse),
            float(arrest_limit), int(moving), int(contact_length), float(arrested),
            int(tau_msd), int(tau_euclid), formatting_options, savefile,
            segments_file_name, tracks_file, parent_id2, category_col_name,
            parameters, pca_filter, attract_params)

        if parameters.get('generate_figures', False):
            fig_segments = graph_sorted_segments(df_segments, df_sum, parameters['infile_tracks'], savefile)
            sum_fig = generate_figures(df_sum)
            sum_fig.append(fig_segments)
            if df_pca is not None:
                fig_pca = generate_PCA(df_pca)
                sum_fig.append(fig_pca)
                with open(f'{savefile}_figures.html', 'a') as f:
                    for i in sum_fig:
                        f.write(i.to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                with open(f'{savefile}_Figures.html', 'a') as f:
                    for i in sum_fig:
                        f.write(i.to_html(full_html=False, include_plotlyjs='cdn'))
        complete_progress_step("Generate Figures")

        with thread_lock:
            messages.append("You may close the Anaconda prompt and the GUI browser tab, or just terminate the Python process.")

    except Exception as e:
        with thread_lock:
            messages.append(f"Error: {str(e)}")
        traceback.print_exc()

@app.callback(
    Output('dummy', 'children'),
    Input('parent_id', 'value'),
    Input('time_formatting', 'value'),
    Input('x_axis', 'value'),
    Input('y_axis', 'value'),
    Input('z_axis', 'value'),
    Input('Timelapse', 'value'),
    Input('arrest_limit', 'value'),
    Input('moving', 'value'),
    Input('contact_length', 'value'),
    Input('arrested', 'value'),
    Input('tau_msd', 'value'),
    Input('tau_euclid', 'value'),
    Input('formatting_options', 'value'),
    Input('save_file', 'value'),
    Input('segments_upload', 'contents'),
    State('segments_upload', 'filename'),
    Input('category_upload', 'contents'),
    State('category_upload', 'filename'),
    Input('parent_id2', 'value'),
    Input('category_col', 'value'),
    Input('PCA_filter', 'value'),
    Input('Run_migrate', 'n_clicks'),
    Input('dist_thr', 'value'),
    Input('approach_ratio', 'value'),
    Input('min_proximity', 'value'),
    Input('time_persistence', 'value'),
    Input('max_gaps', 'value'),
    Input('allowed_attractors', 'value'),
    Input('allowed_attracted', 'value'),
    prevent_initial_call=True
)
def run_migrate(*vals):
    (parent_id, time_for, x_for, y_for, z_for,
     timelapse, arrest_limit, moving, contact_length, arrested,
     tau_msd, tau_euclid, formatting_options, savefile,
     segments_contents, segments_file,
     category_contents, category_file,
     parent_id2, category_col_name, pca_filter,
     run_clicks, dist_thr, approach_ratio, min_proximity,
     time_persistence, max_gaps,
     allowed_attractors, allowed_attracted) = vals

    if run_clicks == 0:
        raise exceptions.PreventUpdate

    if isinstance(pca_filter, str):
        pca_filter = pca_filter.strip()
        pca_filter = pca_filter.split() if pca_filter != "" else None
    else:
        pca_filter = None

    if isinstance(allowed_attractors, str) and allowed_attractors.strip():
        allowed_attractor_types = [int(i) for i in allowed_attractors.split()]
    else:
        allowed_attractor_types = []

    if isinstance(allowed_attracted, str) and allowed_attracted.strip():
        allowed_attracted_types = [int(i) for i in allowed_attracted.split()]
    else:
        allowed_attracted_types = []

    attract_params = {
        'distance_threshold': float(dist_thr),
        'approach_ratio': float(approach_ratio),
        'min_proximity': float(min_proximity),
        'time_persistence': int(time_persistence),
        'max_gaps': int(max_gaps),
        'allowed_attractor_types': allowed_attractor_types,
        'allowed_attracted_types': allowed_attracted_types
    }

    args = [
        parent_id, time_for, x_for, y_for, z_for,
        float(timelapse), float(arrest_limit), int(moving),
        int(contact_length), float(arrested), int(tau_msd),
        int(tau_euclid), formatting_options, savefile,
        segments_file, category_file,
        parent_id2, category_col_name,
        parameters, pca_filter, attract_params
    ]
    thread = threading.Thread(target=run_migrate_thread, args=(args,))
    thread.start()
    return None

@app.callback(
    Output('segments_upload', 'children'),
    Output('parent_id', 'options'), Output('parent_id', 'value'),
    Output('time_formatting', 'options'), Output('time_formatting', 'value'),
    Output('x_axis', 'options'), Output('x_axis', 'value'),
    Output('y_axis', 'options'), Output('y_axis', 'value'),
    Output('z_axis', 'options'), Output('z_axis', 'value'),
    Input('segments_upload', 'contents'),
    State('segments_upload', 'filename'),
    prevent_initial_call=True
)
def get_segments_file(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception:
        return html.Div(['There was an error processing this file.']), [], None, [], None, [], None, [], None, [], None
    file_storing['Segments'] = df
    def guess_column(df, keywords):
        for col in df.columns:
            if any(key.lower() in col.lower() for key in keywords):
                return col
        return None
    id_guess = guess_column(df, ['id', 'track', 'parent', 'cell', 'object'])
    time_guess = guess_column(df, ['time', 'frame'])
    x_guess = guess_column(df, ['x'])
    y_guess = guess_column(df, ['y'])
    z_guess = guess_column(df, ['z'])
    options = list(df.columns)
    return (
        filename,
        options, id_guess,
        options, time_guess,
        options, x_guess,
        options, y_guess,
        options, z_guess
    )

@app.callback(
    Output('category_upload', 'children'),
    Output('parent_id2', 'options'), Output('parent_id2', 'value'),
    Output('category_col', 'options'), Output('category_col', 'value'),
    Input('category_upload', 'contents'),
    State('category_upload', 'filename'),
    prevent_initial_call=True
)
def get_category_file(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception:
        return html.Div(['There was an error processing this file.']), [], None, [], None
    file_storing['Categories'] = df
    def guess_column(df, keywords):
        for col in df.columns:
            if any(key.lower() in col.lower() for key in keywords):
                return col
        return None
    id_guess = guess_column(df, ['id', 'track', 'parent', 'cell', 'object'])
    category_guess = guess_column(df, ['category', 'label', 'type', 'code'])
    options = list(df.columns)
    return filename, options, id_guess, options, category_guess

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'color'),
    Input("progress-interval", 'n_intervals'),
)
def update_pbar(n):
    progress = get_progress()
    if progress == 100:
        return progress, 'warning'
    else:
        return progress, 'success'

@app.callback(
    Output('alert_box', 'children'),
    Input('progress-interval', 'n_intervals')
)
def update_alert_box(n):
    with thread_lock:
        return html.Div([
            html.Pre('\n'.join(messages), style={'whiteSpace': 'pre-wrap', 'margin': 0})
        ])

@app.callback(
    Output('Timelapse', 'value'),
    Output('Timelapse', 'placeholder'),
    Output('tau_msd', 'value'),
    Output('tau_msd', 'placeholder'),
    Output('tau_euclid', 'value'),
    Output('tau_euclid', 'placeholder'),
    Input('segments_upload', 'contents'),
    Input('parent_id', 'value'),
    Input('time_formatting', 'value'),
)
def update_time_and_tau(contents, id_col, time_col):
    if contents is None or id_col is None or time_col is None:
        raise exceptions.PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )
    if id_col not in df.columns or time_col not in df.columns:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )
    try:
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    except Exception:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )

    intervals = []
    for _, group in df.groupby(id_col):
        times = group[time_col].dropna().sort_values().unique()
        if len(times) > 1:
            diffs = pd.Series(times).diff().dropna()
            intervals.extend(diffs)
    if not intervals:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )
    rounded_intervals = pd.Series(intervals).round(6)
    detected_interval = rounded_intervals.mode()
    if detected_interval.empty:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )
    detected_interval = float(detected_interval.iloc[0])
    counts = df.groupby(id_col)[time_col].nunique()
    if len(counts) == 0:
        return (
            None, "Enter timelapse interval manually",
            None, "Enter MSD Tau manually",
            None, "Enter Euclidean Tau manually"
        )
    msd_tau = int(np.percentile(counts, 80))
    euclid_tau = int(math.ceil(msd_tau / 2))
    return detected_interval, "", msd_tau, "", euclid_tau, ""

@app.callback(
    Output('attractor-settings-div', 'style'),
    Output('toggle-attractor-settings', 'children'),
    Input('toggle-attractor-settings', 'n_clicks'),
)
def toggle_attractor_settings(n_clicks):
    is_open = bool(n_clicks and n_clicks % 2)
    style = {'display': 'block'} if is_open else {'display': 'none'}
    label = 'Close Attractors Parameters' if is_open else 'Open Attractors Parameters'
    return style, label

@app.callback(
    Output('Run_migrate', 'children'),
    Output('Run_migrate', 'style'),
    Output('Run_migrate', 'disabled'),
    Input('Run_migrate', 'n_clicks'),
    Input('progress-bar', 'value'),
    prevent_initial_call=False
)
def update_run_button(n_clicks, progress):
    if progress == 100 and n_clicks and n_clicks > 0:
        return (
            "Done!",
            {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '50px',
                'backgroundColor': '#FFD700',
                'color': 'black',
                'border': 'none'
            },
            True
        )
    elif n_clicks and n_clicks > 0:
        return (
            "Running!",
            {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '50px',
                'backgroundColor': '#7bb77b',
                'color': 'white',
                'border': 'none'
            },
            True
        )
    else:
        return (
            "Run Migrate3D",
            {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '50px',
                'backgroundColor': '#e0e0e0',
                'color': 'black',
                'border': 'none'
            },
            False
        )

if __name__ == '__main__':
    app.run(port=5555, debug=True)
