from dash import Dash, dcc, html, Input, Output, State, exceptions, callback_context
import dash_bootstrap_components as dbc
import dash_uploader as du
from datetime import date
import numpy as np
import os
import pandas as pd
import shutil
import threading
import traceback

from governor import migrate3D
from shared_state import messages, thread_lock, get_progress, init_progress_tracker, is_aborted

# Welcome to Migrate3D version 2.5, under construction!
# Please see README.md before running this package.
# Migrate3D was developed by Menelaos Symeonides, Emily Mynar, Matthew Kinahan and Jonah Harris
# at the University of Vermont, funded by NIH R56-AI172486 and NIH R01-AI172486 (PI: Markus Thali).
# For more information, see https://github.com/msymeonides/Migrate3D/

# Defaults for tunable parameters can be set here
parameters = {'arrest_limit': 3,    # Arrest limit
              'moving': 4,          # Minimum timepoints
              'contact_length': 12, # Contact length
              'arrested': 0.95,     # Maximum arrest coefficient
              'min_maxeuclid': 0,   # Minimum Max. Euclidean filter
              'timelapse': 1,       # Timelapse interval
              'tau': 10,            # Maximum Tau value
              'savefile': '{:%Y-%m-%d}'.format(date.today()) + '_Migrate3D',
              'multi_track': False,
              'interpolate': False,
              'verbose': False,
              'contact': False,
              'contact_div_filter': False,
              'attractors': False,
              'helicity': False,
              'generate_figures': False,
              'infile_categories': False, 'object_id_col_name': 'Parent ID', 'time_col_name': "Time",
              'x_col_name': 'X Coordinate', 'y_col_name': 'Y Coordinate', 'z_col_name': 'Z Coordinate',
              'object_id_2_col': 'ID', 'category_col': 'Category',
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

UPLOAD_FOLDER_ROOT = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER_ROOT, exist_ok=True)
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/', external_stylesheets=[dbc.themes.BOOTSTRAP])
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

with thread_lock:
    messages.append('Waiting for user input. Load data, adjust parameters and options, and click "Run Migrate3D" to start the analysis.')

option_map = {
    'multi_track': 'Multitrack',
    'interpolate': 'Interpolate',
    'verbose': 'Verbose',
    'contact': 'Contacts',
    'contact_div_filter': 'ContactDivFilter',
    'attractors': 'Attractors',
    'helicity': 'Helicity',
    'generate_figures': 'Generate Figures'
}
default_options = [
    v for k, v in option_map.items() if parameters.get(k, False)
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
            "Comprehensive motion analysis made easy",
            style={"fontWeight": "normal", "color": "#555", "textAlign": "center"}
        ),
        html.H6(
            "Version 2.5, published July 2025. Developed at the University of Vermont by Menelaos Symeonides, Emily Mynar, Matt Kinahan, and Jonah Harris.",
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
                    [
                    du.Upload(
                        id='segments_upload',
                        text='Segments .csv file (drag and drop, or click to select a file):',
                        text_completed='Uploaded: ',
                        cancel_button=False,
                        pause_button=False,
                        filetypes=['csv'],
                        max_file_size=5120,
                        default_style={
                            'width': '50%',
                            'minWidth': '250px',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'fontSize': '18px',
                            'margin': '1 auto'
                        }
                    ),
                    html.Div(
                        id='segments_status',
                        style={
                            'marginTop': '5px',
                            'textAlign': 'center',
                            'fontSize': '16px',
                            'color': '#666',
                            'fontWeight': 'bolder'
                        }
                    )
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gridColumn': '1'}
                ),
                html.Div(
                    [
                    du.Upload(
                        id='category_upload',
                        text='Categories .csv file (drag and drop, or click to select a file):',
                        text_completed='Uploaded: ',
                        cancel_button=False,
                        pause_button=False,
                        filetypes=['csv'],
                        max_file_size=5120,
                        default_style={
                            'width': '50%',
                            'minWidth': '250px',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'fontSize': '18px',
                            'margin': '1 auto'
                        }
                    ),
                    html.Div(
                        id='categories_status',
                        style={
                            'marginTop': '5px',
                            'textAlign': 'center',
                            'fontSize': '16px',
                            'color': '#666',
                            'fontWeight': 'bolder'
                        }
                    )
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gridColumn': '2'}
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
                                html.Div([html.H6(children=['Minimum timepoints (objects must be moving for at least this many timepoints for its summary velocity/acceleration features to be reported)']),
                                dcc.Input(id='moving', value=parameters['moving']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Contact length (if the distance between two objects is less than this, they will be considered to be in contact)']),
                                dcc.Input(id='contact_length', value=parameters['contact_length']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Maximum arrest coefficient (objects with arrest coefficient above this value will be considered arrested)']),
                                dcc.Input(id='arrested', value=parameters['arrested']),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Minimum Max. Euclidean (objects whose value of Max. Euclidean is below this value will be removed from the dataset entirely)']),
                                dcc.Input(id='min_maxeuclid', value=parameters['min_maxeuclid'], placeholder='Enter minimum Max. Euclidean value'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.H4('Autodetected parameters'),
                                html.Div(html.H6('The timelapse interval and maximum tau value will be autodetected from the segments file immediately after it is loaded. Any value can be manually entered in case these are not detected correctly.'),style={'marginLeft': '2%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Timelapse interval']),
                                dcc.Input(id='Timelapse', value=parameters['timelapse'], placeholder='Enter timelapse interval'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.Div([html.H6(children=['Maximum Tau value (Default: The maximum number of timepoints in the dataset)']),
                                dcc.Input(id='tau', value=parameters['tau'], placeholder='Enter max. tau value'),],style={'marginLeft': '5%'}),
                                html.Hr(),
                                html.H4(children=['Options']),
                                dcc.Checklist(
                                    id='options',
                                    options=[
                                        {'label': ' Multitracking (if an object ID is represented by multiple segments at a given timepoint, they will be spatially averaged into one segment)',
                                            'value': 'Multitrack'},
                                        {'label': ' Interpolation (if an object ID is missing a timepoint, that timepoint will be inferred by simple linear interpolation and inserted)',
                                            'value': 'Interpolate'},
                                        {'label': ' Verbose (enables more detailed outputs, including step-wise calculations and intermediate datasets from ML processing steps)',
                                            'value': 'Verbose'},
                                        {'label': ' Contacts (identifies contacts between objects)',
                                            'value': 'Contacts'},
                                        {'label': ' Dividing Contacts Filter (filters out contacts between objects resulting from cell division)',
                                            'value': 'ContactDivFilter'},
                                        {'label': ' Attractors (identifies instances where an object is attracting other objects towards it)',
                                            'value': 'Attractors'},
                                        {'label': ' Helicity (calculates additional summary features; use for tracks which exhibit helical motion)',
                                            'value': 'Helicity'},
                                        {'label': ' Generate Figures (creates figures for summary features, PCA, and MSD)',
                                            'value': 'Generate Figures'}
                                    ],
                                    value=default_options,
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
                                        'marginTop': '50px',
                                        'backgroundColor': '#e0e0e0',
                                        'color': 'black',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
                                        'cursor': 'pointer'
                                    }
                                ),
                                html.Div(style={'height': '150px'}),
                                html.Div(
                                    id='replicate-analysis-container',
                                    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Pre(
                                                    'Use the button below if you are done analyzing all the replicates of your experiment. This assumes that '
                                                    'they were all run with the same parameters, that the results files are all still present in the Migrate3D '
                                                    'working folder, and that the filenames are identical and end in "-r#" where # is the index number of the '
                                                    'replicate.\n\n'
                                                    'In the "Save results as:" box at the bottom left, enter the filename prefix that is common to all of the '
                                                    'replicates belonging to this experiment, then click the button below to perform the replicate analysis.',
                                                    style={
                                                        'whiteSpace': 'pre-wrap',
                                                        'margin': 0
                                                    }
                                                )
                                            ],
                                            style={
                                                'width': '100%',
                                                'alignSelf': 'center',
                                                'minWidth': '200px',
                                                'maxHeight': '1000px',
                                                'overflowY': 'auto',
                                                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)',
                                                'borderRadius': '8px',
                                                'padding': '16px',
                                                'zIndex': 2000,
                                                'border': '1px solid #ccc'
                                            }
                                        ),
                                        html.Button(
                                            'Perform replicate analysis',
                                            id='replicate-analysis-button',
                                            n_clicks=0,
                                            disabled=False,
                                            style={
                                                'fontSize': '2rem',
                                                'padding': '20px 40px',
                                                'width': '40%',
                                                'alignSelf': 'center',
                                                'marginTop': '20px',
                                                'backgroundColor': '#e0e0e0',
                                                'color': 'black',
                                                'border': 'none',
                                                'borderRadius': '5px',
                                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
                                                'cursor': 'pointer'
                                            }
                                        )
                                    ]
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
        html.Div(id='dummy', style={'display': 'none'}),
    ],
    className="body",
    fluid=True
)

def run_migrate_thread(args):
    (parent_id, time_for, x_for, y_for, z_for, timelapse, arrest_limit, moving,
     contact_length, arrested, min_maxeuclid, tau, options, savefile,
     segments_file_name, categories_file, segments_filename, categories_filename, parent_id2, category_col_name,
     parameters, pca_filter, attract_params) = args

    if categories_file is not None:
        parameters['object_id_2_col'] = parent_id2
        parameters['category_col'] = category_col_name
        parameters['infile_categories'] = True
    else:
        parameters['object_id_2_col'] = parent_id
        parameters['category_col'] = 'Category'
        parameters['infile_categories'] = False

    parameters['contact_div_filter'] = (
            options is not None and "ContactDivFilter" in options
    )
    optional_flags = {
        "pca_xgb": parameters.get("infile_categories", False),
        "contacts": True if options and "Contacts" in options else False,
        "attractors": True if options and "Attractors" in options else False,
        "helicity": True if options and "Helicity" in options else False,
        "generate_figures": True if options and "Generate Figures" in options else False
    }
    init_progress_tracker(optional_flags)

    try:
        df_segments, df_sum, df_pca = migrate3D(
            parent_id, time_for, x_for, y_for, z_for, float(timelapse),
            float(arrest_limit), int(moving), int(contact_length), float(arrested),
            float(min_maxeuclid), int(tau), options, savefile,
            segments_file_name, categories_file, segments_filename, categories_filename, parameters, pca_filter, attract_params)

        with thread_lock:
            messages.append("You may close the GUI browser tab and terminate the Python process.")
        print()
        print("Migrate3D run completed. You may close the GUI browser tab and terminate the Python process.")

    except Exception as e:
        with thread_lock:
            messages.append(f"Error: {str(e)}")
        traceback.print_exc()

    finally:
        try:
            shutil.rmtree(UPLOAD_FOLDER_ROOT)
            os.makedirs(UPLOAD_FOLDER_ROOT, exist_ok=True)
        except Exception as cleanup_error:
            print(f"Error cleaning uploads folder: {cleanup_error}")

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
    Input('min_maxeuclid', 'value'),
    Input('tau', 'value'),
    Input('options', 'value'),
    Input('save_file', 'value'),
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
     min_maxeuclid, tau, options, savefile,
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

    segments_input = file_storing.get('segments_dataframe', None)
    categories_input = file_storing.get('categories_dataframe', None)
    segments_filename = file_storing.get('segments_filename', None)
    categories_filename = file_storing.get('categories_filename', None)

    args = [
        parent_id, time_for, x_for, y_for, z_for,
        float(timelapse), float(arrest_limit), int(moving),
        float(contact_length), float(arrested), float(min_maxeuclid), int(tau),
        options, savefile,
        segments_input,
        categories_input,
        segments_filename,
        categories_filename,
        parent_id2, category_col_name,
        parameters, pca_filter, attract_params
    ]
    thread = threading.Thread(target=run_migrate_thread, args=(args,))
    thread.start()
    return None

@app.callback(
    Output('parent_id', 'options'), Output('parent_id', 'value'),
    Output('time_formatting', 'options'), Output('time_formatting', 'value'),
    Output('x_axis', 'options'), Output('x_axis', 'value'),
    Output('y_axis', 'options'), Output('y_axis', 'value'),
    Output('z_axis', 'options'), Output('z_axis', 'value'),
    Output('Timelapse', 'value'), Output('tau', 'value'),
    Input('segments_upload', 'isCompleted'),
    State('segments_upload', 'fileNames'),
    State('segments_upload', 'upload_id'),
    prevent_initial_call=True
)
def get_segments_file(isCompleted, fileNames, upload_id):
    if not isCompleted or not fileNames or not upload_id:
        raise exceptions.PreventUpdate

    filename = fileNames[0]
    file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)

    if not os.path.exists(file_path):
        return [], None, [], None, [], None, [], None, [], None, None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [], None, [], None, [], None, [], None, [], None, None, None

    file_storing['segments_dataframe'] = df
    file_storing['segments_filename'] = filename

    def guess_column(df, keywords):
        for col in df.columns:
            col_lower = col.lower()
            if any(col_lower == key.lower() for key in keywords):
                return col
        return None

    id_guess = guess_column(df, ['id', 'track', 'track id', 'trackid', 'cell', 'cell id', 'cellid', 'object',
                                 'object id', 'objectid','parent', 'parent id', 'parentid'])
    time_guess = guess_column(df, ['t', 'time', 'frame'])
    x_guess = guess_column(df, ['x', 'x coordinate', 'x position', 'coordinate x', 'position x'])
    y_guess = guess_column(df, ['y', 'y coordinate', 'y position', 'coordinate y', 'position y'])
    z_guess = guess_column(df, ['z', 'z coordinate', 'z position', 'coordinate z', 'position z'])
    options = list(df.columns)

    detected_timelapse = None
    detected_tau = None

    if id_guess and time_guess and id_guess in df.columns and time_guess in df.columns:
        try:
            df_copy = df.copy()
            df_copy[time_guess] = pd.to_numeric(df_copy[time_guess], errors='coerce')

            intervals = []
            for _, group in df_copy.groupby(id_guess):
                times = group[time_guess].dropna().sort_values().unique()
                if len(times) > 1:
                    diffs = pd.Series(times).diff().dropna()
                    intervals.extend(diffs)

            if intervals:
                rounded_intervals = pd.Series(intervals).round(6)
                detected_interval = rounded_intervals.mode()
                if not detected_interval.empty:
                    detected_timelapse = float(detected_interval.iloc[0])

            counts = df_copy.groupby(id_guess)[time_guess].nunique()
            if len(counts) > 0:
                detected_tau = int(np.max(counts))

        except Exception:
            pass

    return (
        options, id_guess,
        options, time_guess,
        options, x_guess,
        options, y_guess,
        options, z_guess,
        detected_timelapse, detected_tau
    )

@app.callback(
    Output('parent_id2', 'options'), Output('parent_id2', 'value'),
    Output('category_col', 'options'), Output('category_col', 'value'),
    Input('category_upload', 'isCompleted'),
    State('category_upload', 'fileNames'),
    State('category_upload', 'upload_id'),
    prevent_initial_call=True
)
def get_category_file(isCompleted, fileNames, upload_id):
    if not isCompleted or not fileNames or not upload_id:
        raise exceptions.PreventUpdate

    filename = fileNames[0]
    file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)

    if not os.path.exists(file_path):
        return [], None, [], None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [], None, [], None

    file_storing['categories_dataframe'] = df
    file_storing['categories_filename'] = filename

    def guess_column(df, keywords):
        for col in df.columns:
            col_lower = col.lower()
            if any(col_lower == key.lower() for key in keywords):
                return col
        return None

    id_guess = guess_column(df, ['id', 'track', 'track id', 'trackid', 'cell', 'cell id', 'cellid', 'object',
                                 'object id', 'objectid','parent', 'parent id', 'parentid'])
    category_guess = guess_column(df, ['category', 'label', 'type', 'code'])
    options = list(df.columns)

    return options, id_guess, options, category_guess

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'color'),
    Input('progress-interval', 'n_intervals')
)
def update_progress_bar(n):
    progress = get_progress()
    if progress == 100:
        return progress, 'danger' if is_aborted() else 'success'
    else:
        return progress, 'info'

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
    [
        Output('Run_migrate', 'children'),
        Output('Run_migrate', 'style'),
        Output('Run_migrate', 'disabled'),
        Output('segments_upload', 'disabled'),
        Output('category_upload', 'disabled'),
        Output('parent_id', 'disabled'),
        Output('time_formatting', 'disabled'),
        Output('x_axis', 'disabled'),
        Output('y_axis', 'disabled'),
        Output('z_axis', 'disabled'),
        Output('parent_id2', 'disabled'),
        Output('category_col', 'disabled'),
        Output('arrest_limit', 'disabled'),
        Output('moving', 'disabled'),
        Output('contact_length', 'disabled'),
        Output('arrested', 'disabled'),
        Output('min_maxeuclid', 'disabled'),
        Output('Timelapse', 'disabled'),
        Output('tau', 'disabled'),
        Output('PCA_filter', 'disabled'),
        Output('save_file', 'disabled'),
        Output('dist_thr', 'disabled'),
        Output('approach_ratio', 'disabled'),
        Output('min_proximity', 'disabled'),
        Output('time_persistence', 'disabled'),
        Output('max_gaps', 'disabled'),
        Output('allowed_attractors', 'disabled'),
        Output('allowed_attracted', 'disabled'),
        Output('options', 'style'),
    ],
    [Input('Run_migrate', 'n_clicks'), Input('progress-bar', 'value')],
    prevent_initial_call=False
)
def update_run_and_freeze(n_clicks, progress):
    if progress == 100 and n_clicks and n_clicks > 0:
        if is_aborted():
            btn_text = "Aborted"
            btn_style = {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '50px',
                'backgroundColor': '#dc3545',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
                'cursor': 'pointer'
            }
        else:
            btn_text = "Done!"
            btn_style = {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '50px',
                'backgroundColor': '#FFD700',
                'color': 'black',
                'border': 'none',
                'borderRadius': '5px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
                'cursor': 'pointer'
            }
        btn_disabled = True
    elif n_clicks and n_clicks > 0:
        btn_text = "Running!"
        btn_style = {
            'fontSize': '2rem',
            'padding': '20px 40px',
            'width': '40%',
            'alignSelf': 'center',
            'marginTop': '50px',
            'backgroundColor': '#7bb77b',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
            'cursor': 'pointer'
        }
        btn_disabled = True
    else:
        btn_text = "Run Migrate3D"
        btn_style = {
            'fontSize': '2rem',
            'padding': '20px 40px',
            'width': '40%',
            'alignSelf': 'center',
            'marginTop': '50px',
            'backgroundColor': '#e0e0e0',
            'color': 'black',
            'border': 'none',
            'borderRadius': '5px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
            'cursor': 'pointer'
        }
        btn_disabled = False

    freeze = bool(n_clicks and n_clicks > 0)
    freeze_outputs = [freeze] * 25
    formatting_style = {'pointerEvents': 'none', 'opacity': 0.5} if freeze else {}

    return [btn_text, btn_style, btn_disabled] + freeze_outputs + [formatting_style]

@app.callback(
    [
        Output('replicate-analysis-button', 'children'),
        Output('replicate-analysis-button', 'style'),
        Output('replicate-analysis-button', 'disabled'),
    ],
    [Input('replicate-analysis-button', 'n_clicks'), Input('progress-interval', 'n_intervals')],
    [State('save_file', 'value')],
    prevent_initial_call=True
)
def run_replicate_analysis(n_clicks, n_intervals, savefile):
    if n_clicks == 0:
        raise exceptions.PreventUpdate

    with thread_lock:
        if "Replicate analysis completed successfully!" in messages:
            btn_style = {
                'fontSize': '2rem',
                'padding': '20px 40px',
                'width': '40%',
                'alignSelf': 'center',
                'marginTop': '20px',
                'backgroundColor': '#FFD700',
                'color': 'black',
                'border': 'none',
                'borderRadius': '5px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
                'cursor': 'pointer'
            }
            return "Replicate analysis done!", btn_style, True

    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'replicate-analysis-button.n_clicks':
        def run_superplots_thread():
            try:
                from superplots import superplots
                import re
                base_name = re.sub(r'-r\d+$', '', savefile)
                input_pattern = f"{base_name}-r*_Results.xlsx"
                superplots(input_pattern)
                with thread_lock:
                    messages.append("")
                    messages.append("Replicate analysis completed successfully!")
            except Exception as e:
                with thread_lock:
                    messages.append("")
                    messages.append(f"Error in replicate analysis: {str(e)}")

        thread = threading.Thread(target=run_superplots_thread)
        thread.start()

    btn_style = {
        'fontSize': '2rem',
        'padding': '20px 40px',
        'width': '40%',
        'alignSelf': 'center',
        'marginTop': '20px',
        'backgroundColor': '#7bb77b',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)',
        'cursor': 'pointer'
    }
    return "Running replicate analysis...", btn_style, True

app.clientside_callback(
    """
    function(isCompleted, fileNames) {
        if (isCompleted && fileNames && fileNames.length > 0) {
            return "Please wait, processing file to autodetect column names and values...";
        }
        return "";
    }
    """,
    Output('segments_status', 'children'),
    Input('segments_upload', 'isCompleted'),
    State('segments_upload', 'fileNames'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(isCompleted, fileNames) {
        if (isCompleted && fileNames && fileNames.length > 0) {
            return "Please wait, processing file to autodetect column names...";
        }
        return "";
    }
    """,
    Output('categories_status', 'children'),
    Input('category_upload', 'isCompleted'),
    State('category_upload', 'fileNames'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_intervals) {
        try {
            // Check segments dropdown
            var segmentsParentDropdown = document.getElementById('parent_id');
            var segmentsStatusDiv = document.getElementById('segments_status');
            
            if (segmentsParentDropdown && segmentsStatusDiv) {
                var dropdownDiv = segmentsParentDropdown.querySelector('.Select-value-label') || 
                                 segmentsParentDropdown.querySelector('.Select-single-value') ||
                                 segmentsParentDropdown.querySelector('[class*="singleValue"]');
                
                var currentText = segmentsStatusDiv.textContent;
                
                if (dropdownDiv && currentText.indexOf('Please wait') !== -1) {
                    var dropdownText = dropdownDiv.textContent || dropdownDiv.innerText || '';
                    if (dropdownText && dropdownText.indexOf('Select') === -1) {
                        segmentsStatusDiv.textContent = "";
                    }
                }
            }
            
            // Check categories dropdown
            var categoriesParentDropdown = document.getElementById('parent_id2');
            var categoriesStatusDiv = document.getElementById('categories_status');
            
            if (categoriesParentDropdown && categoriesStatusDiv) {
                var dropdownDiv = categoriesParentDropdown.querySelector('.Select-value-label') || 
                                 categoriesParentDropdown.querySelector('.Select-single-value') ||
                                 categoriesParentDropdown.querySelector('[class*="singleValue"]');
                
                var currentText = categoriesStatusDiv.textContent;
                
                if (dropdownDiv && currentText.indexOf('Please wait') !== -1) {
                    var dropdownText = dropdownDiv.textContent || dropdownDiv.innerText || '';
                    if (dropdownText && dropdownText.indexOf('Select') === -1) {
                        categoriesStatusDiv.textContent = "";
                    }
                }
            }
        } catch (e) {
            // Silently handle any errors to prevent callback failures
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy', 'style'),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)

if __name__ == '__main__':
    app.run(port=5555, debug=True)
