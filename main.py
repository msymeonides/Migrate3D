from dash import Dash
from dash import dcc, html, Input, Output, State, exceptions
import pandas as pd
import base64
import io
import os
import threading
import dash_bootstrap_components as dbc
from datetime import date

from run_migrate import migrate3D
from graph_all_segments import graph_sorted_segments
from generate_PCA import generate_PCA
from summary_statistics_figures import generate_figures
from shared_state import messages, thread_lock, get_progress, set_progress


# Defaults for tunable parameters can be set here
parameters = {'timelapse': 4,       # Timelapse interval
              'arrest_limit': 3.0,  # Arrest limit
              'moving': 4,          # Minimum timepoints
              'contact_length': 12, # Contact length
              'arrested': 0.95,     # Maximum arrest coefficient
              'tau_msd': 50,        # Maximum MSD Tau value
              'tau_euclid': 25,     # Maximum Euclidean distance Tau value
              'savefile': '{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results', 'verbose': False,
              'object_id_col_name': 'Parent ID', 'time_col_name': "Time", 'x_col_name': 'X Coordinate',
              'y_col_name': 'Y Coordinate', 'z_col_name': 'Z Coordinate', 'object_id_2_col': 'ID',
              'category_col': 'Category', 'interpolate': False, 'multi_track': True, 'contact': False,
              'attractors': False, 'pca_filter': None, 'infile_tracks': False}

file_storing = {}
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/', external_stylesheets=[dbc.themes.BOOTSTRAP])
with thread_lock:
    messages.append('Waiting for user input. Load data, adjust parameters and options, and click "Run Migrate3D" to start the analysis.')

app.layout = dbc.Container(
    children=[
        dbc.Row([
            dbc.Col(
                html.H1(children='Migrate3D'),
                width="auto",
                className="d-flex align-items-center"
            ),
            dbc.Col(
                html.Img(
                    src="assets/uvm_asset.jpeg",
                    style={
                        "position": "absolute",
                        "top": "10px",
                        "right": "50px",
                        "width": "200px",
                        "height": "auto",
                        "zIndex": 1000,
                    }
                )
            ),
        ], style={"height": "150px"}),
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
                            'margin': '10px auto'
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
                            'margin': '10px auto'
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
                        html.H4('Select Column Identifiers for Segments file:'),
                        dcc.Dropdown(id='parent_id', placeholder='Select object ID column'),
                        dcc.Dropdown(id='time_formatting', placeholder='Select time column'),
                        dcc.Dropdown(id='x_axis', placeholder='Select X coordinate column'),
                        dcc.Dropdown(id='y_axis', placeholder='Select Y coordinate column'),
                        dcc.Dropdown(id='z_axis', placeholder='Select Z coordinate column (leave blank for 2D data)'),
                    ],
                    style={'width': '90%', 'display': 'inline-block'}
                ),
                html.Div(
                    id='Categories_dropdown',
                    children=[
                        html.H4('Select Column Identifiers for Categories file (if used):'),
                        dcc.Dropdown(id='parent_id2', placeholder='Select object ID column'),
                        dcc.Dropdown(id='category_col', placeholder='Select Category column'),
                    ],
                    style={'width': '90%', 'display': 'inline-block'}
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
                                html.H6(children=['Timelapse interval']),
                                dcc.Input(id='Timelapse', value=4),
                                html.Hr(),
                                html.H6(children=['Arrest limit (displacements below this value will not count as movement)']),
                                dcc.Input(id='arrest_limit', value=3.0),
                                html.Hr(),
                                html.H6(children=['Minimum timepoints (objects must be moving for at least this many timepoints to be fully analyzed)']),
                                dcc.Input(id='moving', value=4),
                                html.Hr(),
                                html.H6(children=['Contact length (if the distance between two objects is less than this, they will be considered to be in contact)']),
                                dcc.Input(id='contact_length', value=12),
                                html.Hr(),
                                html.H6(children=['Maximum arrest coefficient (objects with arrest coefficient above this value will be considered arrested)']),
                                dcc.Input(id='arrested', value=0.95),
                                html.Hr(),
                                html.H6(children=['Maximum MSD Tau value (should be equal to the median number of timepoints in the dataset)']),
                                dcc.Input(id='tau_msd', value=50),
                                html.Hr(),
                                html.H6(children=['Maximum Euclidean distance Tau value (should be half of the Maximum MSD Tau Value)']),
                                dcc.Input(id='tau_euclid', value=25),
                                html.Hr(),
                                html.H4(children=['Formatting options']),
                                dcc.Checklist(
                                    id='formatting_options',
                                    options=[
                                        {'label': ' Multitracking (if an object ID is represented by multiple segments at a given timepoint, they will be spatially averaged into one segment)', 'value': 'Multitrack'},
                                        {'label': ' Interpolation (if an object ID is missing a timepoint, that timepoint will be inferred by simple linear interpolation and inserted)', 'value': 'Interpolate'},
                                        {'label': ' Verbose (includes the results of all calculations in the output file)', 'value': 'Verbose'},
                                        {'label': ' Contacts (identifies contacts between objects)', 'value': 'Contacts'},
                                        {'label': ' Attractors (identifies instances where an object is attracting other objects towards it)', 'value': 'Attractors'},
                                        {'label': ' Generate Figures (creates figures for summary statistics and PCA)', 'value': 'Generate Figures'}
                                    ],
                                    inputStyle={'width': '30px', 'height': '30px', 'marginRight': '5px', 'marginBottom': '5px', 'marginTop': '5px'},
                                    labelStyle={'display': 'flex', 'alignItems': 'center'}
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
                                    placeholder='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
                                    value='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
                                    style={'width': '250px'}
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'justifyContent': 'space-between',
                                'height': '100%'
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
                                    style={
                                        'fontSize': '2rem',
                                        'padding': '20px 40px',
                                        'width': '40%',
                                        'alignSelf': 'center',
                                        'marginBottom': '70px'
                                    }
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
    (parent_id, time_for, x_for, y_for, z_for, timelapse, arrest_limit, moving, contact_length, arrested, tau_msd,
     tau_euclid, formatting_options, savefile, segments_file_name, tracks_file, parent_id2, category_col_name,
     pca_filter) = args

    try:
        if pca_filter is not None and pca_filter.strip() != '':
            pca_filter = pca_filter.split(sep=' ')
        else:
            pca_filter = None
        set_progress(5)
        df_segments, df_sum, df_pca = migrate3D(
            parent_id, time_for, x_for, y_for, z_for, int(timelapse),
            float(arrest_limit), int(moving), int(contact_length), float(arrested),
            int(tau_msd), int(tau_euclid), formatting_options, savefile,
            segments_file_name, tracks_file, parent_id2, category_col_name,
            parameters, pca_filter)

        if formatting_options and 'Generate Figures' in formatting_options:
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
        with thread_lock:
            messages.append("You may close the Anaconda prompt and the GUI browser tab, or just terminate the Python process.")
        set_progress(100)
    except Exception as e:
        with thread_lock:
            messages.append(f"Error: {str(e)}")
        set_progress(100)

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
    Input('segments_upload', 'children'),
    Input('category_upload', 'children'),
    Input('parent_id2', 'value'),
    Input('category_col', 'value'),
    Input('PCA_filter', 'value'),
    Input('Run_migrate', 'n_clicks'),
    prevent_initial_call=True
)
def run_migrate(*vals):
    run_button = vals[-1]
    if run_button == 0:
        raise exceptions.PreventUpdate
    with thread_lock:
        messages.clear()
    set_progress(0)
    args = vals[:-1]
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
    Input("progress-interval", 'n_intervals'),
)
def update_pbar(n):
    return get_progress()

@app.callback(
    Output('alert_box', 'children'),
    Input('progress-interval', 'n_intervals')
)
def update_alert_box(n):
    with thread_lock:
        return html.Div([
            html.Pre('\n'.join(messages), style={'whiteSpace': 'pre-wrap', 'margin': 0})
        ])

if __name__ == '__main__':
    app.run(port=5555, debug=True)
