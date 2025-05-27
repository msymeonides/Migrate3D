from dash import Dash
from dash import dcc, html, Input, Output, State, exceptions
import pandas as pd
import base64
import io
import os
import threading
from datetime import date
from run_migrate import migrate3D
from graph_all_segments import graph_sorted_segments
from generate_PCA import generate_PCA
from summary_statistics_figures import generate_figures
import dash_bootstrap_components as dbc

parameters = {'timelapse': 4, 'arrest_limit': 3.0, 'moving': 4, 'contact_length': 12, 'arrested': 0.95, 'tau_msd': 50,
              'tau_euclid': 25, 'savefile': '{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results', 'verbose': False,
              'object_id_col_name': 'Parent ID', 'time_col_name': "Time", 'x_col_name': 'X Coordinate',
              'y_col_name': 'Y Coordinate', 'z_col_name': 'Z Coordinate', 'object_id_2_col': 'ID',
              'category_col': 'Category', 'interpolate': False, 'multi_track': True, 'contact': False,
              'attractors': False, 'pca_filter': None, 'infile_tracks': False}

# initialize the app
file_storing = {}
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/', external_stylesheets=[dbc.themes.BOOTSTRAP])

increase = 0

app.layout = dbc.Container(children=[dbc.Row([
    dbc.Col(html.H1(children='Migrate3D'), width=10, className="d-flex align-items-end"),
    dbc.Col(html.Img(src="assets/uvm_asset.jpeg",
                     style={"position": "absolute",
                            "top": "10px",
                            "right": "10px",
                            "width": "150px",
                            "height": "auto",
                            "zIndex": 1000,
                            })),
], style={"height": "100px"}),
    html.Div(id='master_div',
             children=[html.Hr(),
                       # start of file inputs here
                       html.Div(id='inputs',
                                children=[
                                    html.Div(className='segment_div',
                                             id='segment_div',
                                             children=[
                                                 "Segments input files should be a .csv with cell ID, time, X, Y, and Z coordinates. "
                                                 "Please ensure that column headers are in the first row of the .csv file input.",
                                                 dcc.Upload(id='segments_upload',
                                                            children='Enter your segments .csv file here by clicking or '
                                                                     'dropping: ',
                                                            style={'width': '100%',
                                                                   'height': '60px',
                                                                   'lineHeight': '60px',
                                                                   'borderWidth': '1px',
                                                                   'borderStyle': 'dashed',
                                                                   'borderRadius': '5px',
                                                                   'textAlign': 'center',
                                                                   'margin': '10px'})],
                                             style={'width': '45%', 'display': 'inline-block'}),
                                    # end segment upload div
                                    html.Div(className='categories_div', id='categories_div',
                                             children=[
                                                 'Categories input files should be a .csv with cell ID and cell category. Please ensure that column headers are in the '
                                                 'first row of the .csv file input.',
                                                 dcc.Upload(id='category_upload',
                                                            children='Enter your category .csv file here by clicking or '
                                                                     'dropping (optional):',
                                                            style={'width': '100%',
                                                                   'height': '60px',
                                                                   'lineHeight': '60px',
                                                                   'borderWidth': '1px',
                                                                   'borderStyle': 'dashed',
                                                                   'borderRadius': '5px',
                                                                   'textAlign': 'center',
                                                                   'margin': '10px'})],
                                             style={'width': '45%', 'display': 'inline-block'})],
                                style={'display': 'flex', 'justify-content': 'space-between'}),  # End Categories Div
                       html.Hr(),

                       html.Div(id='identifier_divs',
                                children=[
                                    html.Div(id='column_populate_segments',
                                             children=[html.H4('Select Column Identifiers for segments file'),
                                                       dcc.Dropdown(id='parent_id', placeholder='Select ID column'),
                                                       dcc.Dropdown(id='time_formatting',
                                                                    placeholder='Select time column'),
                                                       dcc.Dropdown(id='x_axis',
                                                                    placeholder='Select x-coordinate column'),
                                                       dcc.Dropdown(id='y_axis',
                                                                    placeholder='Select y-coordinate column'),
                                                       dcc.Dropdown(id='z_axis',
                                                                    placeholder='Select z-coordinate column (leave blank for 2D data)')],
                                             style={'width':'45%', 'display': 'inline-block'}),

                                    html.Div(id='Categories_dropdown',
                                             children=[html.H4('Enter Column Identifiers for tracks (optional)'),
                                                       dcc.Dropdown(id='parent_id2', placeholder='Select ID column'),
                                                       dcc.Dropdown(id='category_col',
                                                                    placeholder='Column header name in input Categories file'
                                                                                ' for object category'),
                                                       ], style={'width': '45%', 'display': 'inline-block'})],
                                                    style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Hr(),
                       html.Div(id='Parameters',
                                children=[
                                    html.H6(children=['Enter timelapse interval']),
                                    dcc.Input(id='Timelapse', value=4),
                                    html.Hr(),
                                    html.H6(children=['Enter maximum displacement to consider an object arrested']),
                                    dcc.Input(id='arrest_limit', value=3.0),
                                    html.Hr(),
                                    html.H6(children=[
                                        'Enter minimum timepoints an object has to be moving for to be considered moving']),
                                    dcc.Input(id='moving', value=4),
                                    html.Hr(),
                                    html.H6(children=['Enter minimum distance between objects to consider a contact']),
                                    dcc.Input(id='contact_length', value=12),
                                    html.Hr(),
                                    html.H6(
                                        children=['Enter minimum arrest coefficient to consider an object arrested']),
                                    dcc.Input(id='arrested', value=0.95),
                                    html.Hr(),
                                    html.H6(children=['Enter tau value for MSD calculations']),
                                    dcc.Input(id='tau_msd', value=50),
                                    html.Hr(),
                                    html.H6(children=['Enter tau value for Euclidean distance calculations']),
                                    dcc.Input(id='tau_euclid', value=25),
                                    html.Hr(),
                                    html.H6(children=['Select formatting options if needed']),
                                    dcc.Checklist(id='formatting_options',
                                                  options=['Multitrack', 'Interpolate', 'Verbose', 'Contacts',
                                                           'Attractors',
                                                           'Generate Figures']),
                                    html.Hr(),
                                    html.H6(
                                        children='Enter subset of categories for PCA and xgboost analysis (separated by space)'),
                                    dcc.Input(id='PCA_filter', placeholder='e.g. 4 5 6', ),
                                    html.Hr(),
                                    html.H6(children=['Save results as:']),
                                    dcc.Input(id='save_file',
                                              placeholder='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
                                              value='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results'),
                                    html.Hr(),
                                    html.Button('Run Migrate3D', id='Run_migrate', n_clicks=0)
                                ]),
                       html.Hr(),
                       dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3",
                                    color='success'),
                       dcc.Interval(id='progress-interval', interval=1000, n_intervals=0),
                       # interval = 1000 ms = 1 second
                       ])], className="body", fluid=True)


@app.callback(
    Input(component_id='parent_id', component_property='value'),
    Input(component_id='time_formatting', component_property='value'),
    Input(component_id='x_axis', component_property='value'),
    Input(component_id='y_axis', component_property='value'),
    Input(component_id='z_axis', component_property='value'),
    Input(component_id='Timelapse', component_property='value'),
    Input(component_id='arrest_limit', component_property='value'),
    Input(component_id='moving', component_property='value'),
    Input(component_id='contact_length', component_property='value'),
    Input(component_id='arrested', component_property='value'),
    Input(component_id='tau_msd', component_property='value'),
    Input(component_id='tau_euclid', component_property='value'),
    Input(component_id='formatting_options', component_property='value'),
    Input(component_id='save_file', component_property='value'),
    Input(component_id='segments_upload', component_property='children'),
    Input(component_id='category_upload', component_property='children'),
    Input(component_id='parent_id2', component_property='value'),
    Input(component_id='category_col', component_property='value'),
    Input(component_id='PCA_filter', component_property='value'),
    Input(component_id='Run_migrate', component_property='n_clicks'),
    prevent_initial_call=True)
def run_migrate(*vals):
    (parent_id, time_for, x_for, y_for, z_for, timelapse, arrest_limit, moving, contact_length, arrested, tau_msd,
     tau_euclid, formatting_options, savefile, segments_file_name, tracks_file, parent_id2, category_col_name,
     pca_filter,
     run_button) = vals
    if run_button == 0:
        raise exceptions.PreventUpdate
    else:
        if pca_filter is None:
            pass
        else:
            pca_filter = pca_filter.split(sep=' ')

        df_segments, df_sum, df_pca = migrate3D(parent_id, time_for, x_for, y_for, z_for, int(timelapse),
                                                float(arrest_limit),
                                                int(moving),
                                                int(contact_length), float(arrested), int(tau_msd), int(tau_euclid),
                                                formatting_options,
                                                savefile, segments_file_name, tracks_file,
                                                parent_id2, category_col_name, parameters, pca_filter,
                                                progress_callback=update_progress)
        if formatting_options is None:
            pass
        else:
            if 'Generate Figures' in formatting_options:
                fig_segments = graph_sorted_segments(df_segments, df_sum, parameters['infile_tracks'], savefile)
                sum_fig = generate_figures(df_sum)
                sum_fig.append(fig_segments)

                if df_pca is None:
                    fig_pca = None
                    with open(f'{savefile}_figures.html', 'a') as f:
                        for i in sum_fig:
                            f.write(i.to_html(full_html=False, include_plotlyjs='cdn'))

                else:
                    fig_pca = generate_PCA(df_pca)
                    sum_fig.append(fig_pca)
                    with open(f'{savefile}_figures.html', 'a') as f:
                        for i in sum_fig:
                            f.write(i.to_html(full_html=False, include_plotlyjs='cdn'))

    print("Migrate3D run completed! You may terminate the Python process.")
    update_progress(100)
    alert = dbc.Alert([html.H4("Migrate3D run completed! You may terminate the Python process",
                               className="alert-heading"),

                       ])
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
    except Exception as e:
        return html.Div(['There was an error processing this file.']), [], None, [], None, [], None, [], None, [], None

    file_storing['Segments'] = df

    # Auto-detection logic
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
    except Exception as e:
        return html.Div(['There was an error processing this file.']), [], None, [], None

    file_storing['Categories'] = df

    # Auto-detection logic
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
    return increase


def update_progress(state_):
    global increase
    increase += state_
    if increase > 100:
        increase = 100
    print(increase)


if __name__ == '__main__':
    app.run(port=5555, debug=True)
