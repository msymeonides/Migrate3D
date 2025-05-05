from dash import Dash
from dash import dcc, html, Input, Output, State, exceptions
import pandas as pd
import base64
import io
import os
import numpy as np
from warnings import simplefilter
from datetime import date
from formatting import multi_tracking, adjust_2D, interpolate_lazy
from run_migrate import migrate3D
from openpyxl import load_workbook
from graph_all_segments import graph_sorted_segments
from generate_PCA import generate_PCA
from summary_statistics_figures import generate_figures

parameters = {'timelapse': 4, 'arrest_limit': 3.0, 'moving': 4, 'contact_length': 12, 'arrested': 0.95, 'tau_msd': 50,
              'tau_euclid': 25, 'savefile': '{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results', 'verbose': False,
              'object_id_col_name': 'Parent ID', 'time_col_name': "Time", 'x_col_name': 'X Coordinate',
              'y_col_name': 'Y Coordinate', 'z_col_name': 'Z Coordinate', 'object_id_2_col': 'ID',
              'category_col': 'Category', 'interpolate': False, 'multi_track': True, 'two_dim': False,
              'contact': False, 'attract': False, 'pca_filter': None, 'infile_tracks': False}

# initialize the app
file_storing = {}
app = Dash(__name__)

app.layout = (
    html.Div(
        children=[
            html.H1(children='Migrate3D'),  # title
            html.Hr(),
            # start of file inputs here
            html.Div(id='Segments',
                     children=["Segments input files should be a .csv with cell ID, time, X, Y, and Z coordinates. "
                               "Please ensure that column headers are in the first row of the .csv file input.",
                               html.Div(className='segment_div',
                                        id='segment_div',
                                        children=[
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
                                                              'margin': '10px'})])]),  # end segment upload div
            html.Hr(),
            html.Div(id="Categories",
                     children=['Categories input files should be a .csv with cell ID and cell category (No categories '
                               'file is necessary to run the program). Please ensure that column headers are in the '
                               'first row of the .csv file input.',
                               html.Div(className='categories_div', id='categories_div',
                                        children=[
                                            dcc.Upload(id='category_upload',
                                                       children='Enter your category .csv file here by clicking or '
                                                                'dropping:',
                                                       style={'width': '100%',
                                                              'height': '60px',
                                                              'lineHeight': '60px',
                                                              'borderWidth': '1px',
                                                              'borderStyle': 'dashed',
                                                              'borderRadius': '5px',
                                                              'textAlign': 'center',
                                                              'margin': '10px'})])]),  # End Categories Div

            html.Div(id='column_populate_segments',
                     children=[html.H4('Select Column Identifiers for segments file'),
                               dcc.Dropdown(id='parent_id', placeholder='Select ID column'),
                               dcc.Dropdown(id='time_formatting', placeholder='Select time column'),
                               dcc.Dropdown(id='x_axis', placeholder='Select x-coordinate column'),
                               dcc.Dropdown(id='y_axis', placeholder='Select y-coordinate column'),
                               dcc.Dropdown(id='z_axis', placeholder='Select z-coordinate column (leave empty for 2D data)')]),

            html.Div(id='Categories_dropdown',
                     children=[html.H4('Enter Column Identifiers for tracks (optional)'),
                               dcc.Dropdown(id='parent_id2', placeholder='Select ID column'),
                               dcc.Dropdown(id='category_col', placeholder='Column header name in input Categories file'
                                                                           'for object category'),
                               ]),
            html.Hr(),
            html.Div(id='Parmeters',
                     children=[
                         html.H4(children=['Enter Timelapse interval']),
                         dcc.Input(id='Timelapse', value=4),
                         html.Hr(),
                         html.H4(children=['Enter arrest limit']),
                         dcc.Input(id='arrest_limit', value=3.0),
                         html.Hr(),
                         html.H4(children=['Enter moving']),
                         dcc.Input(id='moving', value=4),
                         html.Hr(),
                         html.H4(children=['Enter Contact length']),
                         dcc.Input(id='contact_length', value=12),
                         html.Hr(),
                         html.H4(children=['Enter arrested']),
                         dcc.Input(id='arrested', value=0.95),
                         html.Hr(),
                         html.H4(children=['Enter tau MSD']),
                         dcc.Input(id='tau_msd', value=50),
                         html.Hr(),
                         html.H4(children=['Enter tau euclid']),
                         dcc.Input(id='tau_euclid', value=25),
                         html.Hr(),
                         html.H4(children=['Select formatting options if needed']),
                         dcc.Checklist(id='formatting_options',
                                       options=['Multitrack', 'Two-dimensional', 'Interpolate', 'Verbose', 'Contacts', 'Attractors', 'Generate Figures']),
                         html.Hr(),
                         html.H4(children='Enter subset of categories for PCA and xgboost analysis (separated by space)'),
                         dcc.Input(id='PCA_filter', placeholder='ex: 4 5 6',),
                         html.H4(children=['Save results as:']),
                         dcc.Input(id='save_file',
                                   placeholder='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results',
                                   value='{:%Y_%m_%d}'.format(date.today()) + '_Migrate3D_Results'),
                         html.Hr(),
                         html.Button('Run Migrate3D', id='Run_migrate', n_clicks=0)
                     ]),

        html.Div(id='dummy', style={'display': 'none'})
        ]))


@app.callback(
    Output(component_id='dummy', component_property='children'),
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
     tau_euclid, formatting_options, savefile, segments_file_name, tracks_file, parent_id2, category_col_name, pca_filter,
     run_button) = vals
    if run_button == 0:
        raise exceptions.PreventUpdate
    else:
        if pca_filter is None:
            pass
        else:
            pca_filter = pca_filter.split(sep=' ')
            print(type(pca_filter))

        df_segments, df_sum, df_pca = migrate3D(parent_id, time_for, x_for, y_for, z_for, int(timelapse), float(arrest_limit),
                                        int(moving),
                                        int(contact_length), float(arrested), int(tau_msd), int(tau_euclid),
                                        formatting_options,
                                        savefile, segments_file_name, tracks_file,
                                        parent_id2, category_col_name, parameters, pca_filter)
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

    # Dummy return to satisfy the output
    return "Migrate3D run completed"


# callback for segments file
@app.callback(
    Output(component_id='segments_upload', component_property='children'),
    Output(component_id='parent_id', component_property='options'),
    Output(component_id='time_formatting', component_property='options'),
    Output(component_id='x_axis', component_property='options'),
    Output(component_id='y_axis', component_property='options'),
    Output(component_id='z_axis', component_property='options'),
    Input(component_id='segments_upload', component_property='contents'),
    State(component_id='segments_upload', component_property='filename'),
    prevent_initial_call=True)
def get_file(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate

    elif contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        except Exception as e:
            return html.Div([
                'There was an error processing this file.'
            ])

        file_storing['Segments'] = df
        return filename, list(df.columns), list(df.columns), list(df.columns), list(df.columns), list(df.columns)

    else:
        pass


@app.callback(
    Output(component_id='category_upload', component_property='children'),
    Output(component_id='parent_id2', component_property='options'),
    Output(component_id='category_col', component_property='options'),
    Input(component_id='category_upload', component_property='contents'),
    State(component_id='category_upload', component_property='filename'),
    prevent_initial_call=True)
def get_file(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate

    elif contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        except Exception as e:
            return html.Div([
                'There was an error processing this file.'
            ])

        file_storing['Categories'] = df
        return filename, list(df.columns), list(df.columns)

    else:
        pass


if __name__ == '__main__':
    app.run(port=5555, debug=True)
