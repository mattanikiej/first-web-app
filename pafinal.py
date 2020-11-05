from plasma_properties import transport
from plasma_properties import zbar
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import io
import flask
import csv
from random import randint

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#colors of the app
colors = {
    'background': '#33363E', #dark blue
    'text': '#7FDBFF', #blue
    'graph': '#FFFE00', #black
    'plot': '#26282C' #yellow
        }

#output of the about tab reads from a .txt file
def about_tab_writing():
    with open("assets/about_tab.txt") as file:
        data = file.read()
    return data

#numpy arrays of current values on the graphs
current_x_values = np.array([])
current_mi_values = np.array([])
current_sd_values = np.array([])
current_vc_values = np.array([])
current_tc_values = np.array([])
x_axis_unit = ''

#layout of the app
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        
    #Title
    html.H1(children='Stanton-Murillo Ionic Transport Calculator',
            style = {'color': colors['text'], 'padding': '10px 15px', 'font-family': 'serif'}),
    #columns for each part
    html.Div(style={'backgroundColor': colors['background']}, className = "row", children=[
        #first column for tabs and sliders
        
        html.Div(style={'backgroundColor': colors['background'], 'padding': '10px 15px'}, className = "three columns", children=[
        #creates multiple tabs for app
            dcc.Tabs(style={'padding': '10px 0px'}, children=[
                #main tab contains graphs and sliders
                dcc.Tab(label = 'Main', children=[
                    
                    #Allows user to download the values of the graphs into a csv:
                    html.A('Click To Download Points', 
                                id='graph-points', 
                                style = {'color': colors['graph'], 'font-family': 'serif', 'font-size': '20px'},
                                ),
                    
                    #Shows what value is being used in the graphs:
                    #Z value used
                    html.H6(
                        id='z-output',
                        style = {'color': colors['text'], 'font-family': 'serif'}
                        ),
                    #t value used
                    html.H6(
                        id='t-output',
                        style = {'color': colors['text'], 'font-family': 'serif'}
                        ),
                    #density value used
                    html.H6(
                        id='rho-output',
                        style = {'color': colors['text'], 'font-family': 'serif'}
                        ),
                    
                    #x-axis options dropdown
                    html.H6('x-axis:',
                            style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[
                            {'label': 'Species', 'value': 'Z'},
                            {'label': 'Temperature [eV]', 'value': 'temp'},
                            {'label': 'Density [g/cc]', 'value': 'density'}
                            ],
                        value='temp'
                        ),
                    #Atomic Mass input
                    html.H6('Atomic Mass [g]:',
                            style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='atomic-mass',
                        placeholder='Atomic Mass',
                        type='number',
                        value=1.9944235e-23,
                        ),
                    #z slider
                    html.H6('Species:',
                            style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.RangeSlider(
                        id='species-slider',
                        min=1,
                        max=118,
                        step=1,
                        value=[1, 6],
                        updatemode='drag'
                        ),
                    #left input of slider for species
                    html.Label('Min:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='species-min',
                        type='number',
                        value=1
                        ),
                    #right input of slider for species
                    html.Label('Max:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='species-max',
                        type='number',
                        value=6
                        ),
                    #temperature slider
                    html.H6('Temperature [eV]:',
                            style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.RangeSlider(
                        id='temp-slider',
                        min=0,
                        max=1000,
                        step=0.1,
                        value = [0.2, 200],
                        updatemode='drag'
                        ),
                    #left input of slider for temp
                    html.Label('Min:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='temp-min',
                        type='number',
                        value=0.2
                        ),
                    #right input of slider for temp
                    html.Label('Max:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='temp-max',
                        type='number',
                        value=200
                        ),
                    #density slider
                    html.H6('Density [g/cc]:',
                            style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.RangeSlider(
                        id='density-slider',
                        min=1,
                        max=10,
                        step=0.1,
                        value=[1, 3],
                        updatemode='drag'
                        ),
                    #left input of slider for density
                    html.Label('Min:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='density-min',
                        type='number',
                        value=1
                        ),
                    #right input of slider for density
                    html.Label('Max:',
                               style = {'color': colors['text'], 'font-family': 'serif'}),
                    dcc.Input(
                        id='density-max',
                        type='number',
                        value=3
                        ),
                    ]),
            #about tab for documentation
            dcc.Tab(label = 'About', children=[
                #reads from a .txt that explains how to use the app
                html.H6(id='about-tab', 
                        style = {'color': colors['text'], 'font-family': 'serif'},
                        children=[
                            about_tab_writing()
                            ]),
                html.A( 'Click here for a pdf with more information.',
                    id='pdf-download', 
                    href='/assets/SM_Ionic_Transport_Calculator.pdf', 
                    style = {'color': colors['graph'], 'font-family': 'serif', 'font-size': '20px'},
                    download='SM_Ionic_Transport_Calculator.pdf',
                    )
                ]),
            ]),
        ]),
    
    
        #creates second column for mean ionization and graphs
        html.Div(style={'backgroundColor': colors['background'], 'padding': '10px 0px'}, className="four columns", children=[
            #Mean Ionization Graph
            dcc.Graph(
                id='mean-ionization-graph',
                config={
                    'showSendToCloud': True,
                    'plotlyServerURL': 'https://plot.ly'
                    },
                style={'padding': '10px 0px'}
                ),
            
            #Viscosity Graph
            dcc.Graph(
                id='viscosity-graph',
                config={
                    'showSendToCloud': True,
                    'plotlyServerURL': 'https://plot.ly'
                    },
                style={'padding': '10px 0px'}
                )
            
            ]),
        
            #creates third column for self diffusion and graphs
            html.Div(style={'backgroundColor': colors['background'], 'padding': '10px 0px'}, className="four columns", children=[
                #Self Diffusion Graph
                dcc.Graph(
                    id='self-diffusion-graph',
                    config={
                        'showSendToCloud': True,
                        'plotlyServerURL': 'https://plot.ly'
                        },
                    style={'padding': '10px 0px'}
                    ),
                
                #Thermal Conductivity Graph
                dcc.Graph(
                    id='thermal-conductivity-graph',
                    config={
                        'showSendToCloud': True,
                        'plotlyServerURL': 'https://plot.ly'
                        },
                    style={'padding': '10px 0px'}
                    ),
                
                ]),
    ]),
])  

#Callbacks for dcc objects

    
#Callback to show values of slider in the input
#Inputs for species
@app.callback(
    [Output('species-min', 'value'),
    Output('species-max', 'value')],
    [Input('species-slider', 'value')])
def update_species_min_max(species_slider):
    return species_slider[0], species_slider[1]

#inputs for temperature
@app.callback(
    [Output('temp-min', 'value'),
    Output('temp-max', 'value')],
    [Input('temp-slider', 'value')])
def update_temp_min_max(temp_slider):
    return temp_slider[0], temp_slider[1]

#inputs for density
@app.callback(
    [Output('density-min', 'value'),
    Output('density-max', 'value')],
    [Input('density-slider', 'value')])
def update_density_min_max(density_slider):
    return density_slider[0], density_slider[1]

#Callback that tells user what values are being used in the graphs
@app.callback(
    [Output('z-output', 'children'),
     Output('t-output', 'children'),
     Output('rho-output', 'children')],
    [Input('species-max', 'value'),
     Input('temp-max', 'value'),
     Input('density-max', 'value'),
     Input('x-axis-dropdown', 'value')]
)
def update_output_div(species, temp, density, dropdown):
    if dropdown == 'Z':
        return '', 'Temperature [eV]: {}'.format(temp), 'Density [g/cc]: {}'.format(density)
    elif dropdown == 'temp':
        return 'Z: {}'.format(species), '', 'Density [g/cc]: {}'.format(density)
    elif dropdown == 'density':
        return 'Z: {}'.format(species), 'Temperature [eV]: {}'.format(temp), ''
    return 'Z: {}'.format(species), 'Temperature [eV]: {}'.format(temp), 'Density [g/cc]: {}'.format(density)

#Callbacks that provide points and layouts for the graphs
#Callback for mean ionization graph
@app.callback(
    Output('mean-ionization-graph', 'figure'), 
    [Input('x-axis-dropdown', 'value'),
     Input('atomic-mass','value'),
     Input('species-min','value'),
     Input('species-max', 'value'),
     Input('temp-min','value'),
     Input('temp-max', 'value'),
     Input('density-min','value'),
     Input('density-max', 'value')
    ])
def update_mi_graph(dropdown, atomic_mass, species_min, species_max, temp_min, temp_max, density_min, density_max):
    #dictionary containing arrays of values for x axis
    x_values_dict = {'Z': np.geomspace(species_min, species_max),
                     'temp': np.geomspace(temp_min, temp_max),
                     'density': np.geomspace(density_min, density_max)}
    #dictionary containing layput for x axis
    x_layout_dict = {'Z': {'type': 'log', 'title': 'Z', 'color': colors['text']},
                     'temp': {'type': 'log', 'title': 'Temperature [eV]', 'color': colors['text']},
                     'density': {'type': 'log', 'title': 'Density [g/cc]', 'color': colors['text']}}
    #used for csv to download
    global x_axis_unit
    x_axis_unit = dropdown
    return {
        'data': [{
            'type': 'scatter',
            'x': x_values_dict[dropdown],
            'y': mean_ionization_points(atomic_mass, x_values_dict[dropdown], [species_min, species_max], 
                                        [temp_min, temp_max], [density_min, density_max]),
            'line': {'color': colors['graph']}
            }],
        'layout': go.Layout(
            title = 'Mean Ionization',
            xaxis = x_layout_dict[dropdown],
            yaxis = {'type': 'log', 'color': colors['text']},
            plot_bgcolor = colors['plot'],
            paper_bgcolor = colors['plot'],
            font = {'color': colors['text'], 'family': 'serif'}
        )
    }

#Callback for viscosity graph
@app.callback(
    Output('viscosity-graph', 'figure'), 
    [Input('x-axis-dropdown', 'value'),
     Input('atomic-mass','value'),
     Input('species-min','value'),
     Input('species-max', 'value'),
     Input('temp-min','value'),
     Input('temp-max', 'value'),
     Input('density-min','value'),
     Input('density-max', 'value')
    ])
def update_v_graph(dropdown, atomic_mass, species_min, species_max, temp_min, temp_max, density_min, density_max):
    #dictionary containing arrays of values for x axis
    x_values_dict = {'Z': np.geomspace(species_min, species_max),
                     'temp': np.geomspace(temp_min, temp_max),
                     'density': np.geomspace(density_min, density_max)}
    #dictionary containing layput for x axis
    x_layout_dict = {'Z': {'type': 'log', 'title': 'Z', 'color': colors['text']},
                     'temp': {'type': 'log', 'title': 'Temperature [eV]', 'color': colors['text']},
                     'density': {'type': 'log', 'title': 'Density [g/cc]', 'color': colors['text']}}
    #used for csv to download
    global x_axis_unit
    x_axis_unit = dropdown
    return {
        'data': [{
            'type': 'scatter',
            'x': x_values_dict[dropdown],
            'y': viscosity_points(atomic_mass, x_values_dict[dropdown], [species_min, species_max], 
                                  [temp_min, temp_max], [density_min, density_max]),
            'line': {'color': colors['graph']}
            }],
        'layout': go.Layout(
            title = 'Viscosity [g / (cm * s)]',
            xaxis = x_layout_dict[dropdown],
            yaxis = {'type': 'log', 'color': colors['text']},
            plot_bgcolor = colors['plot'],
            paper_bgcolor = colors['plot'],
            font = {'color': colors['text'], 'family': 'serif'}
        )
    }

#Callback for self diffusion graph
@app.callback(
    Output('self-diffusion-graph', 'figure'), 
    [Input('x-axis-dropdown', 'value'),
     Input('atomic-mass','value'),
     Input('species-min','value'),
     Input('species-max', 'value'),
     Input('temp-min','value'),
     Input('temp-max', 'value'),
     Input('density-min','value'),
     Input('density-max', 'value')
    ])
def update_sd_graph(dropdown, atomic_mass, species_min, species_max, temp_min, temp_max, density_min, density_max):
    #dictionary containing arrays of values for x axis
    x_values_dict = {'Z': np.geomspace(species_min, species_max),
                     'temp': np.geomspace(temp_min, temp_max),
                     'density': np.geomspace(density_min, density_max)}
    #dictionary containing layput for x axis
    x_layout_dict = {'Z': {'type': 'log', 'title': 'Z', 'color': colors['text']},
                     'temp': {'type': 'log', 'title': 'Temperature [eV]', 'color': colors['text']},
                     'density': {'type': 'log', 'title': 'Density [g/cc]', 'color': colors['text']}}
    #used for csv to download
    global x_axis_unit
    x_axis_unit = dropdown
    return {
        'data': [{
            'type': 'scatter',
            'x': x_values_dict[dropdown],
            'y': self_diffusion_points(atomic_mass, x_values_dict[dropdown], [species_min, species_max], 
                                       [temp_min, temp_max], [density_min, density_max]),
            'line': {'color': colors['graph']}
            }],
        'layout': go.Layout(
            title = 'Self Diffusion [cm^2 / s]',
            xaxis = x_layout_dict[dropdown],
            yaxis = {'type': 'log', 'color': colors['text']},
            plot_bgcolor = colors['plot'],
            paper_bgcolor = colors['plot'],
            font = {'color': colors['text'], 'family': 'serif'}
        )
    }

#Callback for thermal conductivity graph
@app.callback(
    Output('thermal-conductivity-graph', 'figure'), 
    [Input('x-axis-dropdown', 'value'),
     Input('atomic-mass','value'),
     Input('species-min','value'),
     Input('species-max', 'value'),
     Input('temp-min','value'),
     Input('temp-max', 'value'),
     Input('density-min','value'),
     Input('density-max', 'value')
    ])
def update_tc_graph(dropdown, atomic_mass, species_min, species_max, temp_min, temp_max, density_min, density_max):
    #dictionary containing arrays of values for x axis
    x_values_dict = {'Z': np.geomspace(species_min, species_max),
                     'temp': np.geomspace(temp_min, temp_max),
                     'density': np.geomspace(density_min, density_max)}
    #dictionary containing layput for x axis
    x_layout_dict = {'Z': {'type': 'log', 'title': 'Z', 'color': colors['text']},
                     'temp': {'type': 'log', 'title': 'Temperature [eV]', 'color': colors['text']},
                     'density': {'type': 'log', 'title': 'Density [g/cc]', 'color': colors['text']}}
    #used for csv to download
    global x_axis_unit
    x_axis_unit = dropdown
    return {
        'data': [{
            'type': 'scatter',
            'x': x_values_dict[dropdown],
            'y': thermal_conductivity_points(atomic_mass, x_values_dict[dropdown], [species_min, species_max], 
                                             [temp_min, temp_max], [density_min, density_max]),
            'line': {'color': colors['graph']}
            }],
        'layout': go.Layout(
            title = 'Thermal Conductivity [erg / (cm * s * K)]',
            xaxis = x_layout_dict[dropdown],
            yaxis = {'type': 'log', 'color': colors['text']},
            plot_bgcolor = colors['plot'],
            paper_bgcolor = colors['plot'],
            font = {'color': colors['text'], 'family': 'serif'}
        )
    }

#Callback for button to download values
@app.callback(
    Output('graph-points', 'href'), 
    [Input('x-axis-dropdown', 'value'),
     Input('atomic-mass','value'),
     Input('species-min','value'),
     Input('species-max', 'value'),
     Input('temp-min','value'),
     Input('temp-max', 'value'),
     Input('density-min','value'),
     Input('density-max', 'value')
    ])
def update_link(value, am, z_min, z_max, t_min, t_max, d_min, d_max):
    #random numbers to more or less ensure unique url
    r = randint(1, 10000)
    r2 = randint(1, 10000)
    return '/dash/urlToDownload?value={}{}{}{}{}{}{}{}{}{}.'.format(value, am, z_min, z_max, t_min, t_max, d_min, d_max, r, r2)


@app.server.route('/dash/urlToDownload')
def download_csv():
    global x_axis_unit
    if x_axis_unit == 'temp':
        x_axis_unit = 'Temperature [eV]'
    elif x_axis_unit == 'density':
        x_axis_unit = 'Density [g/cc]'
    elif x_axis_unit == 'Z':
        x_axis_unit = 'Species'
    
    str_io = io.StringIO()
    fieldnames = ['X Axis: {}'.format(x_axis_unit), 'Mean Ionization', 'Viscosity [g / (cm * s)]', 
                  'Self Diffusion [cm^2 / s]', 'Thermal Conductivity [erg / (cm * s * K)]']
    x = current_x_values
    ZBar = current_mi_values
    D = current_sd_values
    eta = current_vc_values
    K = current_tc_values
    writer = csv.DictWriter(str_io, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, 50):
        writer.writerow({'X Axis: {}'.format(x_axis_unit): x[i], 'Mean Ionization': ZBar[i], 'Viscosity [g / (cm * s)]': eta[i], 
                         'Self Diffusion [cm^2 / s]': D[i], 'Thermal Conductivity [erg / (cm * s * K)]': K[i]})
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(mem,
                           mimetype='text/csv',
                           attachment_filename='SMGraphs.csv',
                           as_attachment=True)

#Functions used by callbacks to return numpy arrays that contain the points for each graph
#iterates through temperature values to make points for mean ionization graph
def mean_ionization_points(atomic_mass, x_values, species_list, temp_list, density_list):
    points = np.array([])
    density = density_list[1]
    temp = temp_list[1]
    species = species_list[1]
    #checks to see what variable is being used for x axis
    #checks for species
    if np.array_equal(x_values, np.geomspace(species_list[0], species_list[1])):
        for x in x_values:
            mi = zbar.MeanIonization(atomic_mass, density, temp, x)
            Zbar = mi.tf_zbar()
            points = np.append(points, Zbar)
            
    #checks for temp
    elif np.array_equal(x_values, np.geomspace(temp_list[0], temp_list[1])):
        for x in x_values:
            mi = zbar.MeanIonization(atomic_mass, density, x, species)
            Zbar = mi.tf_zbar()
            points = np.append(points, Zbar)
            
    #checks for density
    elif np.array_equal(x_values, np.geomspace(density_list[0], density_list[1])):
        for x in x_values:
            mi = zbar.MeanIonization(atomic_mass, x, temp, species)
            Zbar = mi.tf_zbar()
            points = np.append(points, Zbar)
            
    global current_mi_values 
    current_mi_values = points
    global current_x_values 
    current_x_values = x_values
    return points

#iterates through temperature values to make points for viscosity graph
def viscosity_points(atomic_mass, x_values, species_list, temp_list, density_list):
    points = np.array([])
    density = density_list[1]
    temp = temp_list[1]
    species = species_list[1]
    #checks to see what variable is being used for x axis
    #checks for species
    if np.array_equal(x_values, np.geomspace(species_list[0], species_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, temp, x, units_out = 'cgs')
             eta = sm.viscosity()
             points = np.append(points, eta)
             
     #checks for temp
    elif np.array_equal(x_values, np.geomspace(temp_list[0], temp_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, x, species, units_out = 'cgs')
             eta = sm.viscosity()
             points = np.append(points, eta)
             
     #checks for density
    elif np.array_equal(x_values, np.geomspace(density_list[0], density_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, x, temp, species, units_out = 'cgs')
             eta = sm.viscosity()
             points = np.append(points, eta)
      
    global current_vc_values 
    current_vc_values = points
    global current_x_values 
    current_x_values = x_values
    return points

#iterates through temperature values to make points for self diffusion graph
def self_diffusion_points(atomic_mass, x_values, species_list, temp_list, density_list):
     points = np.array([])
     density = density_list[1]
     temp = temp_list[1]
     species = species_list[1]
     #checks to see what variable is being used for x axis
     #checks for species
     if np.array_equal(x_values, np.geomspace(species_list[0], species_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, temp, x, units_out = 'cgs')
             D = sm.self_diffusion()
             points = np.append(points, D)
             
     #checks for temp
     elif np.array_equal(x_values, np.geomspace(temp_list[0], temp_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, x, species, units_out = 'cgs')
             D = sm.self_diffusion()
             points = np.append(points, D)
             
     #checks for density
     elif np.array_equal(x_values, np.geomspace(density_list[0], density_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, x, temp, species, units_out = 'cgs')
             D = sm.self_diffusion()
             points = np.append(points, D)
             
     global current_sd_values 
     current_sd_values = points
     global current_x_values 
     current_x_values = x_values
     return points


#iterates through temperature values to make points for thermal conductivity graph
def thermal_conductivity_points(atomic_mass, x_values, species_list, temp_list, density_list):
    points = np.array([])
    density = density_list[1]
    temp = temp_list[1]
    species = species_list[1]
    #checks to see what variable is being used for x axis
    #checks for species
    if np.array_equal(x_values, np.geomspace(species_list[0], species_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, temp, x, units_out = 'cgs')
             K = sm.thermal_conductivity()
             points = np.append(points, K)
             
     #checks for temp
    elif np.array_equal(x_values, np.geomspace(temp_list[0], temp_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, density, x, species, units_out = 'cgs')
             K = sm.thermal_conductivity()
             points = np.append(points, K)
             
     #checks for density
    elif np.array_equal(x_values, np.geomspace(density_list[0], density_list[1])):
         for x in x_values:
             sm = transport.SM(atomic_mass, x, temp, species, units_out = 'cgs')
             K = sm.thermal_conductivity()
             points = np.append(points, K)
             
    global current_tc_values 
    current_tc_values = points
    global current_x_values 
    current_x_values = x_values
    return points

if __name__ == '__main__':
    app.run_server(debug=True)