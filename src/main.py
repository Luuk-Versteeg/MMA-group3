from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import dash_ag_grid

from widgets import histogram, table
from dataloaders.load_data import datasets

app = Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


# import pdb; pdb.set_trace()

app.layout = dbc.Container([
    html.Div(children=[
        html.H1(children='Data selection', style={'textAlign':'center'}),
        html.Div(children=[
            html.Div(children=[
                dcc.Dropdown([d["name"] for d in datasets], value=datasets[0]["name"], id='dataset-selection', style={"width": "100%"}),
                html.Div(id="dataset-details"),
                # dcc.Graph(id='graph-content1')
            ], style={"width": "48%"}),
            html.Div(dcc.Tabs(children=[
                dcc.Tab(label="Label frequency", children=histogram.create_histogram()),
                dcc.Tab(label="Words", children="")
            ]), style={"width": "48%"})
        ], style={"display": "flex", "justifyContent": "space-between"}),
        html.Div(children=dash_ag_grid.AgGrid(
            
        ), style={"marginTop": "30px", "marginBottom": "30px"})
    ]),
    html.Div(children=[
        html.H1(children='Second page', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Italy', id='dropdown-selection2'),
        dcc.Graph(id='graph-content2')
    ]),
    html.Div(children=[
        html.H1(children='Third page', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Spain', id='dropdown-selection3'),
        dcc.Graph(id='graph-content3')
    ])
], id="carousel")

@callback(
    Output('dataset-details', 'children'),
    Input('dataset-selection', 'value')
)
def update_dataset_details(value):
    dataset = [d for d in datasets if d["name"] == value][0]
    data_splits = [s for s in dataset["data"].keys()]
    
    return [
        dcc.Dropdown(data_splits, value=data_splits[0], id="dataset-split", style={"marginTop": "10px", "width": "200px"}),
        html.P(children=[
            html.P(children="Select the number of samples:"),
            dbc.Input(type="number", min=0, max=100, value=20, step=1)
        ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
        html.P(children=f'Description: {dataset["description"]}'), 
        html.P(children=f'Scheme: {dataset["scheme"]}')
    ]

# @callback(
#     Output('graph-content2', 'figure'),
#     Input('dropdown-selection2', 'value')
# )
# def update_graph(value):
#     dff = df[df.country==value]
#     return px.line(dff, x='year', y='pop')

@callback(
    Output('graph-content3', 'figure'),
    Input('dropdown-selection3', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')

if __name__ == '__main__':
    app.run(debug=True)
