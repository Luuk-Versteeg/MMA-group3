from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

from dataloaders.load_data import datasets

app = Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


# import pdb; pdb.set_trace()

app.layout = dbc.Container([
    html.Div(children=[
        html.H1(children='Data selection', style={'textAlign':'center'}),
        dcc.Dropdown([d["name"] for d in datasets], value=datasets[0]["name"], id='dropdown-selection1'),
        html.Div(id="details"),
        # dcc.Graph(id='graph-content1')
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
    Output('details', 'children'),
    Input('dropdown-selection1', 'value')
)
def update_graph(value):
    dataset = [d for d in datasets if d["name"] == value][0]
    return [html.P(children=f'Description: {dataset["description"]}'), html.P(children=f'Scheme: {dataset["scheme"]}')]

@callback(
    Output('graph-content2', 'figure'),
    Input('dropdown-selection2', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')

@callback(
    Output('graph-content3', 'figure'),
    Input('dropdown-selection3', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')

if __name__ == '__main__':
    app.run(debug=True)
