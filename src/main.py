from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash(__name__)

app.layout = dbc.Container([
    html.Div(children=[
        html.H1(children='First page', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection1'),
        dcc.Graph(id='graph-content1')
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
    Output('graph-content1', 'figure'),
    Input('dropdown-selection1', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')

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
