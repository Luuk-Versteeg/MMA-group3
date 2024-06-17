from dash import Dash, html, dcc, Output, Input, State, ctx, callback, dependencies
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict
import itertools
import plotly.graph_objects as go
import dash_ag_grid


import plotly
import random
import nltk
from nltk.corpus import stopwords
from collections import Counter


from widgets import histogram
from dataloaders.load_data import datasets


data_selection = html.Div(children=[
    html.H1(children='Data selection', style={'textAlign':'center'}),
    html.Div(children=[
        html.Div(children=[
            dcc.Dropdown([d["name"] for d in datasets], placeholder="Select dataset", id='dataset-selection', style={"width": "100%"}),
            html.Div(children=[
                dcc.Dropdown([], placeholder="Select subset", id="dataset-split", style={"marginTop": "10px", "width": "200px"}),
                html.P(children=[
                    html.P(children="Select the number of samples:"),
                    dbc.Input(type="number", min=0, value=10, max=100, step=1, id="n-samples"),
                    html.P(children="(max: )", id="max-samples")
                ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
                html.P(id="dataset-description"), 
                html.P(children=f'Scheme:', id="dataset-scheme")
            ]),
            html.Div(id="selected-sample", style={"padding": "15px 30px", "border": "1px solid black", "margin": "0px 20px", "marginTop": "30px"})
        ], style={"width": "48%"}),
        html.Div(dcc.Tabs(children=[
            dcc.Tab(label="Labels", children=histogram.create_histogram(id="label-histogram")),
            dcc.Tab(label="Words frequency", children=histogram.create_histogram(id="wordcloud"))
        ]), style={"width": "48%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),
    html.Div(children=dash_ag_grid.AgGrid(
        columnDefs=[],
        rowData=[],
        columnSize="responsiveSizeToFit",
        dashGridOptions={
            "pagination": False,
            "paginationAutoPageSize": True,
            "suppressCellFocus": True,
            "rowSelection": "single",
        },
        selectedRows=[],
        # defaultColDef={"filter": "agTextColumnFilter"},
        # className='stretchy-widget ag-theme-alpine',
        # style={'width': '', 'height': ''},
        id='samples-table'
    ), style={"marginTop": "30px", "marginBottom": "30px"})
])