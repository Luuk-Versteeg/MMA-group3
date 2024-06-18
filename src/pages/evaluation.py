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


evaluation = html.Div(children=[
    html.H1(children='Evaluation', style={'textAlign':'center'}),
    html.Button('Run prompts', id='run-all-prompts-btn'),
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
        id='evaluation-table'
    ), style={"marginTop": "30px", "marginBottom": "30px"}),
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
        id='confusion-matrix'
    ), style={"marginTop": "30px", "marginBottom": "30px", "width": "500px"})
])


@callback(
    Output("evaluation-table", "rowData"),
    Output("evaluation-table", "columnDefs"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
    Input("run-all-prompts-btn", "value")
)
def update_evaluation_table(dataset_name, dataset_split, button_clicked):
    return [], []


@callback(
    Output("confusion-matrix", "rowData"),
    Output("confusion-matrix", "columnDefs"),
    Input("evaluation-table", "rowData"),
    Input("generated-prompts-container", "children"),
    Input("run-all-prompts-btn", "value")

)
def update_confusion_matrix(data, prompts, button_clicked):
    return [], []
