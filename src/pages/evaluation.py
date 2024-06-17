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
    html.H1(children='Prompt Engineering', style={'textAlign':'center'}),
    html.Div(children=[
        html.Div(
            className='box',
            children=[
                html.Button('<', id='button-left', style={'height':'100%', 'width':50, 'display':'inline-block'}),
                html.Div(id="prompt-sample", style={'height':'100%', 'flex':1, 'text-align':'center'}),
                html.Button('>', id='button-right', style={'height':'100%', 'width':50, 'display':'inline-block'})
            ],
            style={'width':'50%', 'height':100, 'display':'flex'},
        ),
        html.Div(
            children='labels: ',
            style={'width':'45%', 'height':100, 'display':'inline-block'},
        )
    ],
    style={'display':'flex', 'justify-content':'space-between'}
    ),
    html.Div(children=[
            dcc.Textarea(
                id='textarea-prompt',
                value='{var1}{text}\n\n{var2}',
                style={'width':'58%', 'height':50, 'display':'inline-block', 
                    'resize':'vertical'},
            ),
            dcc.Input(
                id='possible-answers',
                value='Business, World, Science, Sports',
                type='text',
                style={'width':'38%', 'height': 50, 'display': 'inline-block'},
            )
        ], 
        style={'display':'flex', 'justify-content':'space-between'}
    ),
    html.Div(children=[
        html.Div(
            className='box',
            children=[
                html.Div(className='box', children=[
                    html.Button('Add variable', id='add-variable-button', n_clicks=0),
                    html.Button('Remove selected variable', id='remove-variable-button')
                ]),
                dcc.Tabs(content_style={"width": "100%"},
                            parent_style={"width": "100%"},
                            style={'width':'100%'},
                            id="variable-container", children=[], vertical=True),

            ],
            style={'width':'47%', 'height':400},
        ),
        html.Div(
            className='box',
            children=[
                html.Div(
                    id='tabs-content',
                    children=[ 
                                html.Button('Add variant', id='add-variant-button', n_clicks=0),
                                ],
                    style={'display':'flex',
                            'justify-content':'center',
                            'align-items':'center', 
                            'height':50}
                ),
                html.Div(id='variant-container', children=[
                ]),
                dcc.Store(id='variant-store', data=defaultdict(dict))
            ],
            style={'width':'47%', 'height':400, 'display':'inline-block'},
        )
        ],
        style={'display':'flex', 'justify-content':'space-between'}
    ),
    html.Div(children=[
            html.Button('Generate prompts', id='button-generate-prompts'),
            html.Button('Test prompts', id='button-test-prompts'),
            'Select number of samples',
            html.Div(children=dcc.Dropdown([10, 20, 50], 10, id='select-num-samples'))
        ], 
        style={'width':'100%'}
    ),
    html.Div(id='generated-prompts-container', children=[
    ], style={'width':'100%'}),
    html.Div(id='tested-prompts-container', children=[
        html.Div(children='test prompt')
    ], style={'width':'100%'})
])