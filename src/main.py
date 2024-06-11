from dash import Dash, html, dcc, Output, Input, State, ctx, callback
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash(__name__)

all_variants = defaultdict(dict)

app.layout = dbc.Container([
    html.Div(children=[
        html.H1(children='First page', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection1'),
        dcc.Graph(id='graph-content1')
    ]),
    html.Div(children=[
        html.H1(children='Second page', style={'textAlign':'center'}),
        html.Div(children=[
            html.Div(
                className='box',
                children=[
                    html.Button('<', id='button-left', style={'height':'100%', 'width':50, 'display':'inline-block'}),
                    html.Div(style={'height':'100%', 'flex':1, 'text-align':'center'}, children='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'),
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
                    id='textarea-example',
                    value='{q2}{text}\n\n{q1}',
                    style={'width':'58%', 'height':50, 'display':'inline-block', 
                        'resize':'vertical'},
                ),
                dcc.Input(
                    id='input-test1',
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
                        # dcc.Input(
                        #     id='input-test2',
                        #     value='variant',
                        #     type='text',
                        #     style={'width':'100%', 'height': 35},
                        # )
                    ]),
                ],
                style={'width':'47%', 'height':400, 'display':'inline-block'},
            )
            ],
            style={'display':'flex', 'justify-content':'space-between'}
        ),
        html.Div(children=[
                html.Button('Generate prompts', id='button-generate'),
                html.Button('Test prompts', id='button-test')
            ], 
            style={'width':'100%'}
        ),
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
    Output('variable-container', 'children'),
    Output('variable-container', 'value'),
    Input('add-variable-button', 'n_clicks'),
    Input('remove-variable-button', 'n_clicks'),
    State('variable-container', 'children'),
    State('variable-container', 'value')
)
def update_tabs(add_clicks, remove_clicks, tabs, active_tab):
    ctx_id = ctx.triggered_id

    if ctx_id == 'add-variable-button':
        new_index = len(tabs) + 1 
        new_tab_value = f'tab-{new_index}'
        new_tab = dcc.Tab(label=f'Variable {new_index}', value=new_tab_value, 
                          id={'type': 'tab', 'index': new_index},
                          style={'width':'100%', 'line-width': '100%'},
                          selected_style={'width':'100%', 'line-width': '100%'})
        tabs.append(new_tab)
        active_tab = new_tab_value

    elif ctx_id == 'remove-variable-button' and active_tab is not None:
        if active_tab in all_variants:
            #all_variants.pop(active_tab, None)
            all_variants[active_tab] = {}

        tabs = [tab for tab in tabs if tab['props']['value'] != active_tab]
        if tabs:
            active_tab = tabs[0]['props']['value']
        else:
            active_tab = None

    return tabs, active_tab


@callback(
    Output('variant-container', 'children'),
    Input('add-variant-button', 'n_clicks'),
    Input('variable-container', 'value'),
    State('variant-container', 'children'),
    State('variable-container', 'value')
)
def update_variants(add_clicks, pressed_tab, variants, tabs_state):
    ctx_id = ctx.triggered_id

    if ctx_id == 'add-variant-button':
        if tabs_state == None:
            return variants

        idx = len(variants) + 1
        all_variants[tabs_state][idx] = f"Variant {idx}"

        new_variant = html.Div(style={'display':'grid', 'grid-template-columns':'80% 20%','height':40}, children=[
            dcc.Input(id='input-variant-' + str(idx),
                    value=all_variants[tabs_state][idx],
                    type='text'),
            html.Button('X', id='button-delete'),
        ])
        variants.append(new_variant)

    elif ctx_id == 'variable-container':
        if pressed_tab in all_variants:
            vars = [html.Div(style={'display':'grid', 'grid-template-columns':'80% 20%','height':40}, children=[
                dcc.Input(id='input-variant-' + str(idx),
                        value=val,
                        type='text'),
                html.Button('X', id='button-delete'),
            ]) for idx, val in all_variants[pressed_tab].items()]

            variants = vars
        else:
            variants = []

    return variants


@callback(
    Output('graph-content3', 'figure'),
    Input('dropdown-selection3', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')


if __name__ == '__main__':
    app.run(debug=True)
