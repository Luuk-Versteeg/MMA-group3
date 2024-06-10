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
        html.Div(children=[
            html.Div(
                className='box',
                children=[
                    html.Button('<', id='button-generate', style={'height':'100%', 'width':50, 'display':'inline-block'}),
                    html.Div(style={'height':'100%', 'flex':1, 'text-align':'center'}, children='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'),
                    html.Button('>', id='button-test', style={'height':'100%', 'width':50, 'display':'inline-block'})
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
                id='input-example',
                value='Business, World, Science, Sports',
                type='text',
                style={'width':'38%', 'height': 50, 'display': 'inline-block'},
            )], 
            style={'display':'flex', 'justify-content':'space-between'}
        ),
        html.Div(children=[
            html.Div(
                className='box',
                children=[
                    dcc.Tabs(id="tabs", value='var1', children=[
                        dcc.Tab(label='var1', value='var1'),
                        dcc.Tab(label='var2', value='var2'),
                        dcc.Tab(label='var3', value='var3'),
                    ]),
                    html.Div(id='tabs-content')
                ],
                style={'width':'47%', 'height':400, 'display':'inline-block'},
            ),
            html.Div(
                className='box',
                children=[
                    html.Div(
                        id='tabs-content',
                        children='varX',
                        style={'display':'flex',
                               'justify-content':'center',
                               'align-items':'center', 
                               'height':50}
                    ),
                    dcc.Input(
                        id='input-example',
                        value='A',
                        type='text',
                        style={'width':'100%', 'height': 35},
                    ), 
                    dcc.Input(
                        id='input-example',
                        value='B',
                        type='text',
                        style={'width':'100%', 'height': 35},
                    ),
                    dcc.Input(
                        id='input-example',
                        value='C',
                        type='text',
                        style={'width':'100%', 'height': 35},
                    )
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
        )
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

@callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    print(tab)
    if tab == 'var1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'var2':
        return html.Div([
            html.H3('Tab content 2')
        ])



if __name__ == '__main__':
    app.run(debug=True)
