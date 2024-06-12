from dash import Dash, html, dcc, Output, Input, State, ctx, callback, dependencies
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict
import itertools

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
                    id='textarea-prompt',
                    value='{var1}{text}\n\n{var2}',
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
                html.Button('Test prompts', id='button-test-prompts')
            ], 
            style={'width':'100%'}
        ),
        html.Div(id='generated-prompts-container', children=[
            html.Div(children='{var1}{text}\n\n{var2}')
        ], style={'width':'100%'})
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
    State('variable-container', 'value'),
    State('variant-store','data')
)
def update_tabs(add_clicks, remove_clicks, tabs, active_tab, all_variants):
    ctx_id = ctx.triggered_id

    if ctx_id == 'add-variable-button':
        new_index = len(tabs) + 1 
        new_tab_value = f'{new_index}'
        new_tab = dcc.Tab(label=f'Variable {new_index}', value=new_tab_value, 
                          id={'type': 'tab', 'index': new_index},
                          style={'width':'100%', 'line-width': '100%'},
                          selected_style={'width':'100%', 'line-width': '100%'})
        tabs.append(new_tab)
        active_tab = new_tab_value

    elif ctx_id == 'remove-variable-button' and active_tab is not None:
        if active_tab in all_variants:
            all_variants.pop(active_tab, None)

        tabs = [tab for tab in tabs if tab['props']['value'] != active_tab]
        if tabs:
            active_tab = tabs[0]['props']['value']
        else:
            active_tab = None

    return tabs, active_tab


@callback(
    Output('variant-container', 'children'),
    Output('variant-store', 'data', allow_duplicate=True),
    Input('add-variant-button', 'n_clicks'),
    Input('variable-container', 'value'),
    State('variant-container', 'children'),
    State('variable-container', 'value'),
    State('variant-store', 'data',),
    prevent_initial_call=True
)
def update_variants(add_clicks, pressed_tab, variants, tabs_state, all_variants):
    ctx_id = ctx.triggered_id

    # Add variants.
    if ctx_id == 'add-variant-button':
        if tabs_state == None:
            return variants

        idx = len(variants) + 1

        if tabs_state not in all_variants:
            all_variants[tabs_state] = {}
        all_variants[tabs_state][idx] = f"Variant {idx}"


        new_variant = html.Div(style={'display':'grid', 'grid-template-columns':'80% 20%','height':40}, children=[
            dcc.Input(id={'type': 'variant-input', 'index': idx},
                    value=all_variants[tabs_state][idx],
                    type='text'),
            html.Button('X', id='button-delete'),
        ])
        variants.append(new_variant)
    # Add variables.
    elif ctx_id == 'variable-container':
        if pressed_tab in all_variants:
            vars = [html.Div(style={'display':'grid', 'grid-template-columns':'80% 20%','height':40}, children=[
                dcc.Input(id={'type': 'variant-input', 'index': int(idx)},
                        value=val,
                        type='text'),
                html.Button('X', id='button-delete'),
            ]) for idx, val in all_variants[pressed_tab].items()]

            variants = vars
        else:
            variants = []


    return variants, all_variants


@callback(
    Output('variant-store','data'),
    [Input({'type': 'variant-input', 'index': dependencies.ALL}, 'value')],
    State('variable-container', 'value'),
    State('variant-store','data')
)
def update_variant_dict(values, selected_tab, all_variants):
    ctx_id = ctx.triggered_id
    if not ctx_id:
        return all_variants
    all_variants[selected_tab][str(ctx_id['index'])] = values[ctx_id['index'] - 1]
    return all_variants
    

@callback(
    Output('generated-prompts-container', 'children'),
    Input('button-generate-prompts', 'n_clicks'),
    State('variant-store', 'data'),
    State('textarea-prompt', 'value'),
)
def generate_prompts(generate_clicks, data, prompt):
    ctx_id = ctx.triggered_id
    if not ctx_id:
        return []

    vars = []
    for var_num, variants in data.items():
        vars.append([(var_num, var) for var in list(variants.values())])
    permurations = list(itertools.product(*vars))

    generated_prompts = []
    print("=======")
    for perm in permurations:
        new_prompt = prompt
        for variable in perm:
            new_prompt = new_prompt.replace('{var' + variable[0] + '}', variable[1])

        # Convert to list of lines with html.Br() instead of \n.
        prompt_lines = []
        for line in new_prompt.split('\n'):
            prompt_lines.append(line)
            prompt_lines.append(html.Br())
        prompt_lines = prompt_lines[:-1]

        generated_prompts.append(html.Div(children=prompt_lines, style={'border':'2px solid #000', 'height':200, 'width':200, 'display':'inline-block'}))
# return (html.P(["Model:{}".format(m),html.Br(),
#                 "Prediction:{}".format(y_pred[i]),
#                 html.Br(),"Probability for Yes:{}".format(yes),
#                 html.Br(),"Probability for No:{}".format(no)]))
    # for var_num, variants in data.items():
    #     new_prompt = prompt.replace("{var" + var_num + "}", list(variants.values())[0])

    # for var_num, variants in data.items():

    #     print("=========")
    #     print("var" + var_num)
    #     for variant in variants.values():
    #         print("----------")
    #         print("lol" + variant)
    #         new_prompt = prompt.replace("{var" + var_num + "}", variant)
    #         #print(var_num, variant)
    #         print(new_prompt)    
    return generated_prompts


@callback(
    Output('graph-content3', 'figure'),
    Input('dropdown-selection3', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')


if __name__ == '__main__':
    app.run(debug=True)
