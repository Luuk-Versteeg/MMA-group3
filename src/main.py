from dash import Dash, html, dcc, Output, Input, State, ctx, callback, dependencies
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict
import itertools
import plotly.graph_objects as go
import dash_ag_grid

from widgets import histogram
from dataloaders.load_data import datasets

app = Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app.layout = dbc.Container([
    html.Div(children=[
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
                dcc.Tab(label="Words frequency", children=histogram.create_histogram(id="word-histogram"))
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
    ]),
    html.Div(children=[
        html.H1(children='Third page', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Spain', id='dropdown-selection3'),
        dcc.Graph(id='graph-content3')
    ])
], id="carousel")


def select_dataset(name, datasets=datasets):
    """Selects the dataset dictionary from the list of datasets if the name exists."""
    dataset = [d for d in datasets if d["name"] == name]

    if len(dataset) == 0:
        raise KeyError(f"There's no dataset called: {name}")
    
    return dataset[0]


@callback(
    Output("dataset-split", "options"),
    Output("dataset-description", "children"),
    Output("dataset-scheme", "children"),
    Input("dataset-selection", "value")
)
def update_dataset_details(dataset_name):

    split = list()
    description = "Description: "
    scheme = "Scheme: "

    if dataset_name:
        dataset = select_dataset(dataset_name)

        split += list(dataset["data"].keys())
        description += dataset["description"]
        scheme += dataset["scheme"]
    
    return split, description, scheme


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
        possible_indices = [tab['props']['id']['index'] for tab in tabs]
        new_index = len(tabs) + 1 
        i = 1
        while True:
            if i not in possible_indices:
                new_index = i
                break
            i += 1

        new_tab_value = f'{new_index}'
        new_tab = dcc.Tab(label=f'var{new_index}', value=new_tab_value, 
                          id={'type': 'tab', 'index': new_index},
                          style={'width':'100%', 'line-width': '100%'},
                          selected_style={'width':'100%', 'line-width': '100%'})
        tabs.append(new_tab)
        active_tab = new_tab_value

    elif ctx_id == 'remove-variable-button' and active_tab is not None:
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
    Input({'type': 'variant-input', 'index': dependencies.ALL}, 'value'),
    Input('remove-variable-button', 'n_clicks'),
    State('variable-container', 'value'),
    State('variant-store','data')
)
def update_variant_dict(values, removes, selected_tab, all_variants):
    ctx_id = ctx.triggered_id
    if not ctx_id:
        return all_variants
    
    if ctx_id == 'remove-variable-button':
        #all_variants[selected_tab] = {}
        all_variants.pop(selected_tab, None)
    else:
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
    for idx, perm in enumerate(permurations, start=1):
        new_prompt = prompt
        for variable in perm:
            new_prompt = new_prompt.replace('{var' + variable[0] + '}', variable[1])

        # Convert to list of lines with html.Br() instead of \n.
        prompt_lines = []
        for line in new_prompt.split('\n'):
            prompt_lines.append(line)
            prompt_lines.append(html.Br())
        prompt_lines = prompt_lines[:-1]

        generated_prompts.append(html.Div(id={'type': 'generated-prompt', 'index': int(idx)}, children=prompt_lines, style={'border':'2px solid #000', 'height':200, 'width':200, 'display':'inline-block'}))

    return generated_prompts


@callback(
    Output('tested-prompts-container', 'children'),
    Input('button-test-prompts', 'n_clicks'),
    State('select-num-samples', 'value'),
    State('generated-prompts-container', 'children'),
    State('possible-answers', 'value')
    #State({'type': 'generated-prompt', 'index': dependencies.ALL}, 'children')
)
def test_prompts(test_button, num_samples, generated_prompts, possible_answers):
    return []


@callback(
    Output("samples-table", "columnDefs"),
    Output("samples-table", "rowData"),
    Output("samples-table", "selectedRows"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
    Input("n-samples", "value")
)
def update_grid(dataset_name, dataset_split, n_samples):

    cols, rows, selected = list(), list(), list()

    if not dataset_name or not dataset_split:
        return cols, rows, selected
    
    dataset = select_dataset(dataset_name)

    if dataset_split not in dataset["data"].keys():
        return cols, rows, selected
    
    df = dataset["data"][dataset_split].head(n_samples)
    cols = [{"field": col} for col in df.columns.to_list()]
    rows = df.to_dict("records")
    selected = [rows[0]]

    return cols, rows, selected

@callback(
    Output("selected-sample", "children"),
    Input("samples-table", "selectedRows")
)
def update_sample(selected_rows):

    if len(selected_rows) == 0:
        return html.P("No sample selected...")

    output = []

    for key, value in selected_rows[0].items():
        output.append(html.P(f"{key}: {value}"))

    return output


@callback(
    Output("max-samples", "children"),
    Output("n-samples", "max"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
)
def update_max_samples(dataset_name, dataset_split):

    if not dataset_name or not dataset_split:
        return "(max: 0)", 0
    
    dataset = select_dataset(dataset_name)

    if dataset_split not in dataset["data"].keys():
        return "(max: 0)", 0
    
    df = dataset["data"][dataset_split]
    max_samples = len(df)

    return f"(max: {max_samples})", max_samples


@callback(
    Output("word-histogram", "figure"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
)
def update_word_histogram(dataset_name, dataset_split):
    #TODO WORD HISTOGRAM
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="Select data on the scatterplot",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28, color="gray")
            )
        ],
        margin=dict(b=0, l=0, r=0, t=100)  # Adjust margins to ensure the text is visible
    )

    return fig

@callback(
    Output("label-histogram", "figure"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
)
def update_label_histogram(dataset_name, dataset_split):
    #TODO LABEL HISTOGRAM
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="Select data on the scatterplot",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28, color="gray")
            )
        ],
        margin=dict(b=0, l=0, r=0, t=100)  # Adjust margins to ensure the text is visible
    )

    return fig


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
