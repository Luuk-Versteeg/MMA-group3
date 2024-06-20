from dash import html, dcc, Output, Input, State, ctx, callback, dependencies
from collections import defaultdict
import itertools


prompt_engineering = html.Div(children=[
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


