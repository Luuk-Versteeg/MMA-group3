from dash import html, dcc, Output, Input, State, ctx, callback, dependencies
from collections import defaultdict
import itertools

from .tinyllama import sent_classifier, news_classifier, snellius
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)



prompt_engineering = html.Div(children=[
    html.H1(children='Prompt Engineering', style={'textAlign':'center'}),
    html.Div(children=[
        html.Div(
            className='selected-sample',
            children=[
                html.Button('<', id='button-left', style={"padding": "30px"}),
                html.Div(id="prompt-sample", 
                         children=" Improved Pitching Has the Keys Finishing on an Upswing As the season winds down for the Frederick Keys, Manager Tom Lawless is starting to enjoy the progress his pitching staff has made this season.",
                         style={"padding": "15px 30px", "border": "1px solid black", "width": "100%", 
                                "minHeight": "100px", "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Button('>', id='button-right', style={"padding": "30px"})
            ],
            style={'width':'60%', 'display':'flex', 'gap': '20px', 'alignItems': "center"},
        ),
        html.Div(children=[
            html.P(
                id='possible-answers',
                children='No dataset selected...',
            ),
            html.P(
                children='label: ',
                id='prompt-label',
                style={"margin": "0"}
            )
        ], style={"width": "38%", "display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"})
    ],
    style={'display':'flex'}
    ),

    html.Div(children=[
        html.P("Create a prompt template:", style={"marginBottom": "5px"}),
        dcc.Textarea(
            id='textarea-prompt',
            value='{var1} {text}\n\n{var2}',
            style={'width':'100%', 'height':'75px', 'display':'inline-block', 
                'resize':'vertical', 'padding': '10px', 'boxSizing': 'border-box'},
        ),
    ], style={"width": "70%", "marginTop": "10px"}),

    html.Div(children=[
        html.Div(
            className='box',
            children=[
                html.Div(children=[
                    html.Button('Add variable', id='add-variable-button', n_clicks=0),
                    html.Button('Remove selected variable', id='remove-variable-button')
                ], style={'display': 'flex', 'gap': '15px', 'padding': '10px 20px', 'justifyContent': 'center'}),
                dcc.Tabs(
                    content_style={"width": "100%"},
                    parent_style={"width": "100%"},
                    style={'width':'100%'},
                    id="variable-container", children=[], vertical=True),
            ],
            style={'width':'47%', 'overflowY': 'scroll'},
        ),
        html.Div(
            className='box',
            children=[
                html.Div(
                    id='tabs-content',
                    children=[html.Button('Add variant', id='add-variant-button', n_clicks=0)],
                    style={'display':'flex', 'justify-content':'center', 'padding': '10px 20px'}
                ),
                html.Div(id='variant-container', children=[], style={"marginBottom": "15px"}),
                dcc.Store(id='variant-store', data=defaultdict(dict))
            ],
            style={'width':'47%', 'overflowY': 'scroll'},
        )
        ],
        style={'display':'flex', 'justify-content':'space-between', 'marginTop': '20px', 'height': '250px'}
    ),
    html.Div(children=[
            dcc.Store('prompt-list'),
            dcc.Store('true-label'),
            html.Button('Generate prompts', id='button-generate-prompts'),
            html.Button('Test prompts', id='button-test-prompts')
        ], 
        style={'width':'100%', 'margin': '20px 0px', 'display': 'flex', 'gap': '15px', 'alignItems': 'center', 'justifyContent': 'center'}
    ),
    html.Div(id='generated-prompts-container', children=[
    ], style={'width':'100%', 'maxHeight': '300px', 'overflowY': 'scroll', 'display': 'flex', 'gap': '15px', 'marginTop': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

    html.Div(id='tested-prompts-container', children=[
        html.Div(children='test prompt')
    ], style={'width':'100%', 'maxHeight': '300px', 'overflowY': 'scroll', 'display': 'flex', 'gap': '15px', 'marginTop': '20px', 'flexWrap': 'wrap', 'justifyContent': 'center'})

])


@callback(
    Output('generated-prompts-container', 'children', allow_duplicate=True),
    Output('prompt-list', 'data'),
    Input('button-generate-prompts', 'n_clicks'),
    State('variant-store', 'data'),
    State('textarea-prompt', 'value'),
    prevent_initial_call=True
)
def generate_prompts(generate_clicks, data, prompt):
    ctx_id = ctx.triggered_id
    if not ctx_id:
        return [], []

    # Extract variables and create permutations
    vars = []
    for var_num, variants in data.items():
        vars.append([(var_num, var) for var in list(variants.values())])
    permutations = list(itertools.product(*vars))

    generated_prompts = []
    prompt_list = []

    def get_synonym(word):
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Get the first synonym
            synonym = synonyms[0].lemmas()[0].name()
            if synonym.lower() != word.lower():
                return synonym
        return word

    # Generate prompts for each permutation
    for idx, perm in enumerate(permutations, start=1):
        new_prompt = prompt
        for variable in perm:
            new_prompt = new_prompt.replace('{var' + variable[0] + '}', variable[1])
        
        # Store the prompt string
        prompt_list.append(new_prompt)

        # Convert to list of lines with html.Br() instead of \n, with synonym tooltips
        prompt_lines = []
        for line in new_prompt.split('\n'):
            line_words = line.split()
            for word in line_words:
                synonym = get_synonym(word)
                prompt_lines.append(
                    html.Span(
                        word,
                        style={'cursor': 'pointer'},
                        title=f'Synonym: {synonym}'
                    )
                )
                prompt_lines.append(' ')  # Add space between words
            prompt_lines.append(html.Br())
        prompt_lines = prompt_lines[:-1]  # Remove the last html.Br()

        # Create a div for the generated prompt
        generated_prompts.append(html.Div(
            id={'type': 'generated-prompt', 'index': int(idx)},
            children=prompt_lines,
            style={'border': '1px solid #000', 'height': 200, 'width': 200, 'padding': 15, 'boxSizing': 'border-box', 'display': 'inline-block'}
        ))

    return generated_prompts, prompt_list


@callback(
    # Output('tested-prompts-container', 'children'),
    Output('generated-prompts-container', 'children', allow_duplicate=True),
    Input('button-test-prompts', 'n_clicks'),
    Input("dataset-selection", "value"),
    State('true-label', 'data'),
    State('prompt-list', 'data'),
    State('prompt-sample', 'children'),
    prevent_initial_call = True
)
def test_prompts(test_button, dataset_name, true_label, generated_prompts, text):
    if not test_button:
        return []

    if dataset_name == "AG News":
        classifier = news_classifier
    if dataset_name == "Amazon Polarity" or dataset_name == "GLUE/sst2":
        classifier = sent_classifier

    pred_labels = []

    if snellius:
        with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on your hardware
            future_to_prompt = {executor.submit(classifier, prompt.format(text=text)): prompt for prompt in generated_prompts}
            total_prompts = len(generated_prompts)
            for future in tqdm(as_completed(future_to_prompt), total=total_prompts):
                label, answer = future.result()
                pred_labels.append(label)
    else:
        for prompt in tqdm(generated_prompts):
            prompt = prompt.format(text=text)
            pred_label, answer = classifier(prompt)
            pred_labels.append(pred_label)

    # Convert to list of lines with html.Br() instead of \n.
    colored_prompt_divs = []
    for idx, (pred_label, new_prompt) in enumerate(zip(pred_labels, generated_prompts)):
        if pred_label == true_label:
            color='green' 
        else:
            color='red'
        prompt_lines = []
        for line in new_prompt.split('\n'):
            prompt_lines.append(line)
            prompt_lines.append(html.Br())
        prompt_lines.append(html.Hr())
        prompt_lines.append(f"Predicted: {pred_label}")

        colored_prompt_divs.append(html.Div(
            id={'type': 'generated-prompt', 'index': int(idx)}, 
            children=prompt_lines, 
            style={'border':'1px solid #000', 'height':200, 'width':200, 'padding': 15, 'boxSizing': 'border-box', 'display':'inline-block', 'background-color':color}))

    return colored_prompt_divs


@callback(
    Output('variant-container', 'children'),
    Output('variant-store', 'data', allow_duplicate=True),
    Input('add-variant-button', 'n_clicks'),
    Input('variable-container', 'value'),
    State('variant-container', 'children'),
    State('variable-container', 'value'),
    State('variant-store', 'data'),
    prevent_initial_call=True
)
def update_variants(add_clicks, pressed_tab, variants, tabs_state, all_variants):
    ctx_id = ctx.triggered_id

    # Add variants.
    if ctx_id == 'add-variant-button':
        if tabs_state == None:
            return variants, all_variants

        if tabs_state not in all_variants:
            all_variants[tabs_state] = {}

        idx = 1
        while (True):
            if str(idx) not in all_variants[tabs_state].keys():
                idx = str(idx)
                break
            idx += 1
        all_variants[tabs_state][idx] = f"Variant {idx}"

        new_variant = html.Div(style={'display':'grid', 'grid-template-columns':'77% 20%', 'justifyContent': 'space-between', 'height':40, 'padding': '5px 15px'}, children=[
            dcc.Input(id={'type': 'variant-input', 'index': int(idx)},
                    value=all_variants[tabs_state][idx],
                    type='text',
                    style={"paddingLeft": "15px"}),
            html.Button('X', id={'type': 'variant-button-delete', 'index': f"{tabs_state}-{idx}"}),
        ])
        variants.append(new_variant)
    # Add variables.
    elif ctx_id == 'variable-container':
        if pressed_tab in all_variants:
            vars = [html.Div(style={'display':'grid', 'grid-template-columns':'77% 20%', 'justifyContent': 'space-between', 'height': "40px",'padding': '5px 15px'}, children=[
                dcc.Input(id={'type': 'variant-input', 'index': int(idx)}, value=val, type='text',
                          style={"paddingLeft": "15px"}),
                html.Button('X', id={'type': 'variant-button-delete', 'index': f"{pressed_tab}-{idx}"}),
            ]) for idx, val in all_variants[pressed_tab].items()]

            variants = vars
        else:
            variants = []

    return variants, all_variants



@callback(
    Output('variant-store','data'),
    Output('variant-container', 'children', allow_duplicate=True),
    Input({'type': 'variant-input', 'index': dependencies.ALL}, 'value'),
    Input({'type': 'variant-button-delete', 'index': dependencies.ALL}, 'n_clicks'),
    Input('remove-variable-button', 'n_clicks'),
    State('variable-container', 'value'),
    State('variant-store','data'),
    State('variant-container', 'children'),
    prevent_initial_call=True
)
def update_variant_dict(values, remove_variant, remove_variable, selected_tab, all_variants,variant_container):
    ctx_id = ctx.triggered_id
    if not ctx_id:
        return all_variants, variant_container
    
    # Remove a variable from the dict in store.
    if ctx_id == 'remove-variable-button':
        all_variants.pop(selected_tab, None)
    # Remove variant from store dict.
    elif ctx_id['type'] == 'variant-button-delete':
        idx_variable, idx_variant = ctx_id['index'].split('-')
        all_variants[idx_variable].pop(idx_variant, None)
        for child in variant_container:
            if idx_variant == str(child['props']['children'][0]['props']['id']['index']):
                variant_container.remove(child)
                break
    # Change the value of a variant (also in the store).
    elif ctx_id['type'] == 'variant-input':
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        #input_index = eval(input_id)['index']
        changed_value = ctx.triggered[0]['value']

        all_variants[selected_tab][str(ctx_id['index'])] = changed_value

    return all_variants, variant_container


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
    Output("samples-table", "selectedRows", allow_duplicate=True),
    Output("prompt-sample", "children", allow_duplicate=True),
    Input("button-left", "n_clicks"),
    Input("button-right", "n_clicks"),
    Input("samples-table", "selectedRows"),
    Input("samples-table", "rowData"),
    Input("dataset-selection", "value"),
    prevent_initial_call=True

)
def change_sample(left_clicks, right_clicks, selected_rows, row_data, dataset_name):
    if selected_rows:
        row_index = selected_rows[0]
        for i, entry in enumerate(row_data):
            if entry == selected_rows[0]:
                row_index = i
    else:
        return selected_rows, []

    ctx_id = ctx.triggered_id
    if ctx_id == 'button-left':
        row_index -= 1
    elif ctx_id == 'button-right':
        row_index += 1

    row_index %= len(row_data)

    new_selected = [row_data[row_index]]

    output = []
    if 'content' in new_selected[0].keys():
        output.append(html.P(new_selected[0]['content']))
    elif 'Description' in new_selected[0].keys():
        output.append(html.P(new_selected[0]['Description']))
    
    return new_selected, output
