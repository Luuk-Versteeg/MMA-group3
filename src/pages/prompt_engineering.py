from dash import html, dcc, Output, Input, State, ctx, callback, dependencies
from collections import defaultdict
import itertools
from .tinyllama import sent_classifier, news_classifier, snellius
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import add_synonyms, get_synonyms
import random

import os
import numpy as np

progress_file = os.path.abspath('src/pages/progress.txt')


prompt_engineering = html.Div(children=[
    html.H1(children='Prompt Engineering', style={'textAlign':'center'}),
    html.P(children="""
                    In this section you will create your prompts. 
                    The first window allows you to select one sample from the selected data, 
                    this sample will be used to test your prompts. Below you can define your prompt template.
                    """),
    html.P(children="""
                    After clicking 'Generate prompts', all possible combinations of variants will be made.
                    You can test all the prompts on the selected datasample by clicking 'Test prompts'.
                    """),
    html.Div(children=[
        html.Div(
            className='selected-sample',
            children=[
                html.Button('<', id='button-left', style={"padding": "30px"}),
                html.Div(id="prompt-sample", 
                         children=" Improved Pitching Has the Keys Finishing on an Upswing As the season winds down for the Frederick Keys, Manager Tom Lawless is starting to enjoy the progress his pitching staff has made this season.",
                         style={"padding": "15px 30px", "border": "1px solid black", "width": "100%", 
                                "minHeight": "100px", "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Button('>', id='button-right', style={"padding": "30px"}),
                dcc.Store(id="prompt-sample-text")
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
        html.P("Create a prompt template:", style={"marginBottom": "5px", 'display':'inline-block'}),
        html.Abbr("❔", title=  "The variant placeholders (e.g., {var1} and {var2}) will be replaced by the variants you defined in the field below. The {text} placeholder will be replaced by the sentence selected from the dataset."),
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
                    html.Div([
                        html.Button('Remove selected variable', id='remove-variable-button'),
                        html.Abbr("❔", title="Add or remove as many {var} variables as you need."),
                    ])
                ], style={'display': 'flex', 'gap': '15px', 'padding': '10px 20px', 'justifyContent': 'center'}),
                dcc.Tabs(
                    content_style={"width": "100%", 'flex-direction':'column'},
                    parent_style={"width": "100%",'flex-direction':'column'},
                    style={'width':'100%', 'flex-direction':'column'},
                    id="variable-container", children=[], vertical=True),
            ],
            style={'width':'47%', 'overflowY': 'scroll'},
        ),
        html.Div(
            className='box',
            children=[
                html.Div(
                    id='tabs-content',
                    children=[
                        html.Button('Add variant', id='add-variant-button', n_clicks=0),
                        html.Abbr("❔", title="For every variable in your prompt you can write as many variants as you want. When generating the prompts, all possible combinations of variable variants are be made."),
                        ],
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
            dcc.Store('results-dict'),
            html.Div([
                html.Button('Generate prompts', id='button-generate-prompts'),
                html.Abbr("❔", title="Generate Prompts: Create all possible prompts with defined variants."),
            ]),
            html.Div([
                html.Button('Test prompts', id='button-test-prompts'),
                html.Abbr("❔", title="Add or remove as many {var} variables as you need."),
            ])
        ], 
        style={'width':'100%', 'margin': '20px 0px', 'display': 'flex', 'gap': '15px', 'alignItems': 'center', 'justifyContent': 'center'}
    ),
    html.Div(children=[
        dcc.Interval(id='interval-component', interval=500),
        html.Progress(id='prompt-run-progress', max="100", value="0", style={'width':'90%'}), 
        ], 
        style={'width':'100%', 'margin': '20px 0px', 'display': 'flex', 'gap': '15px', 'alignItems': 'center', 'justifyContent': 'center'}
    ),
    html.Div(id='generated-prompts-container', children=[
    ], style={'width':'100%', 'maxHeight': '300px', 'overflowY': 'scroll', 'display': 'flex', 'gap': '15px', 'marginTop': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

    html.Div(id='tested-prompts-container', children=[
        #html.Div(children='test prompt')
    ], style={'width':'100%', 'maxHeight': '300px', 'overflowY': 'scroll', 'display': 'flex', 'gap': '15px', 'marginTop': '20px', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    
    html.P("Attention visualizer:", style={"marginBottom": "5px", 'display':'inline-block'}),
    html.Abbr("❔", title=  "Click part of the models output text to view the corresponding attention scores of the input. Green and red highlights indicate high and low attention scores respectively."),
    
    html.Div(style={'width': '100%', 'border': '1px solid #ccc',}, children=[
        html.Div(children = [
            html.Div("PROMPT", style={'text-align':'center'}),
            html.Div("No prompt tested...", id='full-prompt', style={'width': '100%', 'padding': '10px', 'overflow': 'hidden', 'whiteSpace': 'normal', 'wordWrap': 'break-word'}),
        ]),
        html.Div(style={
            'width': '100%',
            'height': '1px',
            'background-color': '#000',  # Black color for the line
            'margin': '10px 0'  # Adjust margin as needed
        }),
         html.Div(children = [
            html.Div("ANSWER", style={'text-align':'center'}),
            html.Div("No prompt tested...", id='full-answer', style={'width': '100%', 'padding': '10px', 'overflow': 'hidden', 'whiteSpace': 'normal', 'wordWrap': 'break-word'})
        ]),
    ])
])


@callback(
    Output('generated-prompts-container', 'children', allow_duplicate=True),
    Output('prompt-list', 'data'),
    Input('button-generate-prompts', 'n_clicks'),
    State('variant-store', 'data'),
    State('textarea-prompt', 'value'),
    prevent_initial_call=True,
    running=[(Output("button-generate-prompts", "disabled"), True, False)]
)
def generate_prompts(generate_clicks, data, prompt):
    if not generate_clicks:
        return [], []

    # Extract variables and create permutations
    vars = []
    for var_num, variants in data.items():
        vars.append([(var_num, var) for var in list(variants.values())])
    permutations = list(itertools.product(*vars))

    generated_prompts = []
    prompt_list = []

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
                synonyms = get_synonyms(word)
                if synonyms:
                    random_synonyms = random.sample(synonyms, min(5, len(synonyms)))
                    synonym_tooltip = ', '.join(random_synonyms)
                    word_element = html.Span(
                        word,
                        className='synonym-token',  # CSS class for styling
                        style={'border-bottom': '1px dashed red', 'cursor': 'help'},
                        title=f'Synonyms: {synonym_tooltip}'
                    )
                else:
                    word_element = word

                prompt_lines.append(word_element)
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
    Output('prompt-run-progress', 'value'),
    Input('interval-component', 'n_intervals')
)

def update_progressbar(n_intervals):
    with open(progress_file,  'r') as f:
        progress = f.readline()
    return str(progress)

def update_progressfile(progress):
    with open(progress_file,  'w') as f:
        f.write(str(progress))

def clear_progressfile():
    with open(progress_file, 'w') as f:
        f.write('0')

clear_progressfile()

@callback(
    # Output('tested-prompts-container', 'children'),
    Output('generated-prompts-container', 'children', allow_duplicate=True),
    Output('results-dict','data'),
    Input('button-test-prompts', 'n_clicks'),
    Input("dataset-selection", "value"),
    State('true-label', 'data'),
    State('prompt-list', 'data'),
    State('prompt-sample-text', 'data'),
    prevent_initial_call = True,
    running=[(Output("button-test-prompts", "disabled"), True, False)],
)
def test_prompts(test_button, dataset_name, true_label, generated_prompts, text):
    if not test_button:
        return [], {}

    if dataset_name == "AG News":
        classifier = news_classifier
    if dataset_name == "Amazon Polarity" or dataset_name == "GLUE/sst2":
        classifier = sent_classifier

    pred_labels = []
    pred_words = []
    pred_attentions = []
    full_prompts = []
    n_total = len(generated_prompts)
    clear_progressfile()

    if snellius:
        with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on your hardware
            full_prompt = prompt.format(text=text)

            future_to_prompt = {executor.submit(classifier, full_prompt): prompt for prompt in generated_prompts}
            for i, future in tqdm(as_completed(enumerate(future_to_prompt)), total=n_total):
                label, words, att_data = future.result()
                pred_labels.append(label)
                pred_words.append(words)
                pred_attentions.append(att_data)
                full_prompts.append(full_prompt)
                update_progressfile(((i+1)/n_total)*100)
    else:
        for i, prompt in tqdm(enumerate(generated_prompts)):
            prompt = prompt.format(text=text)
            pred_label, words, att_data = classifier(prompt)
            pred_labels.append(pred_label)
            pred_words.append(words)
            pred_attentions.append(att_data)
            full_prompts.append(prompt)

            update_progressfile(((i+1)/n_total)*100)

    results_dict = {}

    # Convert to list of lines with html.Br() instead of \n.
    colored_prompt_divs = []
    for idx, (pred_label, pred_word, pred_attention, new_prompt, full_prompt) in enumerate(zip(pred_labels, pred_words, pred_attentions, generated_prompts, full_prompts)):
        results_dict[idx] = [full_prompt, pred_label, pred_word, pred_attention]
        
        if pred_label == true_label:
            color='LightGreen' 
        else:
            color='LightCoral'
        prompt_lines = []
        for line in new_prompt.split('\n'):
            prompt_lines.append(line)
            prompt_lines.append(html.Br())
        prompt_lines.append(html.Hr())
        prompt_lines.append(f"Predicted: {pred_label}")

        colored_prompt_divs.append(html.Div(
            id={'type': 'generated-prompt', 'index': int(idx)}, 
            children=prompt_lines + [html.Button('Attention', id={'type': 'prompt-attention-button', 'index': int(idx)})],
            style={'border':'1px solid #000', 'height':200, 'width':200, 'padding': 15, 'boxSizing': 'border-box', 'display':'inline-block', 'background-color':color}))

    return colored_prompt_divs, results_dict


@callback(
    Output('full-prompt', 'children'),
    Output('full-answer', 'children'),
    Input({'type': 'prompt-attention-button', 'index': dependencies.ALL}, 'n_clicks'),
    Input({'type': 'token', 'index': dependencies.ALL}, 'n_clicks'),
    State('results-dict','data'),
    State('full-prompt', 'children'),
    State('full-answer', 'children')
)
def show_answer(prompt_clicks, token_clicks, results_dict, prompt, answer):
    ctx_id = ctx.triggered_id

    if ctx_id == None:
        return prompt, answer

    # Pressed on output token.
    if ctx_id['type'] == 'token':
        prompt_idx, clicked_pos, clicked_token = ctx_id['index'].split("---")
        clicked_pos = int(clicked_pos)

        scores_lists = results_dict[str(prompt_idx)][3]
        answer_tokens_with_spaces = results_dict[str(prompt_idx)][2]
        answer_tokens = list(filter(lambda w: w != ' ', answer_tokens_with_spaces))

        # Find relevant parts of output.    
        sublist = ['</s>', '', '\n', '<', '|', 'user', '|', '>']
        start = find_sublist_index(answer_tokens, sublist) + len(sublist)
        sublist = ['</s>', '', '\n', '<', '|', 'ass', 'istant', '|', '>']
        end = find_sublist_index(answer_tokens, sublist)

        attention_scores = scores_lists[clicked_pos][1]
        attention_scores = np.array(attention_scores)
        prompt_scores = attention_scores[start:end]
        normalized_scores = (attention_scores - prompt_scores.min()) / (prompt_scores.max() - prompt_scores.min())
        mean_score = np.mean(normalized_scores[start:end])
        max_score = np.max(normalized_scores[start:end])

        colored_text = []
        extended_scores = []

        sublist = ['</s>', ' ', '', '\n', '<', '|', 'user', '|', '>']
        start = find_sublist_index(answer_tokens_with_spaces, sublist) + len(sublist)
        sublist = ['</s>', ' ', '', '\n', '<', '|', 'ass', 'istant', '|', '>']
        end = find_sublist_index(answer_tokens_with_spaces, sublist)

        j = 0
        for i, tok in enumerate(answer_tokens_with_spaces):
            if tok == ' ':
                extended_scores.append(None)
            else:
                extended_scores.append(normalized_scores[j])
                j += 1

        j = 0
        for i, (token, score) in enumerate(zip(answer_tokens_with_spaces, extended_scores)):
            if token == ' ':
                colored_text.append(html.Span(token))
                continue
            elif token == '\n':
                colored_text.append(html.Br())
                continue


            if score >= mean_score:
                # lightGreen: rgb(144, 238, 144)
                t = (score - mean_score) / (max_score - mean_score)
                color = f"rgb({lerp(255, 144, t)}, {lerp(255, 238, t)}, {lerp(255, 144, t)})"
            else:
                # lightCoral: rgb(240, 128, 128)
                t = score / mean_score
                color = f"rgb({lerp(240, 255, t)}, {lerp(128, 255, t)}, {lerp(128, 255, t)})"

            
            if score == 0 or i < start or i > end:
                color = "rgb(255, 255, 255)"

            style={"background-color": color}
            id={'type':'token', 'index':f"{prompt_idx}---{j}---{token}"}
            j += 1

            if i > end:
                style['cursor'] = 'pointer'
            else:
                id = ''

            if i == clicked_pos:
                style['textDecoration'] =  'underline'
                style['fontWeight'] = 'bold'
                
            colored_text.append(
                html.Span(token,
                          id=id,
                          style=style)
            )


        prompt = colored_text[start:end]
        answer = colored_text[end + len(sublist):-2]

        return prompt, answer
    # Pressed attention button.
    else:
        full_prompt = results_dict[str(ctx_id['index'])][0]
        answer_tokens_with_spaces = results_dict[str(ctx_id['index'])][2]
        answer_tokens = list(filter(lambda w: w != ' ', answer_tokens_with_spaces))

        sublist = ['</s>', ' ', '', '\n', '<', '|', 'user', '|', '>']
        print("answer_tokens", answer_tokens)
        start = find_sublist_index(answer_tokens_with_spaces, sublist) + len(sublist)
        sublist = [ '</s>', ' ', '', '\n', '<', '|', 'ass', 'istant', '|', '>']        
        end = find_sublist_index(answer_tokens_with_spaces, sublist)

        #text = []

        text = []
        j = 0
        for i, token in enumerate(answer_tokens_with_spaces):
            if token == ' ':
                text.append(html.Span(' '))
                continue
            else: 
                j += 1

            if token == '\n':
                text.append(html.Br())
                continue
            
            id={'type': 'token', 'index':f"{ctx_id['index']}---{j}---{token}"}
            style={'cursor': 'pointer'}
            text.append(html.Span(token, id=id, style=style))

        prompt = text[start:end]
        answer = text[end + len(sublist):-2]

        return prompt, answer


def find_sublist_index(biglist, sublist):
    end = len(biglist) - len(sublist) + 1
    for idx in range(0, end):
        if sublist == biglist[idx:idx+len(sublist)]:
            return idx


def lerp(a, b, t):
    return t * a + (1 - t) * b


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
                          style={'line-width': '100%', 'flex-grow': 1},
                          selected_style={'line-width': '100%'})
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
    Output("prompt-sample", "children"),
    Output("prompt-sample-text", "data"),
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
        return selected_rows, [], []

    ctx_id = ctx.triggered_id
    if ctx_id == 'button-left':
        row_index -= 1
    elif ctx_id == 'button-right':
        row_index += 1

    row_index %= len(row_data)

    new_selected = [row_data[row_index]]

    output = []
    text = ""

    if 'content' in new_selected[0].keys():
        text = new_selected[0]['content']

    elif 'Description' in new_selected[0].keys():
        text = new_selected[0]['Description']

    elif 'text' in new_selected[0].keys():
        text = text = new_selected[0]['text']

    syn_out = add_synonyms(text)
    output.append(html.P(children=syn_out))
    
    return new_selected, output, text
