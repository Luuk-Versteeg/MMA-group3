from dash import Dash, html, dcc, Output, Input, State, ctx, callback, dependencies
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict
import itertools
import plotly.graph_objects as go
import dash_ag_grid

from pages.data_selection import data_selection
from pages.evaluation import evaluation
from pages.prompt_engineering import prompt_engineering

import plotly
import random
import nltk
from nltk.corpus import stopwords
from collections import Counter


from widgets import histogram
from dataloaders.load_data import datasets


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app.layout = dbc.Container([
    data_selection,
    evaluation,
    prompt_engineering
], id="carousel")


def select_dataset(name, datasets=datasets):
    """Selects the dataset dictionary from the list of datasets if the name exists."""
    dataset = [d for d in datasets if d["name"] == name]

    if len(dataset) == 0:
        raise KeyError(f"There's no dataset called: {name}")
    
    return dataset[0]


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


@callback(
    Output("prompt-sample", "children"),
    Input("samples-table", "selectedRows")
)
def update_prompt_sample(selected_rows):
    return 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'


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
    Output("wordcloud", "figure"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
    Input("n-samples", "value")
)
def update_wordcloud_histogram(dataset_name, dataset_split, n_samples):
    # Load your dataset
    if dataset_name == "AG News":
        dn = "agnews"
    elif dataset_name == "Amazon Polarity":
        dn = "amazon_polarity"
    path = f"src/dataloaders/{dn}/data/{dataset_split}.csv"
    df = pd.read_csv(path)

    # Filter the DataFrame based on the selected dataset name and split
    filtered_df = df['Description'].head(n_samples)

    # Preprocess descriptions: Remove stop words
    words = []
    for description in filtered_df:
        tokens = description.split()
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        words.extend(filtered_tokens)

    # Count word frequencies
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(25)
    words, counts = zip(*most_common_words)

    # Generate random positions, colors, and sizes for the word cloud
    x = [random.random() for _ in range(30)]
    y = [random.random() for _ in range(30)]
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for _ in range(30)]
    sizes = [random.randint(15, 35) for _ in range(30)]

    # Create the word cloud using Plotly
    data = go.Scatter(
        x=x,
        y=y,
        mode='text',
        text=words,
        marker={'opacity': 0.3},
        textfont={'size': sizes, 'color': colors}
    )

    # Define layout for the word cloud
    layout = go.Layout(
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        annotations=[
            dict(
                text=f"{dataset_name} - {dataset_split}",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="gray"),
                x=0,
                y=-0.15,  # Position it below the plot
                xanchor='left',
                yanchor='top'
            )
        ],
        margin=dict(b=30, l=30, r=30, t=30)  # Adjust margins for better spacing
    )

    fig = go.Figure(data=[data], layout=layout)

    return fig

@callback(
    Output("label-histogram", "figure"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
    Input("n-samples", "value")
)
def update_label_histogram(dataset_name, dataset_split, n_samples):
    # Load your dataset
    # Replace 'your_dataset.csv' with the actual path to your CSV file
    if dataset_name == "AG News":
        dn = "agnews"
    elif dataset_name == "Amazon Polarity":
        dn = "amazon_polarity"
    path = f"src/dataloaders/{dn}/data/{dataset_split}.csv"
    df = pd.read_csv(path)

    # Filter the DataFrame based on the selected dataset name and split
    filtered_df = df['Class Index'].head(n_samples)


    # Create the histogram for the 'class' column
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=filtered_df.values))

    # Update layout
    fig.update_layout(
        title=f"Distribution of {dataset_name} ({dataset_split})",
        xaxis_title="Class",
        yaxis_title="Count",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(b=30, l=30, r=30, t=30),  # Adjust margins for better spacing
        annotations=[
            dict(
                text=f"{dataset_name} - {dataset_split}",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="gray"),
                x=0,
                y=-0.15,  # Position it below the plot
                xanchor='left',
                yanchor='top'
            )
        ],
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)
