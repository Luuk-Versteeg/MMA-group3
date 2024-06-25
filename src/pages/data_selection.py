from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_ag_grid
import pandas as pd

import plotly
import random
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from collections import Counter

from widgets import histogram
from dataloaders.load_data import datasets

nltk.download('stopwords', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))

data_selection = html.Div(children=[
    html.H1(children='Data selection', style={'textAlign': 'center'}),
    html.Div(children=[
        html.Div(children=[
            dcc.Store(id='selected-dataset'),
            dcc.Dropdown([d["name"] for d in datasets], placeholder="Select dataset", id='dataset-selection', style={"width": "100%"}),
            html.Div(children=[
                dcc.Dropdown([], placeholder="Select subset", id="dataset-split", style={"marginTop": "10px", "width": "200px"}),
                html.P(children=[
                    html.P(children="Select the number of samples:"),
                    dbc.Input(type="number", min=0, value=10, max=100, step=1, id="n-samples"),
                    html.P(children="(max: )", id="max-samples"),
                    html.Button("Resample", id="resample")
                ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
                html.P(id="dataset-description"),
                html.P(children=f'Scheme:', id="dataset-scheme")            ]),
            html.Div(id="selected-sample", style={"padding": "15px 30px", "border": "1px solid black", "margin": "0px 20px", "marginTop": "30px"})
        ], style={"width": "48%"}),
        html.Div(dcc.Tabs(children=[
            dcc.Tab(label="Labels", children=histogram.create_histogram(id="label-histogram")),
            dcc.Tab(label="Words frequency", children=histogram.create_histogram(id="wordcloud"))
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
        id='samples-table'
    ), style={"marginTop": "30px", "marginBottom": "30px"})
])


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
    description = [html.Span("Description: ", style={"fontWeight": "bold"})]
    scheme = [html.Span("Labels: ", style={"fontWeight": "bold"})]

    if dataset_name:
        dataset = select_dataset(dataset_name)
        split += list(dataset["data"].keys())
        description += dataset["description"]
        scheme += dataset["scheme"]

    return split, description, scheme


@callback(
    Output("selected-dataset", "data"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value"),
    Input("n-samples", "value"),
    Input("resample", "n_clicks")
)


def update_selected_dataset(dataset_name, dataset_split, n_samples, _):
    if not dataset_name or not dataset_split:
        return

    dataset = select_dataset(dataset_name)
    if dataset_split not in dataset["data"].keys():
        return

    samples = dataset["data"][dataset_split].sample(n_samples)
    return samples.to_dict()


@callback(
    Output("samples-table", "columnDefs"),
    Output("samples-table", "rowData"),
    Output("samples-table", "selectedRows"),
    Input("selected-dataset", "data"),
)
def update_grid(dataframe_data):
    if not dataframe_data:
        return [], [], []

    dataframe = pd.DataFrame.from_dict(dataframe_data)
    cols = [{"field": col} for col in dataframe.columns.to_list()]
    rows = dataframe.to_dict("records")
    selected = [rows[0]]

    return cols, rows, selected


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)


@callback(
    Output("selected-sample", "children"),
    Input("samples-table", "selectedRows"),
)
def update_sample(selected_rows):
    if len(selected_rows) == 0:
        return html.P("No sample selected...")

    output = []

    for key, value in selected_rows[0].items():
        tokens = word_tokenize(value)
        syn_out = []

        for token in tokens:

            try:
                synonyms = get_synonyms(token)
            except:
                synonyms = []

            if synonyms:
                # Pick 5 random synonyms from the list, or fewer if less than 5 are available
                random_synonyms = random.sample(synonyms, min(5, len(synonyms)))
                # Create a span element with tooltip and apply styling
                token_element = html.Span(
                    token,
                    className='synonym-token',  # CSS class for styling
                    style={'border-bottom': '1px dashed red', 'cursor': 'help'},
                    title=f'Synonyms: {", ".join(random_synonyms)}'  # Tooltip content
                )
                syn_out.append(token_element)
                syn_out.append(" ")  # Add space between tokens for readability
            else:
                syn_out.append(token + " ")  # No tooltip if no synonyms found

        output.append(html.P([html.Span(f"{key}: "),*syn_out]))

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
    Input("selected-dataset", "data"),
    prevent_initial_call=True
)
def update_wordcloud_histogram(dataframe_data):
    if not dataframe_data:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="No dataset selected...",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=28, color="gray")
                )
            ],
            margin=dict(b=0, l=0, r=0, t=100)  # Adjust margins to ensure the text is visible
        )
        return fig

    dataframe = pd.DataFrame.from_dict(dataframe_data)
    words_per_label = {}
    allowed_pos = ['NN', 'NNP', 'NNS', 'VB']
    for _, row in dataframe.iterrows():
        label = row['label']
        description = row['text']
        tokens = word_tokenize(description)
        tokens_pos = nltk.pos_tag(tokens)
        filtered_tokens = [x[0] for x in tokens_pos if x[1] in allowed_pos and x[0] not in stop_words]
        if label not in words_per_label:
            words_per_label[label] = []
        words_per_label[label].extend(filtered_tokens)

    most_common_words = {}
    for label, words in words_per_label.items():
        word_counts = Counter(words)
        most_common_words[label] = word_counts.most_common(10)

    all_words = []
    all_counts = []
    all_labels = []
    label_colors = {label: plotly.colors.DEFAULT_PLOTLY_COLORS[i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)] for i, label in enumerate(most_common_words.keys())}

    for label, words in most_common_words.items():
        for word, count in words:
            all_words.append(word)
            all_counts.append(count)
            all_labels.append(label)

    # Normalize counts to a range suitable for font sizes
    min_size, max_size = 10, 35
    min_count, max_count = min(all_counts), max(all_counts)
    def normalize(size, min_count, max_count, min_size, max_size):
        if max_count == min_count:  # handle the case when all counts are the same
            return (min_size + max_size) / 2
        return min_size + (size - min_count) * (max_size - min_size) / (max_count - min_count)

    sizes = [normalize(count, min_count, max_count, min_size, max_size) for count in all_counts]

    def grid_positions(n, deviation=0.15):
        positions = []
        grid_size = int(n**0.5) + 1
        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            x = col + random.uniform(-deviation, deviation)
            y = row + random.uniform(-deviation, deviation)
            positions.append((x, y))
        return positions

    positions = grid_positions(len(all_words))
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    colors = [label_colors[label] for label in all_labels]

    # Create the word cloud using Plotly
    data = go.Scatter(
        x=x,
        y=y,
        mode='text',
        text=all_words,
        marker={'opacity': 0.3},
        textfont={'size': sizes, 'color': colors},
        showlegend=False  # Hide the main scatter plot from the legend
    )

    fig = go.Figure(data=[data])

    # Add invisible scatter plots for legend entries
    for label, color in label_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label
        ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=0),
        showlegend=True,
        legend=dict(title="Labels", itemsizing='constant')
    )

    return fig


@callback(
    Output("label-histogram", "figure"),
    Input("selected-dataset", "data"),
    State("dataset-selection", "value"),
    State("dataset-split", "value"),
    prevent_initial_call=True,
)
def update_label_histogram(dataframe_data, dataset_name, dataset_split):
    if not dataframe_data:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="No dataset selected...",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=28, color="gray")
                )
            ],
            margin=dict(b=0, l=0, r=0, t=100)  # Adjust margins to ensure the text is visible
        )
        return fig

    dataframe = pd.DataFrame.from_dict(dataframe_data)
    # Create the histogram for the 'class' column
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dataframe.label))
    # Update layout
    fig.update_layout(
        title=f"Distribution of selected samples from {dataset_name} ({dataset_split})",
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


@callback(
    Output("prompt-sample", "children"),
    Output("prompt-label", "children"),
    Output("true-label", "data"),
    Output("possible-answers", "children"),
    Input("samples-table", "selectedRows"),
    Input("dataset-selection", "value"),
    Input("n-samples", "value")
)
def update_prompt_sample(selected_rows, dataset_name, n_samples):
    if len(selected_rows) == 0:
        return (
            html.P("No sample selected..."), 
            "Label of selected sample: no sample selected...", 
            "",
            "Possible labels: no dataset selected..."
        )

    dataset = select_dataset(dataset_name)
    return (
        selected_rows[0]['text'], 
        [html.Span("Label of selected sample: "), html.Span(selected_rows[0]['label'], style={"fontWeight": "bold"})],
        selected_rows[0]['label'],
        "Possible labels: " + dataset['scheme'],
    )
