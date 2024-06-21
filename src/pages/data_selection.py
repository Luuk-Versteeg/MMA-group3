from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_ag_grid
import pandas as pd

import plotly
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from collections import Counter


from widgets import histogram
from dataloaders.load_data import datasets

<<<<<<< Updated upstream
nltk.download('stopwords', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
=======
# nltk.download('stopwords', quiet=True)
# nltk.download('brown', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
>>>>>>> Stashed changes

stop_words = set(stopwords.words('english'))


data_selection = html.Div(children=[
    html.H1(children='Data selection', style={'textAlign':'center'}),
    html.Div(children=[
        html.Div(children=[
            dcc.Store(id='selected-dataset'),
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
        # defaultColDef={"filter": "agTextColumnFilter"},
        # className='stretchy-widget ag-theme-alpine',
        # style={'width': '', 'height': ''},
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
    description = "Description: "
    scheme = "Scheme: "

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
)
def update_selected_dataset(dataset_name, dataset_split, n_samples):

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
    Input("selected-dataset", "data"),
    State("dataset-selection", "value"),
    State("dataset-split", "value"),
)
def update_wordcloud_histogram(dataframe_data, dataset_name, dataset_split):

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
            margin=dict(b=0, l=0, r=0, t=100)
        )

        return fig
    
    dataframe = pd.DataFrame.from_dict(dataframe_data)
    
    allowed_pos = ['NN', 'NNP', 'NNS', 'VB']
    
    label_words = {}
    for label in dataframe['label'].unique():
        label_data = dataframe[dataframe['label'] == label]
        words = []
        for description in label_data["text"]:
            tokens = word_tokenize(description)
            tokens_pos = nltk.pos_tag(tokens)
            filtered_tokens = [x[0] for x in tokens_pos if x[1] in allowed_pos and x[0] not in stop_words]
            words.extend(filtered_tokens)
        word_counts = Counter(words)
        label_words[label] = word_counts.most_common(5)
    
    fig = go.Figure()

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(label_words.keys())}
    
    for label, words_counts in label_words.items():
        words, counts = zip(*words_counts)
        min_size, max_size = 15, 35
        min_count, max_count = min(counts), max(counts)
        
        def normalize(size, min_count, max_count, min_size, max_size):
            if max_count == min_count:  # handle the case when all counts are the same
                return (min_size + max_size) / 2
            return min_size + (size - min_count) * (max_size - min_size) / (max_count - min_count)

        sizes = [normalize(count, min_count, max_count, min_size, max_size) for count in counts]

        x = [random.random() for _ in range(len(words))]
        y = [random.random() for _ in range(len(words))]
        text_colors = [color_map[label] for _ in range(len(words))]
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='text',
                text=words,
                textfont={'size': sizes, 'color': text_colors},
                name=label
            )
        )
    
    legend_items = [html.Span([html.Span(style={'backgroundColor': color_map[label], 'display': 'inline-block', 'width': '10px', 'height': '10px', 'marginRight': '5px'}), label]) for label in color_map]

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=40),
        title=dict(
            text="Word Cloud",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        annotations=[
            dict(
                x=1,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                text=html.Div(legend_items, style={'display': 'flex', 'flexDirection': 'column', 'gap': '5px'}).to_plotly_json()
            )
        ]
    )

    return fig


@callback(
    Output("label-histogram", "figure"),
    Input("selected-dataset", "data"),
    State("dataset-selection", "value"),
    State("dataset-split", "value"),
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
    Output("prompt-labels", "children"),
    Output("select-num-samples", "value"),
    Output("possible-answers", "value"),
    Input("samples-table", "selectedRows"),
    Input("dataset-selection", "value"),
    Input("n-samples", "value")
)
def update_prompt_sample(selected_rows, dataset_name, n_samples):
    if len(selected_rows) == 0:
        return html.P("No sample selected..."), "labels: ", [0], ''

    dataset = select_dataset(dataset_name)

    return selected_rows[0]['text'], selected_rows[0]['label'], n_samples, dataset['scheme']
