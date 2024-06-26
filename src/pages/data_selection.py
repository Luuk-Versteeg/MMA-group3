from dash import html, dcc, Output, Input, callback, State, callback_context 
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_ag_grid
import pandas as pd

import plotly
import random
from collections import Counter

from .utils import add_synonyms, filter_text, preprocess_text

from widgets import histogram
from dataloaders.load_data import datasets
from wordcloud import WordCloud


data_selection = html.Div(children=[
    html.H1(children='Data Selection', style={'textAlign': 'center'}),
    html.P(children='This section allows you to select a dataset. For each available dataset, you can select a subset (e.g., train, validation, test). The selected dataset and the chosen number of samples will be used throughout the other sections. This data will be used to engineer and evaluate your prompt(s).'),
    html.P(children='For every selected subset and number of samples, statistics such as word frequency and label distribution will be shown.'),
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
                ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "0px"}),
                html.Button("Preprocess", id="preprocess"),
                html.P(id="dataset-description"),
                html.P(children=f'Scheme:', id="dataset-scheme")            ]),
            html.Div(id="selected-sample", style={"padding": "15px 30px", "margin": "0px 20px", "marginTop": "30px", "marginBottom": "20px"}, className='box')
        ], style={"width": "48%"}),
        html.Div(dcc.Tabs(children=[
            dcc.Tab(label="Labels", children=histogram.create_histogram(id="label-histogram")),
            dcc.Tab(label="Words frequency", children=histogram.create_histogram(id="wordcloud"), className="wordcloud")
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
    ), style={"marginTop": "10px", "marginBottom": "30px"})
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
    Input("resample", "n_clicks"),
    Input("preprocess", "n_clicks"),
    State("selected-dataset", "data")
)
def update_selected_dataset(dataset_name, dataset_split, n_samples, resample_clicks, preprocess_clicks, current_data):
    ctx = callback_context
    if not dataset_name or not dataset_split:
        return
    
    dataset = select_dataset(dataset_name)
    if dataset_split not in dataset["data"].keys():
        return

    # Check which button was clicked
    if ctx.triggered and 'preprocess' in ctx.triggered[0]['prop_id']:
        if current_data:
            samples = pd.DataFrame.from_dict(current_data)
            samples['text'] = samples['text'].apply(preprocess_text)
        else:
            return
    else:
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
    Input("samples-table", "selectedRows"),
)
def update_sample(selected_rows):
    if len(selected_rows) == 0:
        return html.P("No sample selected...")

    output = []

    for key, value in selected_rows[0].items():
        syn_out = add_synonyms(value)
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
    # Handle the case when no dataset is selected
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
            margin=dict(b=0, l=0, r=0, t=0)
        )
        return fig

    # Convert input data to DataFrame
    dataframe = pd.DataFrame.from_dict(dataframe_data)
    words_per_label = {}

    # Process each row to collect words by label
    for _, row in dataframe.iterrows():
        label = row['label']
        description = row['text']
        filtered_description = filter_text(description)

        if label not in words_per_label:
            words_per_label[label] = []
        words_per_label[label].extend(filtered_description)

    # Find the most common words for each label
    most_common_words = {}
    for label, words in words_per_label.items():
        word_counts = Counter(words)
        most_common_words[label] = word_counts.most_common(10)

    # Prepare data for word cloud generation
    all_words = []
    all_counts = []
    all_labels = []
    label_colors = {label: plotly.colors.DEFAULT_PLOTLY_COLORS[i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)] for i, label in enumerate(most_common_words.keys())}

    for label, words in most_common_words.items():
        for word, count in words:
            all_words.append(word)
            all_counts.append(count)
            all_labels.append(label)

    # Create a dictionary with word frequencies and a dictionary for colors
    word_frequencies = {word: count for word, count in zip(all_words, all_counts)}
    word_colors = {word: label_colors[label] for word, label in zip(all_words, all_labels)}

    # Custom color function for the word cloud
    def color_func(word, *args, **kwargs):
        return word_colors.get(word, "black")

    # Generate the word cloud image with custom coloring
    wc = WordCloud(width=500, height=600, background_color='white', color_func=color_func).generate_from_frequencies(word_frequencies)

    # Convert the word cloud image to a Plotly figure
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=wc.to_image(),
            xref="x",
            yref="y",
            x=0,
            y=3.5,
            sizex=5,
            sizey=5,
            xanchor="left",
            yanchor="top",
            layer="below"
        ),
    )

    # Add invisible scatter plots for legend entries
    for label, color in label_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label
        ))

    # Update layout to hide axes and show legend
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        margin=dict(b=0, l=0, r=0, t=0),
        showlegend=True,
        legend=dict(title="Labels", itemsizing='constant'),
        hovermode=False,
        clickmode=None,
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
            margin=dict(b=0, l=0, r=0, t=0)  # Adjust margins to ensure the text is visible
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
    # Output("prompt-sample", "children"),
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
            # html.P("No sample selected..."), 
            "Label of selected sample: no sample selected...", 
            "",
            "Possible labels: no dataset selected..."
        )

    dataset = select_dataset(dataset_name)
    return (
        # selected_rows[0]['text'], 
        [html.Span("Label of selected sample: "), html.Span(selected_rows[0]['label'], style={"fontWeight": "bold"})],
        selected_rows[0]['label'],
        "Possible labels: " + dataset['scheme'],
    )
