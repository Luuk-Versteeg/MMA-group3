from dash import Dash, html, dcc, Output, Input, State, ctx, callback, dependencies
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from collections import defaultdict
import itertools
import plotly.graph_objects as go
import dash_ag_grid


import plotly
import random
import nltk
from nltk.corpus import stopwords
from collections import Counter


from widgets import histogram
from dataloaders.load_data import datasets

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


data_selection = html.Div(children=[
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


@callback(
    Output("prompt-sample", "children"),
    Input("samples-table", "selectedRows")
)
def update_prompt_sample(selected_rows):
    return 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'
