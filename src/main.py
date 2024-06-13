from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import dash_ag_grid

from widgets import histogram, table
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
        dcc.Dropdown(df.country.unique(), 'Spain', id='dropdown-selection2'),
        dcc.Graph(id='graph-content2')
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
