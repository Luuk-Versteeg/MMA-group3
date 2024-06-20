from dash import html, Output, Input, callback, dcc
import dash_ag_grid
import plotly.figure_factory as ff
from .tinyllama import sent_classifier, news_classifier, make_confusion_matrix


evaluation = html.Div(children=[
    html.H1(children='Evaluation', style={'textAlign':'center'}),
    html.Button('Run prompts', id='run-all-prompts-btn'),
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
        id='evaluation-table'
    ), style={"marginTop": "30px", "marginBottom": "30px"}),
    html.Div(id="confusion-matrix-container", 
             children=dcc.Graph(
                 id="confusion-matrix",
                 figure = {},
                 )
                 , style={"marginTop": "30px", "marginBottom": "30px", "width": "500px"})
    ])


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
    Output("confusion-matrix", "figure"),
    # Input("evaluation-table", "rowData"),
    Input("selected-dataset", "data"),
    Input("generated-prompts-container", "children"),
    Input("run-all-prompts-btn", "n_clicks")
)
def update_confusion_matrix(data, prompts, n_clicked):
    if not n_clicked:
        return ff.create_annotated_heatmap([[0,0],[0,0]], )
    ###########
    # Kan weg zodra model is gelinkt met data_selction.py
    # from datasets import load_dataset
    # data = load_dataset("fancyzhx/amazon_polarity", split="test")
    ###########
    print(data)
    # {sentence} will be replace by the sentence from the dataset.
    prompt = """
    Given this sentence: "{sentence}"

    Would you say this sentence is positive or negative?
    """
    n_samples = 5
    predicted_labels, true_labels = sent_classifier(prompt, data, n_samples, False)
    confusion_matrix = make_confusion_matrix(predicted_labels, true_labels)

    labels = []
    if -1 in predicted_labels or -1 in true_labels:
        labels.append("unknown")
    if 0 in predicted_labels or 0 in true_labels:
        labels.append("negative")
    if 1 in predicted_labels or 1 in true_labels:
        labels.append("positive")
    
    fig = ff.create_annotated_heatmap(confusion_matrix, x=labels, y=labels)
    fig.update_layout(width=500, height=500)
    return fig

