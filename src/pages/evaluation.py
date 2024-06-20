from dash import html, Output, Input, callback
import dash_ag_grid


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
        id='confusion-matrix'
    ), style={"marginTop": "30px", "marginBottom": "30px", "width": "500px"})
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
    Output("confusion-matrix", "rowData"),
    Output("confusion-matrix", "columnDefs"),
    Input("evaluation-table", "rowData"),
    Input("generated-prompts-container", "children"),
    Input("run-all-prompts-btn", "value")

)
def update_confusion_matrix(data, prompts, button_clicked):
    return [], []
