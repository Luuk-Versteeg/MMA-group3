from dash import html, Output, Input, callback, dcc, State, ctx
import dash_ag_grid
import plotly.figure_factory as ff
import plotly.graph_objects as go

import pandas as pd
from .tinyllama import sent_classifier, news_classifier, make_confusion_matrix
from pages.data_selection import select_dataset
import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix


evaluation = html.Div(children=[
    dcc.Store(id='prompt-predictions'),
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
            "rowSelection": "multiple",
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
    Output("prompt-predictions", "data"),
    Input("run-all-prompts-btn", "n_clicks"),
    State("dataset-selection", "value"),
    State("selected-dataset", "data"),
    State('prompt-list', 'data'),
#    State('prompt-predictions', 'data'),
    State("samples-table", "rowData")
)
def update_evaluation_table(button_clicked, dataset_name, selected_dataset, prompt_list, samples):
    ctx_id = ctx.triggered_id
    if ctx_id != 'run-all-prompts-btn':
        return [], [], {}

    if dataset_name == "AG News":
        classifier = news_classifier
    if dataset_name == "Amazon Polarity" or dataset_name == "GLUE/sst2":
        classifier = sent_classifier

    ## prompt : {label1 : #, label2 : #, unknown : #}
    prediction_dict = defaultdict(dict)
    total_per_class = defaultdict(int)

    for sample in samples:
        text = sample['text']
        true_label = sample['label']
        pred_labels = []
        total_per_class[true_label] += 1

        # UNCOMMENT THIS TO USE MODEL PREDICTIONS
        # for i, prompt in enumerate(prompt_list):
        #     prompt = prompt.format(text=text)
        #     pred_label = classifier(prompt)            
        #     pred_labels.append(pred_label)

        #     # Used in the confusion matrix.
        #     if 'preds' not in prediction_dict[i]:
        #         prediction_dict[i]['preds'] = []
        #     prediction_dict[i]['preds'].append((pred_label, true_label))

        ### TEMPORARILY ALWAYS PREDICT BUSINESS
        for i, prompt in enumerate(prompt_list):
            pred_label = 'Business'
            pred_labels.append(pred_label)
            if 'preds' not in prediction_dict[i]:
                prediction_dict[i]['preds'] = []
            prediction_dict[i]['preds'].append((pred_label, true_label))
        ###
                
        for idx, (pred_label, new_prompt) in enumerate(zip(pred_labels, prompt_list)):
            if true_label not in prediction_dict[idx]:
                prediction_dict[idx][true_label] = 0
            if 'Unknown' not in prediction_dict[idx]:
                prediction_dict[idx]['Unknown'] = 0

            if pred_label == 'Unknown':
                prediction_dict[idx]['Unknown'] += 1
            else: 
                if true_label == pred_label:
                    prediction_dict[idx][true_label] += 1

        
        
    labels = select_dataset(dataset_name)['scheme'].split(", ")
    colDefs = [{'field':'#'}, {"field":"Prompt"}, {"field":"Total Correct"}]
    for label in labels:
        colDefs.append({"field": label})

    colDefs.append({"field": "Unknown"})
    predictions = prediction_dict

    rowData = []
    for i, prompt in enumerate(prompt_list):
        if i not in predictions:
            continue

        preds = predictions[i]

        row = {}
        row['#'] = i + 1
        row['Prompt'] = prompt
        total_correct = 0
        for label in labels:
            if label not in preds:
                row[label] = 0
            else:
                row[label] = f"{preds[label]} / {total_per_class[label]}"
                total_correct += preds[label]

        row['Unknown'] = preds['Unknown']
        row['Total Correct'] = f"{total_correct} / {len(samples)}"

        rowData.append(row)

    return rowData, colDefs, prediction_dict

@callback(
    Output("confusion-matrix", "figure"),
    # Input("evaluation-table", "rowData"),
    Input("evaluation-table", "selectedRows"),
    #Input("selected-dataset", "data"),
    #Input("generated-prompts-container", "children"),
    #Input("run-all-prompts-btn", "n_clicks"),
    State("dataset-selection", "value"),
    State("prompt-predictions", "data")
)
def update_confusion_matrix(selected_rows, dataset_name, prediction_dict):
    ctx_id = ctx.triggered_id
    if ctx_id != 'evaluation-table' or len(selected_rows) == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="No prompt selected...",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=28, color="gray")
                )
            ],
            margin=dict(b=0, l=0, r=0, t=100)  # Adjust margins to ensure the text is visible
        )
        return fig

    predicted_labels = []
    gt_labels = []
    for row in selected_rows:
        for (pred, gt) in prediction_dict[str(row['#'] - 1)]['preds']:
            predicted_labels.append(pred)
            gt_labels.append(gt)
    labels = select_dataset(dataset_name)['scheme'].split(", ")
    cm = confusion_matrix(gt_labels, predicted_labels, labels=labels)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size":12},
        hoverongaps=False,
        colorscale='Viridis'))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label')

    return fig

