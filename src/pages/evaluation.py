from dash import html, Output, Input, callback, dcc, State, ctx
import dash_ag_grid
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from .tinyllama import sent_classifier, news_classifier, make_confusion_matrix
from pages.data_selection import select_dataset
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix


evaluation = html.Div(children=[
    dcc.Store(id='prompt-predictions'),
    html.H1(children='Prompt Evaluation', style={'textAlign':'center'}),
    html.P(children="""
                    This section evaluates all the prompt on all the selected samples from the first section. 
                    After testing all the prompts on the data, various interactive statistics will be shown.
                    """),
    html.Div([
        html.Div([
            html.Div([
                html.P([html.Span("Used dataset: ", style={"fontWeight": "bold"}), html.Span(id="eval-dataset")]),
                html.P([html.Span("Number of samples selected: ", style={"fontWeight": "bold"}), html.Span(id="eval-num-samples")]),
                html.P([html.Span("Total number of prompts created: ", style={"fontWeight": "bold"}), html.Span(id="eval-num-prompts")]),
                html.P([html.Span("Description: ", style={"fontWeight": "bold"}), html.Span(id="eval-description")]),
                html.P([html.Span("Labels: ", style={"fontWeight": "bold"}), html.Span(id="eval-labels")]),
            ]), 
            html.Button('Run prompts', id='run-all-prompts-btn'),
        ], style={"width": "48%"}),
        html.Div(
            id="confusion-matrix-container", 
            children=dcc.Graph(id="confusion-matrix",  figure = {},),
            style={"width": "48%"}
        ),
    ], style={"display": "flex", "justifyContent": "space-between"}),

    dcc.Loading(
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
            id='evaluation-table'
        ), 
        style={"marginTop": "30px", "marginBottom": "30px"}),
        id="loading-evaluation"
    ),
    html.Div(
        id="eval-prompts-container", 
        style={'width':'100%',  "border": "1px solid black", 'maxHeight': '300px', 
               'overflowY': 'scroll', 'display': 'flex', 'gap': '15px', 
               'marginTop': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'
               }, 
        children=[]),
    ])

@callback(
    Output("eval-dataset", "children"),
    Input("dataset-selection", "value"),
    Input("dataset-split", "value")
)
def update_name(name, split):
    if not name or not split:
        return "no dataset selected..."
    
    return f"{name} ({split})"

@callback(
    Output("eval-num-samples", "children"),
    Input("n-samples", "value")
)
def update_num_samples(number):
    return f"{number}"

@callback(
    Output("eval-num-prompts", "children"),
    Input("prompt-list", "data"),
)
def update_num_prompts(prompt_list):
    if not prompt_list:
        return "0"
    
    return f"{len(prompt_list)}"

@callback(
    Output("eval-labels", "children"),
    Input("dataset-scheme", "children")
)
def update_labels(labels):
    return labels

@callback(
    Output("eval-description", "children"),
    Input("dataset-description", "children")
)
def update_description(desc):
    return desc

@callback(
    Output("evaluation-table", "rowData"),
    Output("evaluation-table", "columnDefs"),
    Output("prompt-predictions", "data"),
    Input("run-all-prompts-btn", "n_clicks"),
    State("dataset-selection", "value"),
    State("selected-dataset", "data"),
    State('prompt-list', 'data'),
    State("samples-table", "rowData"),
    running=[(Output("run-all-prompts-btn", "disabled"), True, False)]
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

    for sample_idx, sample in enumerate(samples, start=1):
        print(f"Sample {sample_idx}/{len(samples)}")
        
        text = sample['text']
        true_label = sample['label']
        pred_labels = []
        total_per_class[true_label] += 1

        # UNCOMMENT THIS TO USE MODEL PREDICTIONS
        for i, prompt in enumerate(tqdm(prompt_list)):
            prompt = prompt.format(text=text)
            pred_label, words, att_data = classifier(prompt)            
            pred_labels.append(pred_label)

            # Used by the confusion matrix.
            if 'preds' not in prediction_dict[i]:
                prediction_dict[i]['preds'] = []
            prediction_dict[i]['preds'].append((pred_label, true_label))

        ### UNCOMMENT THIS TO TEMPORARILY ALWAYS PREDICT BUSINESS
        # for i, prompt in enumerate(prompt_list):
        #     pred_label = 'Business'
        #     pred_labels.append(pred_label)
        #     if 'preds' not in prediction_dict[i]:
        #         prediction_dict[i]['preds'] = []
        #     prediction_dict[i]['preds'].append((pred_label, true_label))
        ##

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
        
    labels = list(select_dataset(dataset_name)['labels'].values())
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
    Output("eval-prompts-container", "children"),
    Input("evaluation-table", "selectedRows"),
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
            margin=dict(b=0, l=0, r=0, t=0)  # Adjust margins to ensure the text is visible
        )
        return fig, []

    predicted_labels = []
    gt_labels = []
    prompt_container_children = []
    for row in selected_rows:
        for (pred, gt) in prediction_dict[str(row['#'] - 1)]['preds']:
            predicted_labels.append(pred)
            gt_labels.append(gt)
            
        prompt_lines = []
        for line in row['Prompt'].split('\n'):
            prompt_lines.append(line)
            prompt_lines.append(html.Br())
        prompt_lines = prompt_lines[:-1]

        correct = row["Total Correct"].split(" / ")
        correct, total = int(correct[0]), int(correct[1])
        prompt_div = html.Div([
                html.Div(
                    children=[html.B(f'Prompt {row["#"]}:'), html.P(prompt_lines), html.P(row['Total Correct']),
                              
                dbc.Progress(
                    [
                        dbc.Progress(value=correct / total * 100, bar=True),
                    ]
                )],
                    style={"width":150, "padding":"15px 30px", "border": "1px solid black", "margin": "0px 20px", "marginTop": "30px",}           
                ),
            ],
        )

        prompt_container_children.append(prompt_div)

    labels = list(select_dataset(dataset_name)['labels'].values())
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

    return fig, prompt_container_children

