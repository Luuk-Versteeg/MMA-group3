from dash import Dash, html, dcc
import pandas as pd

from pages.data_selection import data_selection
from pages.evaluation import evaluation
from pages.prompt_engineering import prompt_engineering


app = Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app.layout = html.Div(
    children=[
        dcc.Tabs(
            #id='window-tabs',
            children=[
                dcc.Tab(id='tab-dataset-selection', label="Dataset Selection", children=data_selection),
                dcc.Tab(id='tab-prompt-engineering', label="Prompt Engineering", children=prompt_engineering),
                dcc.Tab(id='tab-prompt-evaluation', label="Prompt Evaluation", children=evaluation)
                
            ], style={"width": "100%"}),
        # html.Div(id='page-container', children=[
        #     data_selection,
        #     prompt_engineering,
        #     evaluation
        # ])
    ]
)

# app.layout = dbc.Container([
    
#     data_selection,
#     prompt_engineering,
#     evaluation
# ], id="carousel")

# @callback(
#     Output('dataset-selection', 'style'),
#     Output('prompt-engineering', 'style'),
#     Output('evaluation', 'style'),
#     Input('window-tabs', 'value')
# )
# def change_page(selected_tab):   
#     print(selected_tab)
#     print("--------------")
#     if selected_tab == 'tab-1':
#         return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
#     elif selected_tab == 'tab-2':
#         return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
#     elif selected_tab == 'tab-3':
#         return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}

#     return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run(debug=True)
