from dash import Dash, html, dcc

from pages.data_selection import data_selection
from pages.evaluation import evaluation
from pages.prompt_engineering import prompt_engineering


app = Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Tabs(
            children=[
                dcc.Tab(id='tab-dataset-selection', label="Dataset Selection", children=data_selection),
                dcc.Tab(id='tab-prompt-engineering', label="Prompt Engineering", children=prompt_engineering),
                dcc.Tab(id='tab-prompt-evaluation', label="Prompt Evaluation", children=evaluation)
                
            ], style={"width": "100%"}),
    ],
    style={"padding": "30px 40px"}
)


if __name__ == '__main__':
    app.run(debug=True)
