from typing import Tuple, List, Union
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import dash_bootstrap_components as dbc
from dashboard_non_sup import server
import base64


def data_prep_roc(Y_test: np.ndarray, Y_pred: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    false_positive_rate, true_positive_rate, threshold = metrics.roc_curve(Y_test, Y_pred)
    number_frauds = sum(Y_test == 1)
    number_non_frauds = sum(Y_test == 0)

    df_rate = pd.DataFrame({"false_positive_rate": false_positive_rate, "true_positive_rate": true_positive_rate,
                            "threshold": threshold,
                            "false_positive_number": false_positive_rate * number_non_frauds,
                            "true_positive_number": true_positive_rate * number_frauds})
    df_rate = df_rate.loc[
              (df_rate['false_positive_rate'] > 0) | (df_rate['true_positive_rate'] > 0) | (df_rate["threshold"] <= 1),
              :]
    df_rate["true_positive_rate"] = df_rate["true_positive_rate"].apply(lambda x: round(x, 2) * 100)
    df_rate["false_positive_number"] = df_rate["false_positive_number"].apply(lambda x: int(x))
    df_rate["true_positive_number"] = df_rate["true_positive_number"].apply(lambda x: int(x))
    df_rate_drop_duplicates = df_rate.drop_duplicates(subset="true_positive_rate", keep='first')

    # All values of true_positive_rate (0 to 100)
    df_rate_drop_duplicates_all = pd.DataFrame(columns=["false_positive_rate", "true_positive_rate", "threshold",
                                                        "false_positive_number", "true_positive_number"])
    df_rate_drop_duplicates_all['true_positive_rate'] = np.arange(1, 101, 1)
    for index, row in df_rate_drop_duplicates_all.iterrows():
        closest_value = min(df_rate_drop_duplicates['true_positive_rate'].values,
                            key=lambda x: abs(x - row["true_positive_rate"]))
        df_rate_drop_duplicates_all.loc[index, "false_positive_rate"] = \
            df_rate_drop_duplicates.loc[df_rate["true_positive_rate"] == closest_value, "false_positive_rate"].values[0]
        df_rate_drop_duplicates_all.loc[index, "threshold"] = \
            df_rate_drop_duplicates.loc[df_rate["true_positive_rate"] == closest_value, "threshold"].values[0]
        df_rate_drop_duplicates_all.loc[index, "false_positive_number"] = \
            df_rate_drop_duplicates.loc[df_rate["true_positive_rate"] == closest_value, "false_positive_number"].values[
                0]
        df_rate_drop_duplicates_all.loc[index, "true_positive_number"] = \
            df_rate_drop_duplicates.loc[df_rate["true_positive_rate"] == closest_value, "true_positive_number"].values[
                0]

    # All values of threshold (0 to 1)
    df_all_rate = pd.DataFrame(
        columns=["false_positive_rate", "true_positive_rate", "threshold", "false_positive_number",
                 "true_positive_number"])
    df_all_rate['threshold'] = np.arange(0.0, 1.0, 0.02)
    for index, row in df_all_rate.iterrows():
        closest_value = min(df_rate['threshold'].values, key=lambda x: abs(x - row["threshold"]))
        df_all_rate.loc[index, "false_positive_rate"] = \
            df_rate.loc[df_rate["threshold"] == closest_value, "false_positive_rate"].values[0]
        df_all_rate.loc[index, "true_positive_rate"] = \
            df_rate.loc[df_rate["threshold"] == closest_value, "true_positive_rate"].values[0]
        df_all_rate.loc[index, "false_positive_number"] = \
            df_rate.loc[df_rate["threshold"] == closest_value, "false_positive_number"].values[0]
        df_all_rate.loc[index, "true_positive_number"] = \
            df_rate.loc[df_rate["threshold"] == closest_value, "true_positive_number"].values[0]

    return df_all_rate, df_rate_drop_duplicates_all


# The ROC curve is used to display different pair of false positives, true positives for different thresholds.
# The area under this curve allows us to obtain a measure of the quality of our model.
def function_roc_curve(Y_test: np.ndarray, Y_pred: np.ndarray) -> Tuple[go.Figure, float]:
    # calculation of false positives and true positives for all classification thresholds.
    false_positive_rate, true_positive_rate, threshold = metrics.roc_curve(Y_test, Y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    # we keep only what is less than 10% because the proportion of false positive beyond is directly to be rejected
    fpr_10p, tpr_10p = false_positive_rate[false_positive_rate < 0.1], true_positive_rate[false_positive_rate < 0.1]
    roc_auc_10p = metrics.auc(fpr_10p, tpr_10p)

    # matrixRoc is used to give the threshold, as legend, for each pair "rate of true positives" / "rate of false positives"
    matrixRoc = pd.DataFrame({'threshold': threshold})
    matrixRoc['threshold'] = matrixRoc['threshold'].apply(lambda row: 'Threshold = ' + str(row))

    # Draw the ROC curve
    roc_curve = go.Scatter(
        x=false_positive_rate,
        y=true_positive_rate,
        text=matrixRoc['threshold'],
        name='ROC curve of the prediction'
    )

    # Trace the worst case (random : for any threshold, true_positive_rate = false_positive_rate because they are detected identically
    worst_case = go.Scatter(
        x=np.linspace(0, 1, 500),
        y=np.linspace(0, 1, 500),
        name="Worst case : random"
    )

    layout_comp = go.Layout(
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='closest',
        xaxis=dict(
            title='False positive rate',
            zeroline=False,
        ),
        yaxis=dict(
            title='True positive rate'
        ),
        annotations=[
            dict(text="Area Under the ROC Curve (AUC) : " + str(int(roc_auc * 100000) / 100000), x=0.75, y=0.25,
                 showarrow=False),
            dict(text="Area Under the ROC Curve : " + str(
                int(roc_auc_10p * 100000) / 100000) + " for the first 10%", x=0.75, y=0.15,
                 showarrow=False)
        ]
    )

    data_comp = [roc_curve, worst_case]
    graph = go.Figure(data=data_comp, layout=layout_comp)
    return graph, roc_auc_10p


def function_pr_curve(Y_test: np.ndarray, Y_pred: np.ndarray) -> Tuple[go.Figure, float]:
    precision, recall, threshold = metrics.precision_recall_curve(Y_test, Y_pred)

    pr_auc = metrics.auc(recall, precision)

    matrixPr = pd.DataFrame({'threshold': threshold})
    matrixPr['threshold'] = matrixPr['threshold'].apply(lambda row: 'Threshold = ' + str(row))

    # Pr curve
    pr_curve = go.Scatter(
        x=recall,
        y=precision,
        text=matrixPr['threshold'],
        name='PR curve of the prediction'
    )

    layout_comp = go.Layout(
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='closest',
        xaxis=dict(
            title='Recall',
            zeroline=False,
        ),
        yaxis=dict(
            title='Precision'
        ),
        annotations=[dict(text="Area Under the ROC Curve (AUC) : " + str(int(pr_auc * 100000) / 100000), x=0.75, y=0.5,
                          showarrow=False)]
    )

    data_comp = [pr_curve]
    graph = go.Figure(data=data_comp, layout=layout_comp)
    return graph, pr_auc


def function_df_score_roc_pr(roc_auc_10p: float, pr_auc: float,
                             models: List[List[Union[str, List[int]]]]) -> pd.DataFrame:
    df_score = pd.DataFrame(columns=["model", "score_auc_roc_10", "score_auc_pr"])
    df_score['model'] = [models[model][2] for model in range(len(models))]
    df_score['score_auc_roc_10'] = [
        round(function_roc_curve(models[model][0], models[model][1])[1], 4) for model in
        range(len(models))]
    df_score['score_auc_pr'] = [round(function_pr_curve(models[model][0], models[model][1])[1], 4) for
                                model in
                                range(len(models))]

    return df_score


def function_array_tab3(false_positive_number: int, true_positive_number: int, threshold: float) -> go.Figure:
    fig = go.Figure(
        data=[go.Table(
            header=dict(values=['Number of false positives', 'Number of detected frauds', "Threshold of the model"],
                        fill=dict(color='#C2D4FF'),
                        align=['center'] * 5),
            cells=dict(values=[false_positive_number, true_positive_number, threshold],
                       fill=dict(color='#F5F8FF'),
                       align=['center'] * 5)
        )],
        layout=go.Layout(margin=dict(l=0, r=0, t=10, b=10), height=100)
    )
    fig.layout.uirevision = True
    return fig


def function_pie_chart(false_positive_number: int, true_positive_number: int) -> go.Figure:
    fig = go.Figure(
        data=[go.Pie(labels=['False positive', 'True positive'], values=[false_positive_number, true_positive_number],
                     marker=dict(colors=['#8050ff', '#e60000'])
                     )],
        layout=go.Layout(margin=dict(l=0, r=0, t=20, b=10))
    )
    fig.layout.uirevision = True
    return fig


def function_graph_fraud_tab3(threshold: float, df_all_rate: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_all_rate.loc[df_all_rate['threshold'] >= threshold, "threshold"].values,
               y=df_all_rate.loc[
                   df_all_rate['threshold'] >= threshold, "true_positive_number"].values,
               marker_color='#e60000', marker_opacity=1, name='True positives'
               ))
    fig.add_trace(
        go.Bar(x=df_all_rate.loc[df_all_rate['threshold'] < threshold, "threshold"].values,
               y=df_all_rate.loc[
                   df_all_rate['threshold'] < threshold, "true_positive_number"].values,
               marker_color='#e60000', marker_opacity=0.2, showlegend=False
               ))
    fig.add_trace(
        go.Bar(x=df_all_rate.loc[df_all_rate['threshold'] >= threshold, "threshold"].values,
               y=df_all_rate.loc[
                   df_all_rate['threshold'] >= threshold, "false_positive_number"].values,
               marker_color='#8050ff', marker_opacity=1, name='False positives'
               ))
    fig.add_trace(
        go.Bar(x=df_all_rate.loc[df_all_rate['threshold'] < threshold, "threshold"].values,
               y=df_all_rate.loc[
                   df_all_rate['threshold'] < threshold, "false_positive_number"].values,
               marker_color='#8050ff', marker_opacity=0.2, showlegend=False
               ))
    fig.update_layout(barmode='group', height=550, margin=dict(l=0, r=0, t=10, b=10))
    fig.update_xaxes(title_text="Probability to be a fraud")
    fig.update_yaxes(title_text="Number of alerts")

    return fig


def header_dash(models: List[List[Union[str, List[int]]]]) -> html.Div:
    image_filename: str = 'static/logo_spyra_white.png'
    encoded_image: base64.b64encode = base64.b64encode(open(image_filename, 'rb').read())

    return html.Div([
        dbc.Row([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '40%',
                                                                                                           'margin-left': '10%',
                                                                                                           'margin-top': '5%'})
                        ], width=3),
                        dbc.Col([
                            html.H1(children='Dashboard', style={'fontSize': 60})
                        ], width={"size": 4, "offset": 1}),
                        dbc.Col([
                            html.H3('Model : ')
                        ], width=1),
                        dbc.Col([
                            dcc.Dropdown(
                                options=[
                                    {'label': models[i][2], 'value': models[i][2]} for i in range(0, len(models))
                                ],
                                value=models[0][2],
                                id='model_selector',
                                style={'textAlign': 'center', 'color': 'black'}
                            )
                        ], width=2)
                    ], align='center')
                ])
            ], color="dark", inverse=True)
        ], align='center')
    ])


def tab1_content(Y_test: np.ndarray, Y_pred: np.ndarray) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ROC Curve (Receiver operating characteristic)", style={'textAlign': 'center',
                                                                                           'fontSize': 20}),
                    dbc.CardBody([
                        dcc.Graph(id='id_roc_curve', figure=function_roc_curve(Y_test, Y_pred)[0])
                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ])
        ], align='stretch')
    ])


def tab2_content(Y_test: np.ndarray, Y_pred: np.ndarray) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("PR Curve (Precision Recall)", style={'textAlign': 'center', 'fontSize': 20}),
                    dbc.CardBody([
                        dcc.Graph(id='id_pr_curve', figure=function_pr_curve(Y_test, Y_pred)[0])
                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ])
        ], align='stretch')
    ])


def tab3_content() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualization to help you to set the threshold of the model",
                                   style={'textAlign': 'center', 'fontSize': 20}),
                    dbc.CardBody([
                        dcc.Graph(id='id_array_fraud_tab3'),
                        dbc.Label("% of fraud you want to detect : ", id='id_label_slider', html_for="my_slider"),
                        dcc.Slider(id="my_slider", className='mt-4', value=100, min=1, max=100, step=1),
                        dcc.Graph(id='id_graph_pie_chart')

                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Number of alerts by probability to be a fraud",
                                   style={'textAlign': 'center', 'fontSize': 20}),
                    dbc.CardBody([
                        dbc.Label("Threshold of the model : ", id='id_label_slider_2', html_for="my_slider2"),
                        dcc.Slider(id="my_slider2", className='mt-4', value=0, min=0, max=1, step=0.01),
                        dcc.Graph(id='id_graph_fraud_tab3')
                    ])
                ], className=["h-100 d-inline-block", "w-100"], color="dark", outline=True)
            ], width=6)
        ], align='stretch')
    ])


def tab4_content(df_score: pd.DataFrame, models: List[List[Union[str, List[int]]]]) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Benchmark of all the models", style={'textAlign': 'center', 'fontSize': 20}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            columns=[{"id": "model", "name": "Models", "type": "text"},
                                     {"id": "score_auc_roc_10", "name": "Score AUC ROC 10%", "type": "numeric"},
                                     {"id": "score_auc_pr", "name": "Score AUC PR", "type": "numeric"}],
                            data=df_score.to_dict("rows"),
                            style_header={'backgroundColor': '#C2D4FF', 'fontWeight': 'bold'},
                            style_data_conditional=[{
                                'if': {'column_id': 'Nom du modèle'},
                                'backgroundColor': '#F5F8FF'
                            }],
                            style_cell={'textAlign': 'center'},
                            style_table={'width': '90%', 'margin-left': '5%', 'margin-top': '1%', 'overflowX': 'scroll'},
                            sort_action='native'
                        )
                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ])
        ], align='stretch'),

        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(["Choose the model you want to export : "], style={'textAlign': 'center',
                                                                                    'fontSize': 18}),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    options=[
                                        {'label': models[i][2], 'value': i} for i in range(0, len(models))
                                    ],
                                    value=0,
                                    id='true_model_selector',
                                    style={
                                        'textAlign': 'center', 'margin-top': '10%'
                                    }
                                )
                            ], width={"size": 8, "offset": 2})
                        ], align='stretch'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("EXPORT", id='id_button_export_model', className='d-grid col-4 mx-auto',
                                           n_clicks=0, style={'margin-top': '5%'})
                            ])
                        ], align='stretch')
                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ], width={"size": 4, "offset": 4})
        ], align='stretch')
    ])


def init_var(models: List[List[Union[str, List[int]]]]) -> Tuple[np.ndarray, np.ndarray]:
    Y_test, Y_pred, name = models[0]
    return Y_test, Y_pred


def dashViz(models: List[List[Union[str, List[int]]]]) -> dash.Dash:
    external = [dbc.themes.BOOTSTRAP]
    app: dash.Dash = dash.Dash(__name__, server=server, external_stylesheets=external)
    app.config['suppress_callback_exceptions'] = True

    Y_test, Y_pred = init_var(models)
    df_score = function_df_score_roc_pr(function_roc_curve(Y_test, Y_pred)[1], function_pr_curve(Y_test, Y_pred)[1],
                                        models)

    app.layout = html.Div(children=[
        header_dash(models),
        html.Br(),
        dbc.Tabs([
            dbc.Tab(label='ROC curve', tab_id='id_tab1',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='PR curve', tab_id='id_tab2',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='Threshold', tab_id='id_tab3',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='Benchmark', tab_id='id_tab4',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25})
        ],
            id="tabs",
            active_tab="id_tab1"),
        dbc.CardBody(id="card_contents", className="card-text")
    ])

    @app.callback(Output("card_contents", "children"), [Input("tabs", "active_tab")])
    def switch_tabs(active_tabs: str):
        if active_tabs == "id_tab1":
            return tab1_content(Y_test, Y_pred)
        elif active_tabs == "id_tab2":
            return tab2_content(Y_test, Y_pred)
        elif active_tabs == "id_tab3":
            return tab3_content()
        elif active_tabs == "id_tab4":
            return tab4_content(df_score, models)

    @app.callback(Output('id_roc_curve', 'figure'), [Input('model_selector', 'value')])
    def updateROCCurve(model_name: str):
        for model_number in range(len(models)):
            if model_name == models[model_number][2]:
                Y_test, Y_pred, name = models[model_number]
                return function_roc_curve(Y_test, Y_pred)[0]

    @app.callback(Output('id_pr_curve', 'figure'), [Input('model_selector', 'value')])
    def updatePRCurve(model_name: str):
        for model_number in range(len(models)):
            if model_name == models[model_number][2]:
                Y_test, Y_pred, name = models[model_number]
                return function_pr_curve(Y_test, Y_pred)[0]

    # Link between the slider1 and his label
    @app.callback(Output('id_label_slider', 'children'), [Input('my_slider', 'value')])
    def updateSliderText(value: int):
        return "% of outliers you want to detect : " + str(value)

    # Link between the slider2 and his model
    @app.callback(Output('id_label_slider_2', 'children'), [Input('my_slider2', 'value')])
    def updateSlider2Text(value: int):
        return "Threshold : " + str(value)

    @app.callback([Output('id_array_fraud_tab3', 'figure'), Output('id_graph_pie_chart', 'figure')],
                  [Input('model_selector', 'value'), Input('my_slider', 'value')])
    def updateArrayTab3(model_name: str, value_slider: int):
        for model_number in range(len(models)):
            if model_name == models[model_number][2]:
                Y_test, Y_pred, name = models[model_number]
                df_rate = data_prep_roc(Y_test, Y_pred)[1]
                false_positive_number = df_rate.loc[
                    df_rate["true_positive_rate"] == int(value_slider), "false_positive_number"].values[0]
                true_positive_number = df_rate.loc[
                    df_rate["true_positive_rate"] == int(value_slider), "true_positive_number"].values[0]
                threshold = df_rate.loc[
                    df_rate["true_positive_rate"] == int(value_slider), "threshold"].values[0]
                return function_array_tab3(false_positive_number, true_positive_number, threshold), \
                       function_pie_chart(false_positive_number, true_positive_number)

    @app.callback(Output('id_graph_fraud_tab3', 'figure'),
                  [Input('model_selector', 'value'), Input('my_slider2', 'value')])
    def updateGraphTab3(model_name: str, value_slider: float):
        for model_number in range(len(models)):
            if model_name == models[model_number][2]:
                Y_test, Y_pred, name = models[model_number]
                df_rate = data_prep_roc(Y_test, Y_pred)[0]
                return function_graph_fraud_tab3(float(value_slider), df_rate)

    return app
