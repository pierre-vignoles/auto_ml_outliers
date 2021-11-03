from typing import List, Union
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dash_table
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import base64
from flask import Flask

server = Flask(__name__)


def function_display_outliers(X: np.ndarray, centers: np.ndarray, pred: pd.DataFrame) -> go.Figure:
    x0 = X[pred.iloc[:, 1] == 0, 0]
    y0 = X[pred.iloc[:, 1] == 0, 1]

    x1 = X[pred.iloc[:, 1] == 1, 0]
    y1 = X[pred.iloc[:, 1] == 1, 1]

    x2 = centers[:, 0]
    y2 = centers[:, 1]

    trace = go.Scatter(
        x=x0, y=y0,
        mode='markers',
        name='Legit values',
        marker=dict(
            size=2,
            color='blue'
        )
    )
    trace1 = go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='Outliers',
        marker=dict(
            size=4,
            color='red'
        )
    )
    trace2 = go.Scatter(
        x=x2, y=y2,
        mode='markers',
        name='Centers',
        marker=dict(
            size=10,
            color='black'
        )
    )
    data = [trace, trace1, trace2]
    layout = go.Layout(
        xaxis=dict(
            title='var1'),
        yaxis=dict(
            title='var2'),
        margin=dict(l=0, r=0, t=20, b=10)
    )
    return go.Figure(data=data, layout=layout)


def function_display_outliers2(X: np.ndarray, pred: pd.DataFrame) -> go.Figure:
    x0 = X[pred.iloc[:, 1] == 0, 0]
    y0 = X[pred.iloc[:, 1] == 0, 1]

    x1 = X[pred.iloc[:, 1] == 1, 0]
    y1 = X[pred.iloc[:, 1] == 1, 1]

    trace = go.Scatter(
        x=x0, y=y0,
        mode='markers',
        name='Legit values',
        marker=dict(
            size=2,
            color='blue'
        )
    )
    trace1 = go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='Outliers',
        marker=dict(
            size=4,
            color='red'
        )
    )
    data = [trace, trace1]
    layout = go.Layout(
        xaxis=dict(
            title='var1'),
        yaxis=dict(
            title='var2'),
        margin=dict(l=0, r=0, t=20, b=10)
    )
    return go.Figure(data=data, layout=layout)


def function_display_clusters(X: np.ndarray, centers: np.ndarray, pred: pd.DataFrame) -> go.Figure:
    x0 = X[:, 0]
    y0 = X[:, 1]
    x1 = centers[:, 0]
    y1 = centers[:, 1]

    trace = go.Scatter(
        x=x0, y=y0,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color=pred,
            # colorscale='Blackbody'
        )
    )
    trace1 = go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='Centers',
        marker=dict(
            size=10,
            color='black'
        )
    )
    data = [trace, trace1]
    layout = go.Layout(
        xaxis=dict(
            title='var1'),
        yaxis=dict(
            title='var2'),
        margin=dict(l=0, r=0, t=0, b=10),
        height=600
    )
    a = go.Figure(data=data, layout=layout)
    return a


def function_display_clusters2(X: np.ndarray, pred: pd.DataFrame) -> go.Figure:
    x0 = X[:, 0]
    y0 = X[:, 1]

    trace = go.Scatter(
        x=x0, y=y0,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color=pred,
        )
    )
    data = [trace]
    layout = go.Layout(
        xaxis=dict(
            title='var1'),
        yaxis=dict(
            title='var2'),
        margin=dict(l=0, r=0, t=0, b=10),
        height=500
    )
    a = go.Figure(data=data, layout=layout)
    return a


def function_array_outliers(X2: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    outliers: pd.Int64Index = pred[pred.iloc[:, 0] == 1].index
    moy: float = round(X2.iloc[outliers, :].mean(axis=0), 2)
    mini: float = X2.iloc[outliers, :].min(axis=0)
    maxi: float = X2.iloc[outliers, :].max(axis=0)
    standard_deviation: float = round(X2.iloc[outliers, :].std(axis=0), 1)

    a = go.Figure(data=[go.Table(
        header=dict(values=['Variable', 'Mean', 'Min', 'Max', 'Standard deviation'],
                    fill=dict(color='#C2D4FF'),
                    align=['center'] * 5),
        cells=dict(values=[X2.columns, moy, mini, maxi, standard_deviation],
                   fill=dict(color='#F5F8FF'),
                   align=['center'] * 5)
    )
    ],
        layout=go.Layout(margin=dict(l=0, r=0, t=10, b=10), height=250)
    )
    a.layout.uirevision = True

    return a


def function_array2_outliers(X2: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    outliers: pd.Int64Index = pred[pred.iloc[:, 0] == 1].index
    mini: float = X2.iloc[outliers, :].min(axis=0)
    maxi: float = X2.iloc[outliers, :].max(axis=0)
    m_value: pd.DataFrame = pd.DataFrame(mini)

    for i in range(0, len(mini)):
        if mini[i] == maxi[i]:
            m_value.iloc[i] = mini[i]
        else:
            m_value.iloc[i] = np.NaN
    for index, row in m_value.iterrows():
        if row.isnull().any():
            m_value = m_value.drop(index)

    b = go.Figure(data=[go.Table(
        header=dict(values=['Variable', 'Value'],
                    fill=dict(color='#C2D4FF'),
                    align=['center'] * 5),
        cells=dict(values=[m_value.index, m_value],
                   fill=dict(color='#F5F8FF'),
                   align=['center'] * 5)
    )
    ],
        layout=go.Layout(margin=dict(l=0, r=0, t=10, b=10), height=250)
    )
    b.layout.uirevision = True

    return b


def function_clusters_array(X2: pd.DataFrame, pred: pd.DataFrame, cluster_selector: int) -> go.Figure:
    pred_df: pd.DataFrame = pd.DataFrame(pred)
    cluster_number_df: pd.Int64Index = pred_df[pred_df.iloc[:, 0] == cluster_selector].index

    mean: float = round(X2.iloc[cluster_number_df, :].mean(axis=0), 2)
    mini: float = X2.iloc[cluster_number_df, :].min(axis=0)
    maxi: float = X2.iloc[cluster_number_df, :].max(axis=0)
    standard_deviation: float = round(X2.iloc[cluster_number_df, :].std(axis=0), 1)

    tc = go.Figure(data=[go.Table(
        header=dict(values=['Variable', 'Mean', 'Min', 'Max', 'Standard deviation'],
                    fill=dict(color='#C2D4FF'),
                    align=['center'] * 5),
        cells=dict(values=[X2.columns, mean, mini, maxi, standard_deviation],
                   fill=dict(color='#F5F8FF'),
                   align=['center'] * 5)
    )
    ]).update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)

    tc.layout.uirevision = True

    return tc


def function_extreme_number(X2: pd.DataFrame, pourc: float) -> int:
    return int((1 - float(pourc)) * len(X2))


def sil_score(sil: float) -> str:
    return 'Silhouette Score = ' + str(float(+sil))


def function_array_predict_outliers(X2: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    outliers: pd.Int64Index = pred[pred.iloc[:, 0] == 1].index
    df: pd.DataFrame = pd.DataFrame()
    df = X2.iloc[outliers, :]
    df.insert(0, 'id', outliers)
    return df


def header_dash(model: List[Union[int, float, List[int], str, List[float], pd.DataFrame]]) -> html.Div:
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
                                    {'label': model[i][0][3], 'value': i} for i in range(0, len(model))
                                ],
                                value=0,
                                id='model_selector',
                                style={'textAlign': 'center', 'color': 'black'}
                            )
                        ], width=2)
                    ], align='center')
                ])
            ], color="dark", inverse=True)
        ], align='center')
    ])


def tab1_content(model: List[Union[int, float, List[int], str, List[float], pd.DataFrame]]) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualization of the different clusters", style={'textAlign': 'center',
                                                                                     'fontSize': 20}),
                    dbc.CardBody([
                        dcc.Graph(id="display_of_clusters"),
                        html.Br(),
                        html.Div(id='sil_score', style={'textAlign': 'center'})
                    ])
                ], className=["h-100 d-inline-block", "w-100"], color="dark", outline=True
                )
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.Div("Informations about the selected cluster : ")
                            ],  width={"size": 6, "offset": 1}),
                            dbc.Col([
                                dcc.Dropdown(
                                    options=[
                                        {'label': i, 'value': i} for i in range(0, model[0][0][1])
                                    ],
                                    value=0,
                                    id='cluster_selector'
                                )
                            ], width=2)
                        ], align='center')
                    ], style={'textAlign': 'center', 'fontSize': 20}),
                    dbc.CardBody([
                        dcc.Graph(id='clusters_array')
                    ])
                ], className=["h-100 d-inline-block", "w-100"], color="dark", outline=True)
            ], width=6)
        ], align='stretch')
    ])


def tab2_content() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualization of outliers in each cluster", style={'textAlign': 'center',
                                                                                       'fontSize': 20}),
                    dbc.CardBody([
                        dbc.Label("% of outliers you want : ", id='id_label_slider', html_for="my_slider"),
                        dbc.Input(type="range", id="my_slider", min=0.90, max=1.0, step=0.01, value=0.97,
                                  style={'width': '50%', 'margin-left': 40}, className="form-group"),
                        dcc.Graph(id='graph_outliers')
                    ])

                ], className=["h-100 d-inline-block", "w-100"], color="dark", outline=True)
            ], width=6),

            dbc.Col([
                dbc.Row([
                    dbc.Card([
                        dbc.CardHeader("Informations about the outliers", style={'textAlign': 'center',
                                                                                 'fontSize': 20}),
                        dbc.CardBody([
                            dcc.Graph(id='id_array_outliers')
                        ])
                    ], className=["h-50 d-inline-block", "w-100"], color="dark", outline=True),
                ], align='stretch', className="g-0"),
                html.Br(),
                dbc.Row([
                    dbc.Card([
                        dbc.CardHeader("Informations present in each outliers", style={'textAlign': 'center',
                                                                                       'fontSize': 20}),
                        dbc.CardBody([
                            dcc.Graph(id='id_array2_outliers')
                        ])
                    ], className=["h-50 d-inline-block", "w-100"], color="dark", outline=True)
                ], align='stretch', className="g-0")
            ], width=6)
        ], align='stretch')
    ])


def tab3_content(model: List[Union[int, float, List[int], str, List[float], pd.DataFrame]], X2: pd.DataFrame) -> html.Div:
    return html.Div([
        dbc.Card([
            dbc.CardHeader("Informations about all the outliers", style={'textAlign': 'center', 'fontSize': 20}),
            dbc.CardBody([
                dash_table.DataTable(id='id_array_all_outliers',
                                     columns=[{'id': i, 'name': i} for i in
                                              function_array_predict_outliers(X2, model[0][0][7]).columns],
                                     data=function_array_predict_outliers(X2, model[0][0][7]).to_dict('rows'),
                                     style_header={'backgroundColor': '#C2D4FF', 'fontWeight': 'bold'},
                                     style_data_conditional=[{'if': {'column_id': 'Nom du modèle'},
                                                              'backgroundColor': '#F5F8FF'}],
                                     style_cell={'textAlign': 'center'},
                                     style_table={'width': '90%', 'margin-left': '5%', 'margin-top': '1%',
                                                  'overflowX': 'scroll'},
                                     sort_action='native'
                                     )
            ])
        ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
    ])


def tab4_content(df_sil: pd.DataFrame, model: List[Union[int, float, List[int], str, List[float], pd.DataFrame]]) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Card([
                dbc.CardHeader("Benchmark of all the models", style={'textAlign': 'center', 'fontSize': 20}),
                dbc.CardBody([
                    dash_table.DataTable(
                        columns=[{"id": "model", "name": "Models", "type": "text"},
                                 {"id": "sil_score", "name": "Silhouette score", "type": "numeric"},
                                 {"id": "calin_score", "name": "Calinski score", "type": "numeric"},
                                 {"id": "davies_score", "name": "Davies score", "type": "numeric"}],
                        data=df_sil.to_dict("rows"),
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
        ]),

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
                                        {'label': model[i][0][3], 'value': i} for i in range(0, len(model))
                                    ],
                                    value=0,
                                    id='true_model_selector',
                                    style={
                                        'textAlign': 'center', 'margin-top': '10%'
                                    }
                                )
                            ], width={"size": 8, "offset": 2})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("EXPORT", id='id_button_export_model', className='d-grid col-4 mx-auto',
                                           n_clicks=0, style={'margin-top': '5%'})
                            ])
                        ], align='stretch')
                    ])
                ], className=["h-auto d-inline-block", "w-100"], color="dark", outline=True)
            ], width={"size": 4, "offset": 4})
        ])
    ])


def dashviznsup(X: np.ndarray, X2: pd.DataFrame, model: List[Union[int, float, List[int], str, List[float], pd.DataFrame]]) -> dash.Dash:
    df_sil: pd.DataFrame = pd.DataFrame({"model": model[i][0][3], "sil_score": model[i][0][2], "calin_score": model[i][0][5],
                           "davies_score": model[i][0][6]} for i in range(0, len(model)))

    external = [dbc.themes.BOOTSTRAP]

    app: dash.Dash = dash.Dash(__name__, server=server, external_stylesheets=external)
    app.config['suppress_callback_exceptions'] = True

    app.layout = html.Div(children=[
        header_dash(model),
        html.Br(),
        dbc.Tabs([
            dbc.Tab(label='Clusters', tab_id='id_tab1',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='Predictions', tab_id='id_tab2',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='Outliers', tab_id='id_tab3',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25}),
            dbc.Tab(label='Benchmark', tab_id='id_tab4',
                    tab_style={'width': '25%', 'textAlign': 'center', 'fontSize': 25})
        ],
            id="tabs",
            active_tab="id_tab1"),
        dbc.CardBody(id="card_contents", className="card-text")
    ])

    @app.callback(Output("card_contents", "children"), [Input("tabs", "active_tab")])
    def switch_tabs(active_tabs):
        if active_tabs == "id_tab1":
            return tab1_content(model)
        elif active_tabs == "id_tab2":
            return tab2_content()
        elif active_tabs == "id_tab3":
            return tab3_content(model, X2)
        elif active_tabs == "id_tab4":
            return tab4_content(df_sil, model)

    @app.callback(Output('id_array_all_outliers', 'data'),
                  [Input('my_slider', 'value'), Input('model_selector', 'value')])
    def updatetabtransaction(value_slider, value_model):
        return function_array_predict_outliers(X2, model[value_model][int(100 * value_slider - 90)][7]).to_dict('rows')

    @app.callback(Output('cluster_selector', 'options'), [Input('model_selector', 'value')])
    def updateoptionsdropdown(value_model):
        return [{'label': i, 'value': i} for i in range(0, model[value_model][0][1])]

    # Update of silhouette_score
    @app.callback(Output('sil_score', 'children'), [Input('model_selector', 'value')])
    def updatesil(value):
        return sil_score(model[value][0][2])

    # Link between the slider and his label
    @app.callback(Output('id_label_slider', 'children'), [Input('my_slider', 'value')])
    def updateSliderText(value):
        return "% of outliers you want : " + str(100 - float(value) * 100) + '%' + " / " + \
               str(function_extreme_number(X2, value))

    # Dropdown and slider
    @app.callback(Output('graph_outliers', 'figure'), [Input('my_slider', 'value'), Input('model_selector', 'value')])
    def updateanomaly(value_slider, value_model):
        return function_display_outliers(X, model[value_model][int(100 * float(value_slider) - 90)][4],
                                         model[value_model][int(100 * float(value_slider) - 90)][7]) if (
                value_model == 0 or value_model == 1) else function_display_outliers2(X, model[value_model][
            int(100 * value_slider - 90)][7])

    @app.callback(Output('id_array_outliers', 'figure'),
                  [Input('my_slider', 'value'), Input('model_selector', 'value')])
    def updatetabfraude1(value_slider, value_model):
        return function_array_outliers(X2, model[value_model][int(100 * float(value_slider) - 90)][7])

    @app.callback(Output('id_array2_outliers', 'figure'),
                  [Input('my_slider', 'value'), Input('model_selector', 'value')])
    def updatetabfraude2(value_slider, value_model):
        return function_array2_outliers(X2, model[value_model][int(100 * float(value_slider) - 90)][7])

    @app.callback(Output('clusters_array', 'figure'),
                  [Input('cluster_selector', 'value'), Input('model_selector', 'value')])
    def updatearrayclusters(value_cluster, value_model):
        return function_clusters_array(X2, model[value_model][0][0], int(value_cluster))

    @app.callback(Output('display_of_clusters', 'figure'), [Input('model_selector', 'value')])
    def updatedisplayclusters(value_model):
        return function_display_clusters(X, model[value_model][0][4], model[value_model][0][0]) if (
                value_model == 0 or value_model == 1) else function_display_clusters2(X, model[value_model][0][0])

    return app
