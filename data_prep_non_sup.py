import matplotlib.figure
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS

import plotly.graph_objs as go
from plotly import subplots


def split(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    return X_train, X_test


def scale(dataset: pd.DataFrame) -> pd.DataFrame:
    name_col = dataset.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset)
    X_scaled = pd.DataFrame(data=X_scaled, columns=name_col)
    return X_scaled


def pca2(dataset: pd.DataFrame) -> np.ndarray:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(dataset)
    print("% of Cumulative Explained Variance by the PCA : {} \n".format(sum(pca.explained_variance_ratio_)))
    return X_pca


def pca(dataset: pd.DataFrame, n: int) -> np.ndarray:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(dataset)
    return X_pca


def fig_pca(dataset: pd.DataFrame) -> matplotlib.figure.Figure:
    pca = PCA().fit(dataset)
    fig = Figure()
    # fig1
    axis1 = fig.add_subplot(1, 2, 1)
    axis1.plot(np.cumsum(pca.explained_variance_ratio_))
    axis1.set_xlabel('Number of features')
    axis1.set_ylabel('Cumulative Explained Variance')
    # fig2
    axis2 = fig.add_subplot(122)
    xs = range(0, len(pca.explained_variance_ratio_))
    ys = pca.explained_variance_ratio_
    axis2.bar(xs, ys)
    axis2.set_xlabel('Number of features')
    axis2.set_ylabel('Cumulative Explained Variance')
    # Subplot
    fig.subplots_adjust(wspace=0.5)
    return fig


def tsne(dataset: pd.DataFrame) -> np.ndarray:
    # PCA in dimension 50 before doing a TSNE
    if dataset.shape[1] > 50:
        pca50 = PCA(n_components=50)
        X = pca50.fit_transform(dataset)
        print(pca50.explained_variance_ratio_)
        print(sum(pca50.explained_variance_ratio_))
    # X_pca50=pd.DataFrame(data=X_pca50, index=dataset.index)
    else:
        X = dataset
    # tsne
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X_tsne = tsne.fit_transform(X)
    # X_tsne = pd.DataFrame({'var1':tsne_results[:,0], 'var2':tsne_results[:,1]})
    return X_tsne


def isomap(dataset: pd.DataFrame) -> np.ndarray:
    isomap = Isomap(n_components=2, n_jobs=-1)
    X_isomap = isomap.fit_transform(dataset)
    # X_isomap = pd.DataFrame({'var1':isomap_results[:,0], 'var2':isomap_results[:,1]})
    return X_isomap


def local_lin(dataset: pd.DataFrame) -> np.ndarray:
    local_lin = LocallyLinearEmbedding(n_components=2, n_jobs=-1)
    X_local_lin = local_lin.fit_transform(dataset)
    # X_local_lin = pd.DataFrame({'var1':local_lin_results[:,0], 'var2':local_lin_results[:,1]})
    return X_local_lin


def mds(dataset: pd.DataFrame) -> np.ndarray:
    mds = MDS(n_components=2, n_jobs=-1)
    X_mds = mds.fit_transform(dataset)
    # X_isomap = pd.DataFrame({'var1':isomap_results[:,0], 'var2':isomap_results[:,1]})
    return X_mds


# Creation of a dash which display the dimension reduction
def dash_reduc_dim(X_tsne: np.ndarray, X_isomap: np.ndarray, X_local_lin: np.ndarray, X_pca: np.ndarray,
                   X_mds: np.ndarray) -> Figure:
    xtsne = X_tsne[:, 0]
    ytsne = X_tsne[:, 1]
    xiso = X_isomap[:, 0]
    yiso = X_isomap[:, 1]
    xloc = X_local_lin[:, 0]
    yloc = X_local_lin[:, 1]
    xpca = X_pca[:, 0]
    ypca = X_pca[:, 1]
    xmds = X_mds[:, 0]
    ymds = X_mds[:, 1]

    trace0tsne = go.Scatter(
        x=xtsne, y=ytsne,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color='black'
        )
    )
    trace0iso = go.Scatter(
        x=xiso, y=yiso,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color='black'
        )
    )
    trace0loc = go.Scatter(
        x=xloc, y=yloc,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color='black'
        )
    )
    trace0pca = go.Scatter(
        x=xpca, y=ypca,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color='black'
        )
    )
    trace0mds = go.Scatter(
        x=xmds, y=ymds,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=2,
            color='black'
        )
    )

    fig = subplots.make_subplots(rows=3, cols=2, shared_yaxes=True,
                                 subplot_titles=('TSNE', 'ISOMAP', 'LOCALLY_LINEAR_EMBEDDING', 'PCA', 'MDS'))
    fig.append_trace(trace0tsne, 1, 1)
    fig.append_trace(trace0iso, 1, 2)
    fig.append_trace(trace0loc, 2, 1)
    fig.append_trace(trace0pca, 2, 2)
    fig.append_trace(trace0mds, 3, 1)

    return fig

    # # Dash
    # external = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    # app = dash.Dash(__name__, external_stylesheets=external)
    # app.config['suppress_callback_exceptions']=True
    #
    # app.layout = html.Div(children=[
    #     html.H1(children='Reductions de dimensions', style={'margin-left': '50%'}),
    #     html.Div(children=[
    #         dcc.Graph(id='affichage',
    #         figure=reduc_dim(X_tsne, X_isomap, X_local_lin, X_pca, X_mds))
    #     ])
    # ])
    #
    # return app