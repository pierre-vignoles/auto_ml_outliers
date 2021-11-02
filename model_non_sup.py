from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Speed sklearn lib
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
# from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def fonction_dist(X: List[float], Y: List[float]) -> List[float]:
    # Calculate the distance between each point and the center of its cluster
    X_ar = np.array(X)
    dist = [np.linalg.norm(x - y) for x, y in zip(X_ar, Y)]
    return dist


def n_cluster_KM(X: List[float], deb: int, fin: int) -> int:
    print("KMeans\n")
    Sum_of_squared_distances = []
    K = range(deb, fin)
    cluster_KM_df = pd.DataFrame(columns=['n_clusters', 'silhouette_score'])
    for k in K:
        KM = KMeans(n_clusters=k)
        cluster_labels = KM.fit_predict(X)
        Sum_of_squared_distances.append(KM.inertia_)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg)

        cluster_KM_df = cluster_KM_df.append({'n_clusters': k, 'silhouette_score': silhouette_avg}, ignore_index=True)

    sil_max = cluster_KM_df.silhouette_score.max()
    n_clusters_algo = int(cluster_KM_df.loc[cluster_KM_df['silhouette_score'] == sil_max, 'n_clusters'])
    print("Optimal clusters number : {}".format(int(n_clusters_algo)))

    return n_clusters_algo


def predictions_tous_les_clusters_separe_KM(X: List[float], pourc: float, centers_KM: np.ndarray,
                                            y_pred_KM: np.ndarray) -> pd.DataFrame:
    dist = fonction_dist(X, centers_KM[y_pred_KM])
    KM_y_pred2 = pd.DataFrame({'centers': centers_KM[y_pred_KM][:, 0], 'dist': dist})
    KM_y_pred2[KM_y_pred2.dist >= KM_y_pred2.groupby('centers').dist.transform('quantile', pourc)] = 1
    KM_y_pred2[KM_y_pred2.dist != 1] = 0
    return KM_y_pred2


def Kmeans(X: List[float], deb: int, fin: int) -> Tuple[np.ndarray, int, float, str, np.ndarray, float, float]:
    n_clusters_algo = n_cluster_KM(X, deb, fin)
    KM = KMeans(n_clusters=n_clusters_algo)
    KM = KM.fit(X)
    y_pred_KM = KM.predict(X)
    centers_KM = KM.cluster_centers_
    sil_max = silhouette_score(X, y_pred_KM)
    calin_max = calinski_harabasz_score(X, y_pred_KM)
    davies_max = davies_bouldin_score(X, y_pred_KM)
    print("KMeans done")
    name = 'KMeans'
    return y_pred_KM, n_clusters_algo, sil_max, name, centers_KM, calin_max, davies_max


def predictions_tous_les_clusters_separe_MS(X: List[float], pourc: float, cluster_centers_MS: np.ndarray,
                                            y_pred_MS: np.ndarray) -> pd.DataFrame:
    dist = fonction_dist(X, cluster_centers_MS[y_pred_MS])
    MS_y_pred_df = pd.DataFrame({'centers': cluster_centers_MS[y_pred_MS][:, 0], 'dist': dist})
    MS_y_pred_df[MS_y_pred_df.dist >= MS_y_pred_df.groupby('centers').dist.transform('quantile', pourc)] = 1
    MS_y_pred_df[MS_y_pred_df.dist != 1] = 0
    return MS_y_pred_df


def Meanshift(X: List[float]) -> Tuple[np.ndarray, int, float, str, np.ndarray, float, float]:
    print("Mean Shift\n")

    # we use the function of sklearn to find the hyperparameter of the MeanShift
    from sklearn.cluster import estimate_bandwidth
    nb_bandwidth = estimate_bandwidth(X)
    print(nb_bandwidth)

    """if nb_bandwidth < 2.0:
        ms = MeanShift(bandwidth = 2.0, n_jobs=-1)
    else:
        ms = MeanShift(bandwidth = nb_bandwidth, n_jobs=-1)"""
    ms = MeanShift(bandwidth=nb_bandwidth, n_jobs=-1)
    ms.fit(X)
    y_pred_MS = ms.predict(X)
    cluster_centers_MS = ms.cluster_centers_
    sil_max = silhouette_score(X, y_pred_MS)
    nb_cluster_algo = int(len(cluster_centers_MS))
    calin_max = calinski_harabasz_score(X, y_pred_MS)
    davies_max = davies_bouldin_score(X, y_pred_MS)

    print("MeanShift done")
    name = 'MShift'

    return y_pred_MS, nb_cluster_algo, sil_max, name, cluster_centers_MS, calin_max, davies_max


def nb_cluster_MG(X: List[float], deb: int, fin: int) -> int:
    K = range(deb, fin)
    cluster_MG_df = pd.DataFrame(columns=['n_clusters', 'silhouette_score'])
    for k in K:
        MG = GaussianMixture(n_components=k)
        MG.fit(X)
        cluster_labels_MG = MG.predict(X)
        bic = MG.bic(X)
        silhouette_avg = silhouette_score(X, cluster_labels_MG)
        print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg,
              "The BIC is :", bic)
        cluster_MG_df = cluster_MG_df.append({'n_clusters': k, 'silhouette_score': silhouette_avg, 'bic': bic},
                                             ignore_index=True)

    cluster_MG_df['bic_gradient'] = np.gradient(cluster_MG_df.bic)

    # Difference between i and i+1
    for i in range(0, (len(cluster_MG_df) - 1)):
        cluster_MG_df.loc[i + 1, 'diff_bic_gradient'] = (cluster_MG_df.loc[(i + 1), 'bic_gradient']) - (
            cluster_MG_df.loc[(i), 'bic_gradient'])

    # Absolute value and replace NaN by 0
    cluster_MG_df.diff_bic_gradient = abs(cluster_MG_df.diff_bic_gradient)
    cluster_MG_df = cluster_MG_df.fillna(0)

    # we look for the position of each different nb_cluster according to sil_score and gradient of the bic
    cluster_MG_df = cluster_MG_df.sort_values('silhouette_score')
    cluster_MG_df = cluster_MG_df.reset_index()
    cluster_MG_df['sil_pos'] = (cluster_MG_df.index + 1)

    cluster_MG_df = cluster_MG_df.sort_values('bic', ascending=False)
    cluster_MG_df = cluster_MG_df.reset_index()
    cluster_MG_df['bic_pos'] = (cluster_MG_df.index + 1)

    cluster_MG_df = cluster_MG_df.drop(columns=['level_0', 'index'])

    cluster_MG_df = cluster_MG_df.sort_values('diff_bic_gradient')
    cluster_MG_df = cluster_MG_df.reset_index()
    cluster_MG_df['diff_bic_gradient_pos'] = (cluster_MG_df.index + 1)

    cluster_MG_df = cluster_MG_df.drop(columns=['index'])

    cluster_MG_df = cluster_MG_df.sort_values('n_clusters')

    for index, row in cluster_MG_df.iterrows():
        cluster_MG_df.loc[index, 'comb_lin'] = row['sil_pos'] + row['diff_bic_gradient_pos']

    # we take n_clusters max if there is an equality of points
    a = cluster_MG_df[cluster_MG_df.comb_lin == np.max(cluster_MG_df.comb_lin)]
    comb_lin_max = a[a.n_clusters == np.max(a.n_clusters)].index.values
    n_clusters_algo = int(cluster_MG_df.loc[comb_lin_max, 'n_clusters'])
    print("Number of clusters : {}".format(int(n_clusters_algo)))

    return n_clusters_algo


def prediction_tous_les_clusters_separes_MG(pourc: float, y_pred_MG: np.ndarray, Y_score_MG: np.ndarray) -> pd.DataFrame:
    # we put in a dataframe the cluster and its probability for each point
    y_pred_MG_df = pd.DataFrame({'cluster': y_pred_MG, 'proba': -Y_score_MG})

    y_pred_MG_df[y_pred_MG_df.proba > y_pred_MG_df.groupby('cluster').proba.transform('quantile', pourc)] = 1
    y_pred_MG_df[y_pred_MG_df.proba != 1] = 0

    return y_pred_MG_df


def Mixturegaussian(X: List[float], deb: int, fin: int) -> Tuple[np.ndarray, int, float, str, np.ndarray, float, float]:
    print("Mixture Gaussian\n")

    # we look for the optimal number of clusters according to the score silhouette and the gradient of the bic
    n_clusters_algo = nb_cluster_MG(X, deb, fin)

    MG = GaussianMixture(n_components=int(n_clusters_algo))
    MG.fit(X)
    y_pred_MG = MG.predict(X)
    Y_score_MG = MG.score_samples(X)
    sil_max = silhouette_score(X, y_pred_MG)
    calin_max = calinski_harabasz_score(X, y_pred_MG)
    davies_max = davies_bouldin_score(X, y_pred_MG)

    print("Mixture Gaussian done")
    name = 'MixtureGaussian'
    return y_pred_MG, n_clusters_algo, sil_max, name, Y_score_MG, calin_max, davies_max


def nb_clusters_BMG(X: List[float]) -> pd.Series:
    K: List[float] = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    cluster_BMG_df = pd.DataFrame(columns=['weight_concentration_prior', 'silhouette_score'])

    for k in K:
        BMG = BayesianGaussianMixture(n_components=50, weight_concentration_prior=k, max_iter=200)
        BMG.fit(X)
        cluster_labels_BMG = BMG.predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels_BMG)
        print("For weight_concentration_prior =", k,
              "The average silhouette_score is :", silhouette_avg)

        cluster_BMG_df = cluster_BMG_df.append({'weight_concentration_prior': k, 'silhouette_score': silhouette_avg},
                                               ignore_index=True)

    # Display of the curve
    fig, ax = plt.subplots()
    ax.plot(cluster_BMG_df.weight_concentration_prior, cluster_BMG_df.silhouette_score, 'bx-')
    ax.set_title('silhouette')
    ax.set(xlabel='k', ylabel='silhouette_score')

    # we look for the number of clusters corresponding to the max value of silhouette score
    sil_max = cluster_BMG_df.silhouette_score.max()
    n_weight_concentration_prior = cluster_BMG_df.loc[
        cluster_BMG_df['silhouette_score'] == sil_max, 'weight_concentration_prior']

    return n_weight_concentration_prior


def prediction_tous_les_clusters_separes_BMG(pourc: float, y_labels_BGM: np.ndarray,
                                             y_score_BGM: np.ndarray) -> pd.DataFrame:
    # we put the probabilities and the cluster corresponding to each point in a dataframe
    y_pred_BGM_df_score = pd.DataFrame({'cluster': y_labels_BGM, 'proba': y_score_BGM})
    y_pred_BGM_df_score[
        y_pred_BGM_df_score.proba >= y_pred_BGM_df_score.groupby('cluster').proba.transform('quantile', pourc)] = 1
    y_pred_BGM_df_score[y_pred_BGM_df_score.proba != 1] = 0
    return y_pred_BGM_df_score


def Bayesian_mixture_gaussian(X: List[float]) -> Tuple[np.ndarray, int, float, str, np.ndarray, float, float]:
    print("Bayesian Mixture Gaussian\n")

    n_weight_concentration_prior = nb_clusters_BMG(X)

    BGM = BayesianGaussianMixture(n_components=50, weight_concentration_prior=float(n_weight_concentration_prior))
    BGM.fit(X)
    y_score_BGM = BGM.score_samples(X)
    # we determine the clusters of each transaction
    y_labels_BGM = BGM.predict(X)
    # we change the value of the labels so that they follow each other
    nouveaux_labels = 0
    unique_labels = np.unique(y_labels_BGM)
    for i in unique_labels:
        y_labels_BGM[y_labels_BGM == i] = nouveaux_labels
        nouveaux_labels += 1

    sil_max = silhouette_score(X, y_labels_BGM)
    n_cluster_algo = len(np.unique(y_labels_BGM))
    calin_max = calinski_harabasz_score(X, y_labels_BGM)
    davies_max = davies_bouldin_score(X, y_labels_BGM)

    print("Bayesian Mixture Gaussian done")
    name = 'BayesianMixtureGaussian'
    return y_labels_BGM, n_cluster_algo, sil_max, name, y_score_BGM, calin_max, davies_max


def dbscan():
    pass
