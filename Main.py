import numpy as np
import pandas as pd
from data_prep_sup import cleanDataset, separation
from model_sup import logistic, randomForest, xgBoost, lightGbm, naiveBayes
from dashboard_sup import dashViz
from Check import checksPassed
from diagnostic import function_diagnostic

from model_non_sup import Kmeans, predictions_tous_les_clusters_separe_KM, Meanshift, \
    predictions_tous_les_clusters_separe_MS, Mixturegaussian, \
    prediction_tous_les_clusters_separes_MG, Bayesian_mixture_gaussian, prediction_tous_les_clusters_separes_BMG
from data_prep_non_sup import scale, pca2, pca, fig_pca, tsne, isomap, local_lin, mds
from dashboard_non_sup import dashviznsup, server

from flask import request, render_template, Response
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Launch home page
@server.route('/')
def home():
    return render_template("home.html")


# When a dataset is selected, we call the page 2 to select the kind of learning
@server.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        global dataframe
        dataframe = pd.read_csv(f)
        return render_template("page2.html")


# If the supervised learning is selected, we call the page 3 with the name of all the column of the dataframe
# Otherwise, we call the page 4
@server.route('/Page2', methods=['POST'])
def page2():
    if request.method == 'POST':
        global learning
        if request.form['learning'] == "supervised":
            learning = "supervised"
            return render_template("page3.html", column_name=dataframe.columns)
        elif request.form['learning'] == "unsupervised":
            learning = "unsupervised"
            return render_template("page4.html")
        ##Tabs
        if request.form['learning'] == "Dataset":
            return render_template("home.html")
        elif request.form['learning'] == "Type of dataset":
            return render_template("page2.html")

# If it's in supervised learning mode, the user need to choose the feature to predict
# Then the page 4 is launch
@server.route('/Page3', methods=['POST'])
def page3():
    if request.method == 'POST':
        global feature_target
        feature_target = request.form['feature_target']

        # If the check are passed, no dataprep
        if request.form['tabs'] == "Submit" and checksPassed(dataframe) == True:
            X_train, X_test, Y_train, Y_test = separation(dataframe, feature_target)
            Y_test = pd.Series(Y_test)

            lin = logistic(X_train, X_test, Y_train)
            lin_res = [Y_test, pd.Series(lin), "Linear"]

            nb = naiveBayes(X_train, X_test, Y_train)
            nb_res = [Y_test, pd.Series(nb), "Naive Bayes"]

            xgb = xgBoost(X_train, X_test, Y_train)
            xgb_res = [Y_test, pd.Series(xgb), "XGBoost"]

            rf = randomForest(X_train, X_test, Y_train, grid_search=False)
            rf_res = [Y_test, pd.Series(rf), "Random forest"]

            lgb = lightGbm(X_train, X_test, Y_train, grid_search=False)
            lgb_res = [Y_test, pd.Series(lgb), "LightGbm"]

            dash_appS = dashViz([lin_res, nb_res, xgb_res, rf_res, lgb_res])
            return dash_appS.index()

        elif request.form['tabs'] == "Submit" and checksPassed(dataframe) == False:
            return render_template("page4.html")

        ## Tabs
        if request.form['tabs'] == "Dataset":
            return render_template("home.html")
        elif request.form['tabs'] == "Type of dataset":
            return render_template("page2.html")
        elif request.form['tabs'] == "Feature to predict":
            return render_template("page3.html")


# Choose of data processing
@server.route('/Page4', methods=['POST'])
def page4():
    if request.method == 'POST':

        global dataset_cleaned
        if request.form['Processing'] == "automatic" and learning == "supervised":
            dataset_cleaned = cleanDataset(dataframe, feature_target, manual=False, supervised=True)

            X_train, X_test, Y_train, Y_test = separation(dataset_cleaned, feature_target)

            Y_test = pd.Series(Y_test)

            lin = logistic(X_train, X_test, Y_train)
            lin_res = [Y_test, pd.Series(lin), "Linear"]

            nb = naiveBayes(X_train, X_test, Y_train)
            nb_res = [Y_test, pd.Series(nb), "Naive Bayes"]

            xgb = xgBoost(X_train, X_test, Y_train)
            xgb_res = [Y_test, pd.Series(xgb), "XGBoost"]

            rf = randomForest(X_train, X_test, Y_train, grid_search=False)
            rf_res = [Y_test, pd.Series(rf), "Random forest"]

            lgb = lightGbm(X_train, X_test, Y_train, grid_search=False)
            lgb_res = [Y_test, pd.Series(lgb), "LightGbm"]

            dash_appS = dashViz([lin_res, nb_res, xgb_res, rf_res, lgb_res])

            return dash_appS.index()


        elif request.form['Processing'] == "diagnostic" and learning == "supervised":
            messages = function_diagnostic(dataframe, feature_target, supervised=True)
            return render_template("page_manuelle.html", messages=messages)


        elif request.form['Processing'] == "automatic" and learning == "unsupervised":
            dataset_cleaned = cleanDataset(dataframe, feature_target=None, manual=False, supervised=False)
            return render_template("page5.html", column_number=range(2, len(dataset_cleaned.columns)))

        elif request.form['Processing'] == "diagnostic" and learning == "unsupervised":
            messages = function_diagnostic(dataframe, feature_target=None, supervised=False)
            return render_template("page_manuelle.html", messages=messages)

        ##Tabs
        elif request.form['Processing'] == "Dataset" and (
                learning == "supervised" or learning == "unsupervised"):
            return render_template("home.html")
        elif request.form['Processing'] == "Type of dataset" and (
                learning == "supervised" or learning == "unsupervised"):
            return render_template("page2.html")
        elif request.form['Processing'] == "Feature to predict" and (learning == "supervised"):
            return render_template("page3.html", column_name=dataframe.columns)
        elif request.form['Processing'] == "Feature to predict" and (learning == "unsupervised"):
            return "impossible"
        elif request.form['Processing'] == "Data prep" and (
                learning == "supervised" or learning == "unsupervised"):
            return render_template("page4.html")


# Diagnostic
@server.route('/page_manuelle', methods=['POST'])
def function_return():
    if request.method == 'POST':
        if request.form['return'] == "yes":
            return render_template("page4.html")

        ## Tabs
        elif request.form['return'] == "Dataset" and (learning == "supervised" or learning == "unsupervised"):
            return render_template("home.html")
        elif request.form['return'] == "Type of dataset" and (
                learning == "supervised" or learning == "unsupervise"):
            return render_template("page2.html")
        elif request.form['return'] == "Feature to predict" and (learning == "supervised"):
            return render_template("page3.html", column_name=dataframe.columns)
        elif request.form['return'] == "Feature to predict" and (learning == "unsupervised"):
            return "impossible"
        elif request.form['return'] == "Data prep" and (
                learning == "supervised" or learning == "unsupervised"):
            return render_template("page4.html")
        elif request.form['return'] == "Diagnostic" and learning == "unsupervised":
            messages = function_diagnostic(dataframe, feature_target=None, supervised=False)
            return render_template("page_manuelle.html", messages=messages)
        elif request.form['return'] == "Diagnostic" and learning == "unsupervised":
            messages = function_diagnostic(dataframe, feature_target, supervised=True)
            return render_template("page_manuelle.html", messages=messages)


# image pca
@server.route('/plot.png')
def plot_png():
    global X_scaled
    X_scaled = scale(dataset_cleaned)
    fig = fig_pca(X_scaled)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# We train the differents models and we display the dashboard if the number of dimensions chosen by the user is equal to 2.
# Otherwise, we go to the next page which will ask for the size reduction method
@server.route('/Page5', methods=['POST'])
def page5():
    if request.method == 'POST':
        if request.form['tabs'] == "Submit":
            global nb_pca, X_pca, X2, tab2
            nb_pca = request.form['nb_pca']
            nb_pca = int(nb_pca)
            X_pca = pca(X_scaled, nb_pca)
            X2 = dataset_cleaned
            pourcentage = np.arange(0.9, 1.0, 0.01)
            deb = 2
            fin = 8
            tab2 = []

            # Training of the models
            Km = Kmeans(X_pca, deb, fin)
            Km = list(Km)
            Ms = Meanshift(X_pca)
            Ms = list(Ms)
            Mg = Mixturegaussian(X_pca, deb, fin)
            Mg = list(Mg)
            Bmg = Bayesian_mixture_gaussian(X_pca)
            Bmg = list(Bmg)

            # Detection of outliers for every percentage
            tab1 = []
            for i in pourcentage:
                # KMeans
                Km2 = Km.copy()
                Km_predict_anomaly = predictions_tous_les_clusters_separe_KM(X_pca, i, Km[4], Km[0])
                Km2.insert(7, Km_predict_anomaly)
                tab1.append(Km2)
            tab2.append(tab1)

            tab1 = []
            for i in pourcentage:
                # MShift
                Ms2 = Ms.copy()
                Ms_predict_anomaly = predictions_tous_les_clusters_separe_MS(X_pca, i, Ms[4], Ms[0])
                Ms2.insert(7, Ms_predict_anomaly)
                tab1.append(Ms2)
            tab2.append(tab1)

            tab1 = []
            for i in pourcentage:
                # MixtureGaussian
                Mg2 = Mg.copy()
                Mg_predict_anomaly = prediction_tous_les_clusters_separes_MG(i, Mg[0], Mg[4])
                Mg2.insert(7, Mg_predict_anomaly)
                tab1.append(Mg2)
            tab2.append(tab1)

            tab1 = []
            for i in pourcentage:
                # BayesianMixtureGaussian
                Bmg2 = Bmg.copy()
                Bmg_predict_anomaly = prediction_tous_les_clusters_separes_BMG(i, Bmg[0], Bmg[4])
                Bmg2.insert(7, Bmg_predict_anomaly)
                tab1.append(Bmg2)
            tab2.append(tab1)

            if nb_pca == 2:
                dashboard_non_sup = dashviznsup(X_pca, X2, tab2)
                return dashboard_non_sup.index()

            elif nb_pca != 2:
                return render_template("page6.html", reduc_name=['tsne', 'isomap', 'local lin', 'pca', 'mds'])

        elif request.form['tabs'] == "Dataset":
            return render_template("home.html")
        elif request.form['tabs'] == "Type of dataset":
            return render_template("page2.html")
        elif request.form['tabs'] == "Feature to predict":
            return "impossible"
        elif request.form['tabs'] == "Data prep":
            return render_template("page4.html")
        elif request.form['tabs'] == "PCA":
            return render_template("page5.html", column_number=range(2, len(dataset_cleaned.columns)))

# Choice of the 2D dimension reduction. Then we display the dashboard
@server.route('/Page6', methods=['POST'])
def choix_reduc():
    if request.method == 'POST':
        global type_of_dimension_reduction
        type_of_dimension_reduction = request.form['type_of_dimension_reduction']

        if request.form['tabs'] == "Submit":
            if type_of_dimension_reduction == "tsne":
                X = tsne(X_pca)
            if type_of_dimension_reduction == "isomap":
                X = isomap(X_pca)
            if type_of_dimension_reduction == "local lin":
                X = local_lin(X_pca)
            if type_of_dimension_reduction == "pca":
                X = pca2(X_pca)
            if type_of_dimension_reduction == "mds":
                X = mds(X_pca)
            dashboard_non_sup = dashviznsup(X, X2, tab2)
            return dashboard_non_sup.index()

        elif request.form['tabs'] == "Dataset":
            return render_template("home.html")
        elif request.form['tabs'] == "Type of dataset":
            return render_template("page2.html")
        elif request.form['tabs'] == "Feature to predict":
            return "impossible"
        elif request.form['tabs'] == "Data prep":
            return render_template("page4.html")
        elif request.form['tabs'] == "PCA":
            return render_template("page5.html", column_number=range(2, len(dataset_cleaned.columns)))
        elif request.form['tabs'] == "Other reduction":
            return render_template("page6.html", reduc_name=['tsne', 'isomap', 'local lin', 'pca', 'mds'])


if __name__ == '__main__':
    server.run()
