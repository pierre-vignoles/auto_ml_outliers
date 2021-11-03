![Logo](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/static/logo_spyra.png)

### Auto-ML tool specialized in detecting of outliers.  

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)  ![badge-automl](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/auto-machine-learning.svg)


## Description
This tool will allow you to save a considerable amount of time. It includes automatic data preparation, training of 10 models of ML (supervised and unsupervised), and a visualization to help you to choose and set up a model. All these steps are done specifically for the detection of outliers.

## Different models
### Supervised learning
* Linear model
* Xgboost
* LightGbm
* Random Forrest 
* Naive Bayes

### Unsupervised learning
* KMeans
* Mean Shift
* Mixture Gaussian
* Bayesian Mixture Gaussian

## Getting Started
* You need to have your dataset in a __CSV__ format.
* You have to install the python librairies : `pip install -r requirements.txt`

## Launch
Execute the following command in a terminal : `python main.py`  

Then you need to copy the following link in your Web browser :  

![Screen1](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_1.png)

## Next steps
* First, you have to upload your dataset in a CSV format :
![Screen2](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_2.png)

* Then, you have to choose the learning mode. If your dataset contains the feature to predict, click on __supervised__. Otherwise, click on __unsupervised__.
![Screen3](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_3.png)

* If you have chosen the supervised learning, you have to select the feature to predict :
![Screen4](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_4.png)

### Data preparation
* You want to see the changes made on your dataset ? Select __diagnostic__. 
* Select __automatic__ if you want to launch the data preparation.
![Screen5](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_5.png)

### Dimensionality reduction
To perform and visualize an unsupervised model, you have to reduce the dimensions. The __PCA__ is the most basic and strong one.  
If you choose to keep more than 2 dimensions after the PCA, you have to select an other dimensionality reduction to visualize your data in 2D.
The algorithms implemented are the followings :
* TSNE
* Locally Linear Embedding
* Multi-dimensional Scaling (MDS)
* Isomap  

To understand better these algorithms, __click on this [link](https://scikit-learn.org/stable/modules/manifold.html)__

![Screen6](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_6.png)

## Dashboard
### Supervised
![Screen10](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_10.png)
![Scree11](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_11.png)
![Screen12](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_12.png)
![Screen13](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_13.png)

### Unsupervised
![Screen7](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_7.png)
![Screen8](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_8.png)
![Screen9](https://github.com/pierre-vignoles/auto_ml_outliers/blob/master/img/Screen_9.png)
