from typing import List

import numpy as np
import pandas as pd
from tkinter import messagebox
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from tkinter import Tk


def cleanDataset(dataset: pd.DataFrame, feature_target: str, manual: bool, supervised: bool) -> pd.DataFrame:
    if supervised == True:
        dataframe = dataset.drop(columns=[feature_target])
    else:
        dataframe = dataset

    # Sometimes, a useless index, called 'Unnamed: 0' can be added
    if 'Unnamed: 0' in dataframe.columns:
        dataframe = dataframe.drop('Unnamed: 0', axis=1)

    # Outliers can be na values
    for column in dataframe.columns:
        root = Tk()
        root.withdraw()

        if dataset[column].dtypes == 'int64' or dataset[column].dtypes == 'float64':
            q2 = dataset[column].quantile(0.75)
            q1 = dataset[column].quantile(0.25)
            iqr = q2 - q1
            upper = q2 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            while upper in dataset[column].unique():
                upper += 1
            while lower in dataset[column].unique():
                lower -= 1
            outliers = dataset[(dataset[column] < lower) | (dataset[column] > upper)][column].unique()
            if len(outliers) > 0:
                if manual == True:
                    for outlier in outliers:
                        if not messagebox.askyesno("Search of na values.",
                                                   "Is '" + str(
                                                       outlier) + "' a logical value for the feature '" + column + "' whose values are :" + str(
                                                       sorted(dataframe[column].unique())) + " ?"):
                            dataframe[column] = dataframe[column].apply(lambda row: np.nan if (row == outlier) else row)
                else:
                    for outlier in outliers:
                        dataframe[column] = dataframe[column].apply(lambda row: np.nan if (row == outlier) else row)
        elif dataset[column].dtypes == 'O':
            values = dataset[column].unique()
            outliers = []
            for string in values:
                if string.isdigit():
                    outliers.append(string)
            # if there is only 1 number, it can be an outlier
            if len(outliers) == 1:
                if manual == True:
                    if not messagebox.askyesno("Search of na values.",
                                               "Is '" + outliers[
                                                   0] + "' a logical value for the feature '" + column + "' whose values are :'" +
                                               "'".join([dataframe[column].unique()[string] + "', " for string in
                                                         range(len(dataframe[column].unique()) - 1)]) + "et '" +
                                               dataframe[column].unique()[-1] +
                                               "' ?"):
                        dataframe[column] = dataframe[column].apply(lambda row: np.nan if (row == '0') else row)
                else:
                    dataframe[column] = dataframe[column].apply(lambda row: np.nan if (row == '0') else row)
        root.destroy()

    # Numeric conversions for each column
    avancement = 0
    print('Step =' + str(avancement) + '/' + str(dataset.shape[1]))
    for column in dataframe.columns:
        root = Tk()
        root.withdraw()

        # Keep only the values that are not equal to nan
        column_values = dataframe[dataframe[column].isna() == False][column]
        # Binary variables: one of the two values is replaced by 1 and the other by zero
        if dataframe[column].nunique() == 2:
            dataframe[column] = dataframe[column].apply(lambda row: 1 if row == dataframe[column].unique()[0] else 0)
            # For columns in numeric forms, some can be id
        elif dataframe.dtypes[column] == 'int64' or dataframe.dtypes[column] == 'float64':
            # Unique ID : delete it
            if dataframe.dtypes[column] == 'int64' and dataframe[column].nunique() == len(dataframe):
                dataframe = dataframe.drop(column, axis=1)
            elif manual == True:
                # Nominal : One hot encoding
                if messagebox.askyesno("Is this feature an ID ?",
                                       "Is the feature '" + column + "' whose values are : " + str(
                                           sorted(dataframe[column].unique())) +
                                       " an ID with the values have no real meaning ?"):
                    for nominal in dataframe[column].unique():
                        dataframe["One Hot Encoding " + column + ", id n° " + str(nominal)] = dataframe[column].apply(
                            lambda row: 1 if row == nominal else 0)
                    dataframe = dataframe.drop(column, axis=1)
                # Otherwise, it is a numeric value
                else:
                    pass
        # Categorical feature : Doing a one hot encoding on nominal values and ordinal are ordered by ascending order
        elif manual == False or messagebox.askyesno("Is this feature a nominal or ordinal value ?",
                                                    column + "' is a feature which can be ordered ?'" +
                                                    ''.join([column_values.unique()[string] + ", " for string in
                                                             range(len(column_values.unique()) - 1)] + [
                                                                "and " + column_values.unique()[-1]])):
            # Ordinal
            string = column_values.unique()
            score = []
            for i in range(len(string)):
                numbers = [int(s) for s in string[i].split() if s.isdigit()]
                if numbers != []:
                    score.append(np.mean(numbers))
                else:
                    score.append(0)
            arg_sorted = np.argsort(score)
            string_sorted = string[arg_sorted]
            dataframe[column] = dataframe[column].apply(
                lambda row: row if type(row) == float and np.isnan(row) else np.where(string_sorted == row)[0][0])

        else:
            # Nominal : One hot encoding
            for nominal in dataframe[column].unique():
                dataframe["One Hot Encoding " + column + " : " + nominal] = dataframe[column].apply(
                    lambda row: 1 if row == nominal else 0)
            dataframe = dataframe.drop(column, axis=1)

        avancement += 1
        print('Step =' + str(avancement) + '/' + str(dataset.shape[1]))
        root.destroy()

    # Correlation study
    correlation_matrix = dataframe.corr()
    # Add in lines, the feature's name
    correlation_matrix = correlation_matrix.set_index(pd.Series(correlation_matrix.columns))

    dataframe.to_csv('test.csv', index=False)
    correlation_matrix.to_csv('corr.csv', index=False)

    for column in correlation_matrix.columns:
        # correlated_columns is the set of features correlated more than 90% to the target column
        correlated_columns = correlation_matrix[
            ((correlation_matrix[column] < -0.9) | (correlation_matrix[column] > 0.9)) & (
                        correlation_matrix[column].index != column)].index
        for correlated in correlated_columns:
            # If both of the columns are in the same One Hot Encoding of a feature, nothing is done.
            # Otherwise, the column with the more na values is deleted
            if "One Hot Encoding " not in column or column.rsplit(':')[0] != correlated.rsplit(':')[0]:
                if dataframe.isna().sum()[column] <= dataframe.isna().sum()[correlated]:
                    dataframe = dataframe.drop(correlated, axis=1)
                    correlation_matrix[correlated] = 0
                    correlation_matrix.loc[correlated] = 0

                else:
                    dataframe = dataframe.drop(column, axis=1)
                    correlation_matrix[column] = 0
                    correlation_matrix.loc[column] = 0
                    correlated_columns = []

            # This value and the symetric one in the correlation matrix are put to 0 to avoid to treat them 2 times.
            correlation_matrix[column][correlated] = 0
            correlation_matrix[correlated][column] = 0



    for column in dataframe.columns:
        if dataframe.isna().sum()[column] != 0:
            if dataframe.isna().sum()[column] / len(dataframe) < 0.1:
                dataframe = dataframe.drop(dataframe[dataframe.isna()[column]].index, axis=0)
            else:
                dataframe = dataframe.drop(column, axis=1)

    if supervised == True:
        dataframe[feature_target] = dataset[feature_target]

    return dataframe


# Split train and test (80/20)
def separation(dataset: pd.DataFrame, feature_target: str) -> List[List[float]]:
    X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(feature_target, axis=1),
                                                        dataset[feature_target].to_numpy(), test_size=0.2,
                                                        random_state=0)
    return [X_train, X_test, Y_train, Y_test]


def normalizeData(X_train: List[float], X_test: List[float]) -> List[List[float]]:
    X_normalized = normalize(np.concatenate((X_train, X_test), axis=0), axis=0)
    X_normalized_train = X_normalized[:len(X_train), :]
    X_normalized_test = X_normalized[len(X_train):, :]
    return [X_normalized_train, X_normalized_test]


def standardizeData(X_train: List[float], X_test: List[float]) -> List[List[float]]:
    X = np.concatenate((X_train, X_test), axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)
    X_standardized_train = X_standardized[:len(X_train), :]
    X_standardized_test = X_standardized[len(X_train):, :]
    return [X_standardized_train, X_standardized_test]
