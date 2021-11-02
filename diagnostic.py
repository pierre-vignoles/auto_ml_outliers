from typing import List, Union
import pandas as pd


def function_diagnostic(dataset: pd.DataFrame, feature_target: Union[str, None], supervised: bool) -> List[str]:
    if supervised == True:
        dataframe: pd.DataFrame = dataset.drop(columns=[feature_target])
    else:
        dataframe = dataset

    messages: List[str] = []

    # Sometimes, a useless index, called 'Unnamed: 0' can be added
    if 'Unnamed: 0' in dataset.columns:
        message = "Column 'Unnamed: 0' has been found => Suppression"
        messages.append(message)
        del message

    # Outliers can be na values
    for column in dataset.columns:

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
                for outlier in outliers:
                    message = "Search of na values.\n '" + str(
                        outlier) + "' is not a logical value for the feature '" + column + "' whose values are :" + str(
                        sorted(dataframe[column].unique())) + "'"
                    messages.append(message)
                    del message

        elif dataset[column].dtypes == 'O':
            values = dataset[column].unique()
            outliers = []
            for string in values:
                if string.isdigit():
                    outliers.append(string)
            # If there is only 1 number, it can be an outlier
            if len(outliers) == 1:
                message = "Search of na values.\n '" + outliers[
                    0] + "' is not a logical value for the feature '" + column + "' whose values are :'" + "'".join(
                    [dataframe[column].unique()[string] + "', " for string in
                     range(len(dataframe[column].unique()) - 1)]) + " and '" + dataframe[column].unique()[-1] + "'"
                messages.append(message)
                del message

    for column in dataframe.columns:

        # Keep only the values that are not equal to nan
        column_values = dataframe[dataframe[column].isna() == False][column]
        # Binary variables: one of the two values is replaced by 1 and the other by zero
        if dataframe[column].nunique() == 2:
            message = "Booleans values for the columns '" + column + "' will be replaced by 0 and 1"
            messages.append(message)
            del message
        # For columns in numeric forms, some can be id
        elif dataframe.dtypes[column] == 'int64' or dataframe.dtypes[column] == 'float64':
            # Unique ID : delete it
            if dataframe.dtypes[column] == 'int64' and dataframe[column].nunique() == len(dataframe):
                message = "The feature '" + column + "' is an ID "
                messages.append(message)
                del message
            # Nominal : One hot encoding
        else:
            message = "The column '" + column + "', with values '" + ''.join(
                [column_values.unique()[string] + ", " for string in range(len(column_values.unique()) - 1)] + [
                    "and " + column_values.unique()[-1]]) + "' will be transform with One Hot Encoding method"
            messages.append(message)
            del message

    #### Correlation study
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
                    message = "The column '" + correlated + "' is correlated more than 90% with '" + column + "' \n Suppression of '" + correlated + "' because it has na values. "
                    messages.append(message)
                    del message
                else:
                    message = "The column '" + column + "' is correlated more than 90% with '" + correlated + "' \n Suppression of '" + column + "' because it has na values. "
                    messages.append(message)
                    del message

    # Missing values :
    ## Suppression of the lines if they match for less than 1% of the dataset.
    ## Otherwise, suppression of the feature
    for column in dataframe.columns:
        if dataframe.isna().sum()[column] != 0:
            if dataframe.isna().sum()[column] / len(dataframe) < 0.1:
                message = "The column '" + column + "' has missing values which match less than 10% of the colmun. \n Suppression of the lines."
                messages.append(message)
                del message
            else:
                message = "The column '" + column + "' has missing values which match more than 10% of the colmun. \n Suppression of the colmun"
                messages.append(message)
                del message

    return messages
