def checksPassed(dataset):
    for column in dataset.columns:
        # If there are na values, the dataset must be processing
        if dataset.isna().sum()[column] != 0:
            print("Presence of na values")
            return False
        # If this is a float, this is a numeric value
        if dataset.dtypes[column] != 'float64':
            # If this is a integer, the risk is that it can be an ID
            if dataset.dtypes[column] == 'int64':
                # If this is a boolean (probably a One Hot Encoding), there is no problem
                if dataset[column].nunique() != 2:
                    # Look in the feature's name to see if it contains hints or if each value is unique
                    id_words = ['id', 'type', 'name', 'nom']
                    if True in [word.capitalize() in column or "_" + word in column for word in id_words] or dataset[
                        column].nunique() == len(dataset):
                        return False
                    # If it is a categorical feature, many values are repeated. So if a feature has 10 times the same
                    # value, it is probably a categorical feature.
                    alpha = 10
                    if (dataset[column].nunique()) < (len(dataset) / alpha):
                        print("The feature", column,
                              "seems to be categorical and numeric because it has only the following values : ",
                              dataset[column].nunique())
                        return False
            elif dataset.dtypes[column] == 'object':
                # Look in the feature's name to see if it contains hints or if each value is unique
                numeric_words = ['quantity', "quantite", 'volume', 'number', 'nombre', "day", "jour", "month", "mois",
                                 "year", "annee"]
                if True in [word.capitalize() in column or "_" + word in column for word in numeric_words]:
                    print("The feature ", column, " which is a string, seems to be ordinal")
                    return False
                column_values = dataset[dataset[column].isna() == False][column]
                string = column_values.unique()
                for i in range(len(string)):
                    numbers = [int(s) for s in string[i].split() if s.isdigit()]
                    if numbers:
                        print("The feature ", column, " contains numbers (shape of string)")
                        return False

    return True
