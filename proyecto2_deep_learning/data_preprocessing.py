import pandas as pd


def preprocess_real_data(train, test, target):
    """Preprocesses the real training and testing datasets by selecting relevant features and encoding categorical variables.

    Args:
        train (pd.DataFrame): The real training dataset.
        test (pd.DataFrame): The real testing dataset.
        target (str): The name of the target variable.
    Returns:
        tuple: A tuple containing the preprocessed training features, training target, testing features, and testing target.
    """
    train['Outstanding_Debt'] = train['Outstanding_Debt'] / 1000
    test['Outstanding_Debt'] = test['Outstanding_Debt'] / 1000

    cols = [
        'Num_Credit_Card',
        'Changed_Credit_Limit',
        'Delay_from_due_date',
        'Interest_Rate',
        'Credit_Mix',
        'Outstanding_Debt',
        target
    ]

    train = train[cols]
    test = test[cols]

    train = pd.get_dummies(train, columns=['Credit_Mix'], drop_first=True)
    test = pd.get_dummies(test, columns=['Credit_Mix'], drop_first=True)

    X_real_train = train.drop(columns=[target])
    y_real_train = train[target]

    X_real_test = test.drop(columns=[target])
    y_real_test = test[target]

    return X_real_train, y_real_train, X_real_test, y_real_test


def preprocess_synthetic_data(synthetic_data, target):
    """Preprocesses the synthetic dataset by selecting relevant features and encoding categorical variables.

    Args:
        synthetic_data (pd.DataFrame): The synthetic dataset to preprocess.
        target (str): The name of the target variable.
    Returns:
        tuple: A tuple containing the preprocessed synthetic features and synthetic target.
    """
    synthetic_data['Outstanding_Debt'] = synthetic_data['Outstanding_Debt'] / 1000

    synthetic_data = synthetic_data[[
        'Num_Credit_Card',
        'Changed_Credit_Limit',
        'Delay_from_due_date',
        'Interest_Rate',
        'Credit_Mix',
        'Outstanding_Debt',
        target
    ]]

    synthetic_data = pd.get_dummies(synthetic_data, columns=['Credit_Mix'], drop_first=True)

    X_synthetic_train = synthetic_data.drop(columns=[target])
    y_synthetic_train = synthetic_data[target]

    return X_synthetic_train, y_synthetic_train