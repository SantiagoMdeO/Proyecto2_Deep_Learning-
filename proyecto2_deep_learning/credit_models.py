from sklearn.linear_model import LogisticRegression
import pandas as pd


def compute_credit_score(
        coef_dict,
        num_credit_card, changed_credit_limit, delay_from_due_date, interest_rate, 
        outstanding_debt, credit_mix_good, credit_mix_standard
):
    """Computes a credit score based on the logistic regression coefficients and input features.

    Args:
        coef_dict (dict): A dictionary containing the logistic regression coefficients.
        num_credit_card (float): The number of credit cards.
        changed_credit_limit (float): The change in credit limit.
        delay_from_due_date (float): The delay from the due date.
        interest_rate (float): The interest rate.
        outstanding_debt (float): The outstanding debt.
        credit_mix_good (float): The proportion of good credit mix.
        credit_mix_standard (float): The proportion of standard credit mix.
    Returns:
        float: The computed credit score.
    """
    score = (
        coef_dict['Num_Credit_Card'] * num_credit_card +
        coef_dict['Changed_Credit_Limit'] * changed_credit_limit +
        coef_dict['Delay_from_due_date'] * delay_from_due_date +
        coef_dict['Interest_Rate'] * interest_rate +
        coef_dict['Outstanding_Debt'] * outstanding_debt +
        coef_dict['Credit_Mix_Good'] * credit_mix_good +
        coef_dict['Credit_Mix_Standard'] * credit_mix_standard
    )
    return score


def real_data_credit_model(X_train, y_train, X_test):
    """
    Trains a logistic regression model on the real training data and evaluates it on the real testing data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_test (pd.DataFrame): The testing features.
    Returns:
        tuple: A tuple containing the computed credit scores and classifications for the testing data.
    """
    model = LogisticRegression(
        max_iter=1000, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    coefficients = model.coef_[0]
    coef_dict = dict(zip(X_train.columns, coefficients))

    score = pd.Series([
        compute_credit_score(
            coef_dict,
            row['Num_Credit_Card'], row['Changed_Credit_Limit'], row['Delay_from_due_date'], 
            row['Interest_Rate'], row['Outstanding_Debt'], row['Credit_Mix_Good'], row['Credit_Mix_Standard']
        ) for _, row in X_test.iterrows()
    ])
    classification = model.predict(X_test)

    return score, classification


def synthetic_data_credit_model(X_train, y_train, X_test):
    """Trains a logistic regression model on the synthetic training data and evaluates it on the real testing data.

    Args:
        X_train (pd.DataFrame): The synthetic training features.
        y_train (pd.Series): The synthetic training target variable.
        X_test (pd.DataFrame): The real testing features.
    Returns:
        tuple: A tuple containing the computed credit scores and classifications for the testing data.
    """
    model = LogisticRegression(
        max_iter=1_000, 
    )
    model.fit(X_train, y_train)

    coefficients = model.coef_[0]
    coef_dict = dict(zip(X_train.columns, coefficients))

    score = pd.Series([
        compute_credit_score(
            coef_dict,
            row['Num_Credit_Card'], row['Changed_Credit_Limit'], row['Delay_from_due_date'], 
            row['Interest_Rate'], row['Outstanding_Debt'], row['Credit_Mix_Good'], row['Credit_Mix_Standard']
        ) for _, row in X_test.iterrows()
    ])
    classification = model.predict(X_test)

    return score, classification