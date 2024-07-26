"""
This is a boilerplate pipeline
generated using Kedro 0.18.7
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing dimension reduction using Random forest classifer to keep the most imformative features...")

    # Encode the label into integer form accordingly for label encoding
    data[parameters["target_column"]] = data[parameters["target_column"]].map({'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0})

    # Train a model and use its feature importance to select the most informative features.
    X = data.drop(columns=[parameters["target_column"]])
    y = data[parameters["target_column"]]
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Get the feature importances
    feature_importance = clf.feature_importances_
    # Adjust the threshold as needed
    selected_features = X.columns[feature_importance > 0.01]  
    df_reduced_model = data[selected_features]

    logger.info("")
    # The number of important selected_features may range from 15 to 25 on random because random forest is used
    logger.info(f"These {len(selected_features)} columns holds importance in the dataset: ")
    selected_features_str = ', '.join(selected_features)
    logger.info(selected_features_str)
    logger.info("")
    # The amount of columns dropped
    columns_dropped = data.shape[1] - len(selected_features)
    logger.info(f"The amount of columns dropped: {columns_dropped}")
    logger.info("")

    # Ensure df_reduced_model is a copy of the DataFrame to avoid SettingWithCopyWarning
    df_reduced_model = df_reduced_model.copy()
    # make the variable y as the label feature
    df_reduced_model.loc[:, parameters["target_column"]] = y

    # Split the data into 80% training and 20% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        df_reduced_model.drop(columns=[parameters["target_column"]]),
        df_reduced_model[parameters["target_column"]],
        train_size=parameters["train_fraction"],
        random_state=parameters["random_state"],
        stratify=df_reduced_model[parameters["target_column"]]
    )

    # Train a model on the training set
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_temp)

    # Evaluate accuracy after dimension reduction
    accuracy = accuracy_score(y_temp, y_pred)
    logger.info("Remaining dataset accuracy after dimension reduction: {:.4}%".format(accuracy * 100))
    logger.info("")

    # Split the temporary set into 50% testing and 50% validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, 
        y_temp,
        test_size=parameters["test_fraction"],
        random_state=parameters["random_state"],
        stratify=y_temp
    )

    return X_train, y_train, X_test, y_test, X_val, y_val


def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """

    X_train_numpy = X_train.to_numpy()
    X_test_numpy = X_test.to_numpy()

    squared_distances = np.sum(
        (X_train_numpy[:, None, :] - X_test_numpy[None, :, :]) ** 2, axis=-1
    )
    nearest_neighbour = squared_distances.argmin(axis=0)
    y_pred = y_train.iloc[nearest_neighbour]
    y_pred.index = X_test.index

    return y_pred


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = (y_pred == y_test).sum() / len(y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
