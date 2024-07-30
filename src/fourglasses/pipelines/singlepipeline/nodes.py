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
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import svm


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
    logger.info("Performing dimension reduction using Random forest classifer to keep the most informative features...")

    # Encode the label into integer form accordingly for label encoding
    data[parameters["target_column"]] = data[parameters["target_column"]].map({'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0})

    # Identify and remove highly correlated features, as they may carry redundant information.
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    data_len = len(data.columns)
    data = data.drop(columns=to_drop)

    # Train a model and use its feature importance to select the most informative features.
    X = data.drop(columns=[parameters["target_column"]])
    y = data[parameters["target_column"]]

    # Rodom Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Get the feature importances
    feature_importance = clf.feature_importances_
    # Adjust the threshold as needed
    selected_features = X.columns[feature_importance > 0.0148]
    df_reduced_model = data[selected_features]

    logger.info("")

    # The number of important selected_features may range from 15 to 25 on random because random forest is used
    logger.info(f"These {len(selected_features)} columns holds importance in the dataset: ")
    selected_features_str = ', '.join(selected_features)
    logger.info(selected_features_str)

    logger.info("")

    # The amount of columns dropped
    columns_dropped = data_len - data.shape[1]
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
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_train: pd.Series, X_val: pd.Series, y_val: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """
    logger = logging.getLogger(__name__)
    # define the non-linear SVM model
    SVM_model = svm.SVC(kernel='rbf', C=2400)

    logger.info('Training dataset using SVM rbf kernel model....')

    # Train the SVM model
    SVM_model.fit(X_train, y_train)

    logger.info("")
    logger.info('Predicting test set using trained model....')
    # Predict on the test set
    y_test_pred = SVM_model.predict(X_test)

    logger.info("")
    logger.info('Decoding predicted test labels....')

    # Create a reverse map dictionary for decoding
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}

    # Decode the predicted test labels
    y_test_pred_decoded = [reverse_mapping[label] for label in y_test_pred]
    # Decode the correct test  labels
    y_test_decoded = [reverse_mapping[label] for label in y_test]

    logger.info("")
    logger.info('Showing classification report on testing dataset....')
    logger.info("")
    class_report_test = classification_report(y_test_decoded, y_test_pred_decoded)
    logger.info(class_report_test)
    logger.info("")
    test_accuracy = (y_test_pred == y_test).sum() / len(y_test)
    logger.info("The model has an accuracy of {:.4}% on the TEST data.".format(test_accuracy * 100))
    logger.info("")

    X_valid = X_val
    y_valid = y_val

    return X_valid, y_valid, SVM_model

def report_accuracy(X_valid: pd.Series, y_valid: pd.Series, SVM_model: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    logger = logging.getLogger(__name__)
    logger.info('Predicting validation set using trained model....')
    y_val_pred = SVM_model.predict(X_valid)
    logger.info("")
    
    logger.info('Decoding predicted validaton labels....')
    # Create a reverse map dictionary for decoding
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    # Decode the predicted validation labels
    y_val_pred_decoded = [reverse_mapping[label] for label in y_val_pred]
    # Decode the correct validation labels
    y_val_decoded = [reverse_mapping[label] for label in y_valid]

    logger.info("")
    logger.info('Showing classification report on validation dataset...')
    logger.info("")
    class_report_val = classification_report(y_val_decoded, y_val_pred_decoded)
    logger.info(class_report_val)
    logger.info("")
    valid_accuracy = (y_val_pred == y_valid).sum() / len(y_valid)
    logger.info("The model has an accuracy of {:.4}% on the VALIDATION data.".format(valid_accuracy * 100))
