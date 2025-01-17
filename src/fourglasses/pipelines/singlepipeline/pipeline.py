from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_predictions, report_accuracy, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["emotions", "parameters"],
                outputs=["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"],
                name="split_data_node",
            ),
            node(
                func=make_predictions,
                inputs=["X_train", "X_test", "y_test", "y_train", "X_val", "y_val"],
                outputs=["X_valid", "y_valid", "SVM_model"],
                name="make_predictions",),
            node(
                func=report_accuracy,
                inputs=["X_valid", "y_valid", "SVM_model"],
                outputs=None,
                name="report_accuracy_node",
            ),
        ]
    )
