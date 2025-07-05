import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

def log_parameters(params):
    """Log model parameters"""
    for key, value in params.items():
        # Convert to string
        if isinstance(value, (list, tuple, dict)):
            value = str(value)
        mlflow.log_param(key, value)

    
def log_metrics(y_true, y_pred, y_proba=None, train_time=None):
    """Log classification metrics"""
    mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
    mlflow.log_metric("precision_weighted", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    mlflow.log_metric("recall", recall_score(y_true, y_pred, zero_division=0))
    mlflow.log_metric("f1_score", f1_score(y_true, y_pred, zero_division=0))

    # ROC AUC if probabilities provided
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
            mlflow.log_metric("roc_auc", auc_roc)
        except:
            pass  # In case it's not binary or fails

    # Train time
    if train_time is not None:
        mlflow.log_metric("train_time", train_time)

    # Support 
    report = classification_report(y_true, y_pred, output_dict=True)
    mlflow.log_metric("support_neg", report["0"]["support"])
    mlflow.log_metric("support_pos", report["1"]["support"])
    
    
def log_model(model, model_name="model"):
    """Log model"""
    mlflow.sklearn.log_model(model, model_name)


def track_model(
    experiment_name: str,
    run_name: str,
    model_pipeline,
    results_dict: dict,
    params: dict
):
    """Track a model using a complete results dictionary"""
    try:
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            log_parameters(params)
            
            # Extract required fields
            y_true = results_dict.get("y_true")
            y_pred = results_dict.get("y_pred") 
            y_proba = results_dict.get("y_proba")
            train_time = results_dict.get("train_time")
            
            if y_true is None or y_pred is None:
                raise ValueError("results_dict must contain 'y_true' and 'y_pred'")
            
            # Log metrics
            log_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, train_time=train_time)
            
            # Log model
            log_model(model_pipeline)
            
            print(f"{run_name} tracked successfully in MLflow")
                
    except Exception as e:
        print(f"Error tracking {run_name}: {e}")