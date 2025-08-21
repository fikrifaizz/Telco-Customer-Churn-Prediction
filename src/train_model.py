import mlflow
import optuna
import time
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-training")
data_train = pd.read_csv("../data/processed/train.csv")
data_test = pd.read_csv("../data/processed/test.csv")
xtrain = data_train.drop('Churn', axis=1)
ytrain = data_train['Churn']
xtest = data_test.drop('Churn', axis=1)
ytest = data_test['Churn']

mlflow.sklearn.autolog(log_models=False)
with mlflow.start_run():
    start = time.time()
    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "class_weight": "balanced"
        }
        model = LGBMClassifier(**param)
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        accuracy = accuracy_score(ytest, preds)
        f1_scores = f1_score(ytest, preds)
        precision_scores = precision_score(ytest, preds)
        recall_scores = recall_score(ytest, preds)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("f1_scores", f1_scores)
        trial.set_user_attr("precision_scores", precision_scores)
        trial.set_user_attr("recall_scores", recall_scores)
        trial.set_user_attr("model", model)
        return accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    end = time.time()
    elapsed = end - start
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)
    mlflow.log_metric("best_f1_score", study.best_trial.user_attrs["f1_scores"])
    mlflow.log_metric("best_precision_score", study.best_trial.user_attrs["precision_scores"])
    mlflow.log_metric("best_recall_score", study.best_trial.user_attrs["recall_scores"])
    mlflow.log_metric("training_time_sec", elapsed)
    best_model = study.best_trial.user_attrs["model"]
    mlflow.sklearn.log_model(best_model, artifact_path="model")