import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
import seaborn as sns

model = joblib.load("../models/model.pkl")
data_test = pd.read_csv("../data/processed/test.csv")
data_train = pd.read_csv("../data/processed/train.csv")
best_param = model.best_params
print(f"Best parameters: {best_param}")
ytest = data_test['Churn']
xtest = data_test.drop('Churn', axis=1)
ytrain = data_train['Churn']
xtrain = data_train.drop('Churn', axis=1)
lgbmclf = LGBMClassifier(**best_param)
lgbmclf.fit(xtrain, ytrain)
preds = lgbmclf.predict(xtest)
accuracy = accuracy_score(ytest, preds)
print(f"Accuracy: {accuracy}")
plt.figure(figsize=(10, 6))
plt.title("Confusion Matrix")
sns.heatmap(confusion_matrix(ytest, preds), annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../docs/eval_report/confusion_matrix.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
importances = pd.Series(lgbmclf.feature_importances_, index=xtest.columns).sort_values()
importances.plot(kind="barh")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../docs/eval_report/feature_importance.png")
plt.show()

feature_order = list(xtrain.columns)
print("\nUrutan fitur (index: nama):")
for i, col in enumerate(feature_order):
    print(f"{i:>3}: {col}")

data_baru = [20, 0, 1, 2, 2, 60.2, 700.89]
x_new = pd.DataFrame([data_baru], columns=feature_order)
pred_new = lgbmclf.predict(x_new)[0]
probas_new = lgbmclf.predict_proba(x_new)[0]
if pred_new == 1:
    pred_new = "Churn"
else:
    pred_new = "No Churn"
print(f"Prediksi untuk data baru: {pred_new}")
print(f"Probabilitas untuk data baru: {probas_new}")
