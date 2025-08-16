import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_cli("blastchar/telco-customer-churn", file_name="WA_Fn-UseC_-Telco-Customer-Churn.csv", path="../data/raw")