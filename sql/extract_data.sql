CREATE DATABASE Telco_Customer_Churn;

CREATE TABLE churn_customer (
    gender varchar(255),
    SeniorCitizen int,
    Partner varchar(255),
    Dependents varchar(255),
    tenure int,
    PhoneService varchar(255),
    MultipleLines varchar(255),
    InternetService varchar(255),
    OnlineSecurity varchar(255),
    OnlineBackup varchar(255),
    DeviceProtection varchar(255),
    TechSupport varchar(255),
    StreamingTV varchar(255),
    StreamingMovies varchar(255),
    Contract varchar(255),
    PaperlessBilling varchar(255),
    PaymentMethod varchar(255),
    MonthlyCharges float,
    TotalCharges float,
    Churn varchar(255)
)

SELECT * FROM churn_customer