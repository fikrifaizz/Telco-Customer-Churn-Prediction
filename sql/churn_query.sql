SELECT *
FROM public.churn_customer;

SELECT churn_customer.churn, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.churn;

SELECT churn_customer.gender, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.gender
ORDER BY churn_count DESC;

SELECT churn_customer.onlinebackup, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.onlinebackup
ORDER BY churn_count DESC;

SELECT churn_customer.onlinesecurity, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.onlinesecurity
ORDER BY churn_count DESC;

SELECT churn_customer.paperlessbilling, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.paperlessbilling
ORDER BY churn_count DESC;

SELECT churn_customer.onlinebackup, churn_customer.onlinesecurity, churn_customer.paperlessbilling, churn_customer.gender, COUNT(*) as churn_count
FROM public.churn_customer
GROUP BY churn_customer.onlinebackup, churn_customer.onlinesecurity, churn_customer.paperlessbilling, churn_customer.gender
ORDER BY churn_count DESC;
