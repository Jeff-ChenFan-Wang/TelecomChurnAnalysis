# Telecom Customer Churn Prediction
Due to the interchangeability of telecommunication services, the industry sees monopolistic competition with each industry player vying hard to not only gain subscribers, but retain existing ones as well. Hence, many telecom companies constantly create and update churn prediction models to predict whether a customer would leave their services for a competitor. A telecom provider can then offer incentives to a customer with high churn probability, in so preventing them from leaving for a competitor.

View Exploratory Data Analysis Notebook: [Here](https://github.com/Jeff-ChenFan-Wang/TelecomChurnAnalysis/blob/master/NotebookPDFs/EDA.pdf). <br>
View Model Building Notebook: [Here](https://github.com/Jeff-ChenFan-Wang/TelecomChurnAnalysis/blob/master/NotebookPDFs/Model.pdf).

## Description
This project aims to predict a churn score for an U.S based telecommunication company, the data of which was originally provided by an IBM thinktank. 


## Environment
The environment in which the Exploratory Data Analysis (EDA) and Model creation and testing can be cloned for reproducibility. The steps to do so are as follows:
1. Download the Anaconda distribution 
2. Clone the repository from Github
3. Run the below command in an anaconda terminal to clone the environment from the provided environment.yml in the repository
    ```
    "conda env create -f environment.yml"
    ```


## Dataset 
- Raw data can be downloaded from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
- The zip file of the data is contained within the repository
- To clean the data into a state used for EDA and model building, activate the environment and preprocessing script with the below commands
    ```
    activate telecomChurn
    python preprocess_zip.py
    ```
