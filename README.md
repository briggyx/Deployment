# Deloyment Project 

### [Assignment](assignment.md)

## Project/Goals

**The goal** of this project is to use machine learning to automate the process of determining loan eligibility based on customer information collected through online application forms. The dataset contains details pertaining to the customer's demographic and socioeconomic attributes. 

**The aim** is to predict whether a loan will be approved or not, based on the provided customer information. 

The dataset, which includes variables such as Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status, can be found [here](https://drive.google.com/file/d/1h_jl9xqqqHflI5PsuiQd_soNYxzFfjKw/view?usp=sharing).

## Hypothesis

The initial phase of data analysis involves generating a hypothesis, which requires a clear understanding of the problem and the development of a relevant hypothesis that could positively impact the outcome. 
It is important to create this hypothesis prior to deeply examining the data, so that we can create a list of various analyses that may be performed once data is available.

Brief research on factors involved in determining the approval of loans yields to the following key points:
- Lenders consider **credit scores and reports** as they indicate how well individuals manage borrowed money and a poor credit history raises the risk of default. [Source](https://www.fool.com/the-ascent/personal-loans/articles/7-factors-lenders-look-considering-your-loan-application/)
- To ensure that borrowers can repay their loans, lenders assess their **income level and stability**, with higher income requirements for larger loans. In addition, steady employment is necessary, making it more difficult for self-employed individuals or those who only work part of the year to obtain a loan. [Source](https://www.fool.com/the-ascent/personal-loans/articles/7-factors-lenders-look-considering-your-loan-application/)
- Lenders assess a borrower's ability to make payments on current and new debt by calculating their **Debt-to-Income (DTI) ratio**, which is the percentage of their monthly gross income that goes towards debt service.[Source](https://www.forbes.com/advisor/personal-loans/personal-loan-requirements/)
- When applying for a secured personal loan, lenders typically require **collateral**, such as assets related to the loan's purpose or other valuable items like investments or collectibles, which they can seize in case of default or missed payments to recover the remaining loan amount. [Source](https://www.forbes.com/advisor/personal-loans/personal-loan-requirements/)
- Lenders evaluate loan applications by taking into account **economic conditions**, industry trends, and upcoming legislation that affect the borrower's business. [Source](https://www.bankofamerica.com/smallbusiness/resources/post/factors-that-impact-loan-decisions-and-how-to-increase-your-approval-odds/)

Based on prior research on the domain and the dataset's attributes, several hypothesis for predicting loan approval for applicants:
- High income relative to loan
- Presence of coapplicant (with relatively high income)
- Married
- Low number of dependents 
- Graduate
- Not self-employed
- Urban/semi-urban property
- Credit history present 

It may be worth noting that the dataset *does not specify* the type of loan that's requested for, such as personal,  mortgage, home-equity, etc. Other possiblity important information that's missing includes a person's existing debts and assets, and indication of whether the income is net or gross. 
These attributes may affect loan approval/disapproval, and without it, our predictions may not be as accurate as they could otherwise be. 

## EDA 
**CreditHistory** has the strongest positive correlation with LoanApproval (0.567105).
**PropertyArea_rural** has the most negative correlation wrt LoanApproval (-0.109203).

Negative correlations wrt Loan Approval:
- LoanAmount               -0.018536
- LoanAmountTerm           -0.033173
- SelfEmployed             -0.005660 (closest to 0)
- Gender_Female            -0.055320
- Gender_Unknown           -0.074998
- Married_No               -0.083304
- Dependents_0             -0.011764
- Dependents_1             -0.037075
- PropertyArea_Rural       -0.109203 (most negative)
- PropertyArea_Urban       -0.027789

Positive correlations wrt Loan Approval:
- ApplicantIncome           0.006396 
- CoapplicantIncome         0.066459
- Graduate                  0.099686
- CreditHistory             0.567105 (most positive)
- Gender_Male               0.077791
- Married_Unknown           0.040590
- Married_Yes               0.077816
- Dependents_2              0.047410
- Dependents_3              0.005173 (closest to 0)
- PropertyArea_Semiurban    0.129037

People who are approved for loans are most likely to: have coapplicants with high income, be a graduate, have a credit history, be male, be married/unknown, have two children and live in the suburbs.

Those who aren't approved tend to: ask for a large loan and long terms, are female/unknown gender, aren't married, have none or 1 child, and live in a rural or urban area. 

Interestingly, ApplicantIncome has only a very weak positive correlation with LoanApproval.

Features with a p-value < 5 with LoanApproval are Graduate (0.023) and PropertyArea_Semiurban (0.003), CreditHistory (0.000) and PropertyArea_Rural (0.013).
This implies that their correlations with LoanApproval is statistically significant.

Going back to my list of hypothesis of who gets loan approvals, I can accept/reject some of them based on the information I have currently:
- Applicants with high income in comparison to the loan amount UNSURE 
- Having a co-applicant with a relatively high income UNSURE 
- Being married UNSURE
- Having a small number of dependents UNSURE
- **Being a graduate ACCEPT**
- Not being self-employed UNSURE 
- **Owning a property in a semi-urban area ACCEPT**
- **Having a credit history ACCEPT**

## Process
**Process steps overview:**
1. Hypothesis Generation 
2. Data Cleaning
3. Exploratory Data Analysis  
4. Feature Engineering & Model Building 
5. Pipeline
6. API
7. Deployment 

### 1. Hypothesis Generation 
- Briefly research factors that influence loan approval. 

- Look at info available in dataset. 

- List of educated guesses/hypothesis about factors that influence loan approval in the context of the dataset.


### 2. Data Cleaning
- Impute NaNs with mode or median 
- OHE categorical features 
- Create 3rd ‘Unknown’ category for marriage and gender instead of imputing F/M or M/not-M, to acknowledge more possibilities for those
- Remove outliers 
- Try regression to impute NaNs

### 3. Exploratory Data Analysis
- Correlations between LoanStatus and other features
- Pearson’s test
    -  Significant positive correlations between loan approval and ‘graduated’, suburban, credit history present 
    - Significant negative correlation with living in rural area
- Tentatively accept/reject hypothesis from earlier list 

### 4. Feature Engineering & Model Building 
- MinMax scale, PCA 
- Try k-folds cv with various models, use model with highest av. accuracy
- Grid search on logistic regression model to find best hyperparameters 
- estimator = grid.best_estimator_ ; pickle model

### 5. Pipeline 
- **LoanData** class contains methods for:
    -  loading a CSV
    - cleaning and scaling
    - Returning X and y for model fitting or prediction
- **PredictionPipeline** class contains methods for:
    - Testing and fitting a logistic regression model
    - Making predictions using the model

### 6. API
- I attempted to make 2 APIs
- ran into lots of technical difficulties trying to implement the API's

### 7. Deployment 
- Unfinished 


## Results/Demo
Best parameters from grid search for logistic regression using PC's:
- 'lr__C': 1, 'lr__penalty': 'l1', 'lr__solver': 'liblinear', 'pca__n_components': 15
- Accuracy score: 0.8279226598557153

## Challanges 
- technical difficulties in implementing an API & deployment 

## Future Goals
- deploying to the cloud 
- create a website with polished UI & UX 
