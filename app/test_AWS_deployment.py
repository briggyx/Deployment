## Python test file for flask to test locally
import requests as r
import pandas as pd
import json

base_url =  'http://ec2-54-183-191-103.us-west-1.compute.amazonaws.com:5555/test' #base url local host

json_data = {
    "ApplicantIncome": 0.443788,
    "CoapplicantIncome": 0.264515,
    "LoanAmount": 0.46124,
    "LoanAmountTerm": 0.74359,
    "Graduate": 1,
    "SelfEmployed": 0,
    "CreditHistory": 1,
    "Gender_Female": 0,
    "Gender_Male": 1,
    "Gender_Unknown": 0,
    "Married_No": 0,
    "Married_Unknown": 0,
    "Married_Yes": 1,
    "Dependents_0": 0,
    "Dependents_1": 1,
    "Dependents_2": 0,
    "Dependents_3": 0,
    "PropertyArea_Rural": 0,
    "PropertyArea_Semiurban": 0,
    "PropertyArea_Urban": 1
}


# Get Response
# response = r.get(base_url)
response = r.post(base_url, json = json_data)
print(response.status_code)

if response.status_code == 200:
    print('...')
    print('request successful')
    print('...')
    print(response.json())
else:
    print(response.json())
    print('request failed')

