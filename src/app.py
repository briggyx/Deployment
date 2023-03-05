# import Flask and jsonify
from flask import Flask, jsonify

# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse

# import other packages
from loans import PredictionPipeline
import pandas as pd

# load both pickled prediction pipelines
pipe_on_full = PredictionPipeline()
pipe_on_even = PredictionPipeline()

pipe_on_full.load('PipeOnFull')
pipe_on_even.load('PipeOnEven')

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def get(self):
        
        parser = reqparse.RequestParser()

        # Get all arguments; we will assume no missing data from the web interface
        parser.add_argument('gender', type=str) # input must be m, f, or u (underclared)
        parser.add_argument('married', type=str) # input must be y, n, u (undeclared) 
        parser.add_argument('proptype', type=str) # input must be r, s, u (rural, urban, semiurban)
        parser.add_argument('dependents', type=int) # number of dependents; over 3 will be set to 3 to match training data
        parser.add_argument('graduate', type=int) # input is 0=no, 1=yes
        parser.add_argument('selfemployed', type=str) # input is 0=no, 1=yes
        parser.add_argument('income', type=float) # applicant income; training values = 150-10,000
        parser.add_argument('income_co', type=float) # co-applicant income; training values = 0-5,700
        parser.add_argument('amount', type=float) # loan amount; training values = 9-260
        parser.add_argument('term', type=float) # loan term; training values = 
        parser.add_argument('credithistory', type=int) # input is 0=no, 1=yes

        # Parse all arguments
        gender = parser.parse_args().get('gender')
        married = parser.parse_args().get('married')
        proptype = parser.parse_args().get('proptype')
        Dependents = min(parser.parse_args().get('dependents'),3)
        graduate = parser.parse_args().get('graduate')
        selfemployed = parser.parse_args().get('selfemployed')
        income = parser.parse_args().get('income')
        income_co = parser.parse_args().get('income_co')
        amount = parser.parse_args().get('amount')
        credithistory = parser.parse_args().get('credithistory')
        
        # List to hold data in correct order
        data = []
        
        # Deal with data that needs to be one-hot-encoded
        data.append(1 if gender == 'f' else 0)
        data.append(1 if gender == 'm' else 0)
        data.append(1 if gender == 'u' else 0)
        data.append(1 if married == 'n' else 0)
        data.append(1 if married == 'u' else 0)
        data.append(1 if married == 'y' else 0)
        data.append(1 if proptype == 'r' else 0)
        data.append(1 if proptype == 's' else 0)
        data.append(1 if proptype == 'u' else 0)
        data.append(graduate)
        data.append(selfemployed)
        data.append(income)
        data.append(income_co)
        data.append(amount)
        data.append(credithistory)
        
        data = pd.DataFrame(data).T
        data.columns(['Gender_Female','Gender_Male','Gender_Unknown','Married_No','Married_Unknown','Married_Yes',
                      'Rural','Semiurban','Urban','Dependents','Graduate','SelfEmployed','ApplicantIncome','CoapplicantIncome',
                      'LoanAmount','LoanAmountTerm','CreditHistory','LoanStatus'])
        
        p1, p2, p3, p4 = pipe_on_full.predict(data)[0], pipe_on_full.predict(data,neural=True), pipe_on_even.predict(data), pipe_on_even(data,neural=True)
        
        print('LogReg on full data predicts: %s' % 'Approval' if p1 else 'Rejection')
        print('Neural network on full data predicts: %s' % 'Approval' if p2 else 'Rejection')
        print('LogReg on even data predicts: %s' % 'Approval' if p3 else 'Rejection')
        print('Neural network on even data predicts: %s' % 'Approval' if p4 else 'Rejection')
        
        return data
    
# assign endpoint
api.add_resource(Predict, '/predict',)

if __name__ == '__main__':
    app.run(debug=True)