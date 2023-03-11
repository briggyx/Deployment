from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

model = pickle.load( open( "lr_model.pkl", "rb" ))

class Test(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # to have a result is probability
        # res = model.predict_proba(df)
        res = model.predict(df)
        return res.tolist() 

api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
