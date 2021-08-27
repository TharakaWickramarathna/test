from flask import Flask, request, jsonify
import joblib
import sklearn
import numpy as np
import json 

app = Flask(__name__)

# Load the model
MODEL = joblib.load('swm_model.pkl')


@app.route('/predict')
def predict():
    
    with open('population_data.json') as f:
        dict = json.load(f)
  
    town = request.args.get('town')
    population=dict[town]
    
    
    features = [[population]]
    feature_array=np.array(features)
    converted_feature=feature_array.astype(np.float)
    
   
    quarterly_waste_array = MODEL.predict(converted_feature.reshape(-1,1))
    quarterly_waste=quarterly_waste_array[0]
    monthly_waste=quarterly_waste/3
   
    return jsonify(status='complete', prediction=monthly_waste)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')