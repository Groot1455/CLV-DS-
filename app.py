from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('clv_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['Age'],
        data['Gender'],
        data['Tenure_Months'],
        data['MonthlyRevenue'],
        data['TransactionsPerMonth']
    ]
    prediction = model.predict([features])
    return jsonify({'predicted_clv': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
