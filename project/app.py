from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

log_reg_model, log_reg_model_scaler = joblib.load('best_log_reg_model.pkl').values()
rf_model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            array_input = request.form['array_input']
            array_input = array_input.replace('\n', ' ').replace('\r', ' ').strip()
            features = np.fromstring(array_input, sep=' ')

            if features.shape[0] != log_reg_model.n_features_in_:
                raise ValueError(f"Expected {log_reg_model.n_features_in_} features, got {features.shape[0]}")

            features = features.reshape(1, -1)
            features_log = log_reg_model_scaler.transform(features)



            log_reg_prediction = log_reg_model.predict(features_log)[0]
            rf_prediction = rf_model.predict(features)[0]

            return render_template('index.html',
                                   log_reg_result=log_reg_prediction,
                                   rf_result=rf_prediction)
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)