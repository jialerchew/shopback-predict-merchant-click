from flask import Flask,request, url_for, redirect, render_template, jsonify
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import *


app = Flask(__name__)

model = xgb.Booster()
model.load_model("model/model.json")

cols = ['device',
 'platform',
 'channel',
 'created_hour',
 'created_dayofweek',
 'first_click_in_2021',
 'time_diff_since_last_click_current_year_in_minutes',
 'lifetime_first_merchant_id',
 'account_referral',
 'click_count',
 'average_weekly_click_count',
 'most_clicked_merchant_id',
 'click_count_most_clicked_merchant_id']

device = sorted([x.rstrip() for x in open("metadata/device.txt", "r").readlines()])
platform = sorted([x.rstrip() for x in open("metadata/platform.txt", "r").readlines()])
channel = sorted([x.rstrip() for x in open("metadata/channel.txt", "r").readlines()])
created_dayofweek = sorted([x.rstrip() for x in open("metadata/created_dayofweek.txt", "r").readlines()])
first_click_in_2021 = sorted([x.rstrip() for x in open("metadata/first_click_in_2021.txt", "r").readlines()])
lifetime_first_merchant_id = sorted([x.rstrip() for x in open("metadata/lifetime_first_merchant_id.txt", "r").readlines()])
account_referral = sorted([x.rstrip() for x in open("metadata/account_referral.txt", "r").readlines()])
most_clicked_merchant_id = sorted([x.rstrip() for x in open("metadata/most_clicked_merchant_id.txt", "r").readlines()])

@app.route('/')
def home():
    return render_template("wronghome.html")


@app.route('/predict')
def pred_home():
    return render_template('home.html',
                           device=device,
                           platform=platform,
                           channel=channel,
                           created_dayofweek=created_dayofweek,
                           first_click_in_2021=first_click_in_2021,
                           lifetime_first_merchant_id=lifetime_first_merchant_id,
                           account_referral=account_referral,
                           most_clicked_merchant_id=most_clicked_merchant_id)

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    input_data = np.array(int_features)
    input_df = pd.DataFrame([input_data], columns = cols)
    input_df = adjust_df_datatype(input_df)
    encoded_input_df = encode_df(input_df)
    predict, best_predict = model_predict_one(model, encoded_input_df)
    return render_template('home.html',
                           device=device,
                           platform=platform,
                           channel=channel,
                           created_dayofweek=created_dayofweek,
                           first_click_in_2021=first_click_in_2021,
                           lifetime_first_merchant_id=lifetime_first_merchant_id,
                           account_referral=account_referral,
                           most_clicked_merchant_id=most_clicked_merchant_id,
                           pred='Expected merchant is {}'.format(best_predict))

@app.route('/batch_predict',methods=['POST'])
def predict_api():
    data = request.get_json(force=True,silent=True)
    if data is None:
        return 'Bad input', status.HTTP_400_BAD_REQUEST
    
    input_df = pd.DataFrame.from_dict(data, orient='columns')
    input_df = adjust_df_datatype(input_df)
    encoded_input_df = encode_df(input_df)
    predict, best_predict = model_predict(model, encoded_input_df)
    
    for idx, val in enumerate(data):
        val['prediction'] = best_predict[idx]
        
    return jsonify(data)



if __name__ == '__main__':
    app.run(debug=True)