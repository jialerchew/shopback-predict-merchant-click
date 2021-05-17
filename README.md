# Shopback Merchant Click Analysis and Prediction
**Chew Jia Ler**

## Overview
Given the datasets of clicks of merchants on Shopback Korea between January - March 2021, the following were built:
1. Exploratory data analysis (EDA) on given dataset.
2. A prediction model to predict which merchant a user will click.
3. Web service to serve the model for experiments.

## Files
```
.
├── data                                # Given datasets, contains 3 datasets
├── metadata                            # Metadata for web service
├── model                               # Trained model
│   └── encoding                        # Pickle file that stores a dict of encoding/decoding
├── static                              # Automated tests (alternatively `spec` or `tests`)
├── templates                           # Tools and utilities
├── app.py                              # File to launch web server
├── EDA and Model Tranining.ipynb       # Notebook that documents EDA and model training process
├── README.md
├── requirements.txt                    # List of packages required to run repository
├── Sample batch predict.ipynb          # Example on how to use batch_predict endpoint
└── utils.py                            # Overall utility methods
```


## How to start web service

### Requirements

Please refer to `requirements.txt`. Python version used is 3.9.4.

### Getting started

1. Create a virtual environment with Python 3.9.4.

2. Install dependencies from `requirements.txt` via:

        pip install -r requirements.txt

3. Run web server by running:

        python app.py

### Make predictions

#### Single prediction

1. Visit `http://localhost:5000/predict` on your web browser.

2. Play with it!

#### Batch predictions

* Batch predictions via `http://localhost:5000/batch_predict` endpoint.
* Takes in a list of JSON with the follow keys:

| Key  | Description | Sample Value |
| :------------ | :------------ | :------------ |
| device ** | User Device | 'app_android' |
| platform **  | Platform that user makes the click from | 'Android App' |
| channel ** | Channel that contributes the click  | 'paid_ins_organic' |
| created_hour  | Hour in which user made the click (0 - 23) | '15' |
| created_dayofweek ** | Day in the week in which user made the click | 'Tuesday' |
| first_click_in_2021 ** | If this is user's first click in 2021 | 'False' |
| time_diff_since_last_click_current_year_in_minutes | Timegap between current and previous click, in minutes | '2655' |
| lifetime_first_merchant_id ** | The first merchant that user purchase with Shopback | '12251' |
| account_referral ** | Referral of the account, usually refers to campaigns | 'partnership' |
| click_count | Number of merchant click by user in lifetime | '25' |
| average_weekly_click_count | Average weekly number of merchant click by user in lifetime | '0.514' |
| most_clicked_merchant_id **| Merchant that user clicked most frequently | '12251' |
| click_count_most_clicked_merchant_id | Number of clicks on merchant that user clicked most frequently | '23' |

*\*\* These are columns that do not support free values, please refer to text files in metadata folder for allowable values*


* **All values need to be in string format, including numerics.**
* Predictions are returned as an additional key in the original JSON list.
* Refer to `Sample batch predict.ipynb` notebook for a simple example of using endpoint.
* Try not to intentionally break the web service as it is not test vigorously to support all data types.