import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

def encode_df(df):
    pkl_file = open('model/encoding/le_dic.pkl', 'rb')
    le_dic = pickle.load(pkl_file) 
    pkl_file.close()
    for col, le in le_dic.items():
        if col != 'merchant_id':
            df[col] = le.transform(df[col])
    
    return df
    
def adjust_df_datatype(df):
    df[df == "(No value)"] = None
    df['created_hour'] = pd.to_numeric(df['created_hour'])
    df['created_dayofweek'] = pd.Categorical(df['created_dayofweek'],
                                   categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday' , 'Saturday', 'Sunday'],
                                   ordered=True)
    df['first_click_in_2021'] = df['first_click_in_2021'].map({'True': True, 'False': False})
    df['time_diff_since_last_click_current_year_in_minutes'] = pd.to_numeric(df['time_diff_since_last_click_current_year_in_minutes'])
    df['click_count'] = pd.to_numeric(df['click_count'])
    df['average_weekly_click_count'] = pd.to_numeric(df['average_weekly_click_count'])
    df['click_count_most_clicked_merchant_id'] = pd.to_numeric(df['click_count_most_clicked_merchant_id'])
    
    return df

def decode_one(val,column):
    pkl_file = open('model/encoding/le_dic.pkl', 'rb')
    le_dic = pickle.load(pkl_file) 
    pkl_file.close()
    return le_dic[column].inverse_transform([val])[0]

def decode(val,column):
    pkl_file = open('model/encoding/le_dic.pkl', 'rb')
    le_dic = pickle.load(pkl_file) 
    pkl_file.close()
    return le_dic[column].inverse_transform(val)
    
def model_predict_one(model, df):
    d_df = xgb.DMatrix(df)
    predict = model.predict(d_df)
    best_predict = np.argmax(predict)
    best_predict_decoded = decode_one(best_predict,'merchant_id')
    return predict, best_predict_decoded

def model_predict(model, df):
    d_df = xgb.DMatrix(df)
    predict = model.predict(d_df)
    best_predict = np.asarray([np.argmax(line) for line in predict])
    best_predict_decoded = decode(best_predict,'merchant_id')
    return predict, best_predict_decoded

def plot_compare(metrics,eval_results,steps):
    for m in metrics:
        test_score = eval_results['val'][m]
        train_score = eval_results['train'][m]
        rang = range(0, steps)
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, test_score,"c", label="Val")
        plt.plot(rang, train_score,"orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()

def fitXgb(param, steps, D_train, D_val, D_test, y_test, features, le_dic):
    print('Training model...')
    metrics = ['mlogloss','merror']
    evallist = [(D_val, 'val'),(D_train,'train')]
    store = {}
    target = list(le_dic['merchant_id'].classes_)
    labels = [x for x in range(0, len(target))]
    model = xgb.train(param, D_train, steps, evals=evallist, evals_result=store, verbose_eval=5)
    print('Training done!\n\n')
    print('-- Model Report --')
    y_predict = model.predict(D_test)
    y_best_predict = np.asarray([np.argmax(line) for line in y_predict])
    accuracy = accuracy_score(y_test, y_best_predict, normalize=True)
    f1 = f1_score(y_test, y_best_predict, average='macro')
    print('Accuracy: '+str(accuracy))
    print('F1-Score (Macro): '+str(f1))
    print('Precision: {}'.format(precision_score(y_test, y_best_predict, average='macro')))
    print('Recall: {}'.format(recall_score(y_test, y_best_predict, average='macro')))
    print('\nClassification Report:\n')
    print(classification_report(y_test, y_best_predict, labels=labels, target_names=target))
    plot_compare(metrics, store, steps)
    f, ax = plt.subplots(figsize=(10,5))
    feature_importance = model.get_score(importance_type='gain')
    plot = sns.barplot(x=list(feature_importance.keys()), y=np.fromiter(feature_importance.values(), dtype=float))
    ax.set_title('Feature Importance (Gain)')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()
    return model