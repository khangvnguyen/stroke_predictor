from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import xgboost as xgb
import numpy as np
import pandas as pd

WORK_TYPES = ['0', '2', '255', '1', '254', 'other']
AGE_BINS = [0, 45, 55, 65, 75, np.inf]
FEATURES = ['gender', 'age_range', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']

def load_model(model_path):
    # Load model
    model_xgb = xgb.Booster()
    model_xgb.load_model("xgb_model.json")
    return model_xgb

def read_data_csv(file_path = 'dataset_stroke.csv'):
    df_for_test = pd.read_csv(file_path)
    test_data = df_for_test[df_for_test.columns[:-1]]
    ground_truth_label = df_for_test[df_for_test.columns[-1]]
    return test_data, ground_truth_label

def run_prediction(model, test_data, ground_truth_label):
    # Preprocess
    test_data['work_type'] = np.where(
        test_data.work_type.apply(str).isin(WORK_TYPES),
        test_data.work_type.apply(str),
        'other',
    )
    test_data['work_type'] = test_data['work_type'].astype('category')
    test_data['age_range'] = pd.cut(test_data['age'], AGE_BINS, labels=False)
    test_data['age_range'] = test_data['age_range'].fillna(0).astype(int)
    
    # Predict
    y_test = pd.Series(ground_truth_label)
    X_test = xgb.DMatrix(test_data[FEATURES], enable_categorical=True)
    y_pred = model.predict(X_test).round()
    prediction_output = y_pred
    
    # Evaluate
    accuracy_sco = accuracy_score(y_test, y_pred)
    precision_sco = precision_score(y_test, y_pred)
    recall_sco = recall_score(y_test, y_pred)
    f1_sco = f1_score(y_test, y_pred)
    roc_auc_sco = roc_auc_score(y_test, y_pred)
    
    report_metrics = {
        'accuracy': accuracy_sco,
        'precision': precision_sco,
        'recall': recall_sco,
        'f1_score': f1_sco,
        'roc_auc': roc_auc_sco
    }

    return prediction_output, report_metrics