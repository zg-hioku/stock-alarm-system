# *** GENERATED PIPELINE ***

# LOAD DATA
import pandas as pd
test_dataset = pd.read_pickle("./test.pkl")

import pickle


# PREPROCESSING-1
import re
import string
import nltk
TEXT_COLUMNS = ['processed_summary']
def process_text(__dataset):
    for _col in TEXT_COLUMNS:
        process_text = [t.lower() for t in __dataset[_col]]
        # strip all punctuation
        table = str.maketrans('', '', string.punctuation)
        process_text = [t.translate(table) for t in process_text]
        # convert all numbers in text to 'num'
        process_text = [re.sub(r'\d+', 'num', t) for t in process_text]
        __dataset[_col] = process_text
    return __dataset
test_dataset = process_text(test_dataset)

# DETACH TARGET
TARGET_COLUMNS = ['anomaly']
if set(TARGET_COLUMNS).issubset(test_dataset.columns.tolist()):
    feature_test = test_dataset.drop(TARGET_COLUMNS, axis=1)
    target_test = test_dataset[TARGET_COLUMNS].copy()
else:
    feature_test = test_dataset

# PREPROCESSING-2
TEXT_COLUMNS = ['processed_summary']
temp_test_data = feature_test[TEXT_COLUMNS]
# Make the entire dataframe sparse to avoid it converting into a dense matrix.
feature_test = feature_test.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
with open('tfidfVectorizer.pkl', 'rb') as f:
    vectorizers = pickle.load(f)
for _col in TEXT_COLUMNS:
    tfidfvectorizer = vectorizers[_col]
    feature_names = ['_'.join([_col, name]) for name in tfidfvectorizer.get_feature_names_out()]
    vector_test = tfidfvectorizer.transform(temp_test_data[_col])
    vector_test = pd.DataFrame.sparse.from_spmatrix(vector_test, columns=feature_names, index=temp_test_data.index)
    feature_test = pd.concat([feature_test, vector_test], axis=1)

# PREPROCESSING-3
with open('standardScaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)
feature_test = pd.DataFrame.sparse.from_spmatrix(standard_scaler.transform(feature_test), columns=feature_test.columns, index=feature_test.index)

# MODEL
import numpy as np
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(feature_test)

#EVALUATION
if set(TARGET_COLUMNS).issubset(test_dataset.columns.tolist()):
    from sklearn import metrics
    f1 = metrics.f1_score(target_test, y_pred, average='macro')
    print('RESULT: F1 Score: ' + str(f1))

# OUTPUT PREDICTION
prediction = pd.DataFrame(y_pred, columns=TARGET_COLUMNS, index=feature_test.index)
prediction.to_csv("./prediction_result.csv")
