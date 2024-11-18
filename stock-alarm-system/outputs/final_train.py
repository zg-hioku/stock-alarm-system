# *** GENERATED PIPELINE ***

# LOAD DATA
import pandas as pd
train_dataset = pd.read_pickle("./training.pkl")

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
train_dataset = process_text(train_dataset)

# DETACH TARGET
TARGET_COLUMNS = ['anomaly']
feature_train = train_dataset.drop(TARGET_COLUMNS, axis=1)
target_train = train_dataset[TARGET_COLUMNS].copy()

# PREPROCESSING-2
from sklearn.feature_extraction.text import TfidfVectorizer
TEXT_COLUMNS = ['processed_summary']
temp_train_data = feature_train[TEXT_COLUMNS]
# Make the entire dataframe sparse to avoid it converting into a dense matrix.
feature_train = feature_train.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
vectorizers = {}
for _col in TEXT_COLUMNS:
    tfidfvectorizer = TfidfVectorizer(max_features=3000)
    vector_train = tfidfvectorizer.fit_transform(temp_train_data[_col])
    feature_names = ['_'.join([_col, name]) for name in tfidfvectorizer.get_feature_names_out()]
    vector_train = pd.DataFrame.sparse.from_spmatrix(vector_train, columns=feature_names, index=temp_train_data.index)
    feature_train = pd.concat([feature_train, vector_train], axis=1)
    vectorizers[_col] = tfidfvectorizer
with open('tfidfVectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizers, f)

# PREPROCESSING-3
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler(with_mean=False)
feature_train = pd.DataFrame.sparse.from_spmatrix(standard_scaler.fit_transform(feature_train), columns=feature_train.columns, index=feature_train.index)
with open('standardScaler.pkl', 'wb') as f:
    pickle.dump(standard_scaler, f)

# PREPROCESSING-4
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
feature_columns = feature_train.columns
feature_train = feature_train.sparse.to_coo()
feature_train, target_train = smote.fit_resample(feature_train, target_train)
feature_train =  pd.DataFrame.sparse.from_spmatrix(feature_train, columns=feature_columns)

# MODEL
import numpy as np
from sklearn.linear_model import LogisticRegression
random_state_model = 42
model = LogisticRegression(random_state=random_state_model, )
model.fit(feature_train, target_train.values.ravel())
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
