# *** GENERATED PIPELINE ***

# LOAD DATA
import pandas as pd
train_dataset = pd.read_pickle(r"C:\Users\Hioku\图形\outputs\training.pkl")

# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
def split_dataset(dataset, train_size=0.75, random_state=17):
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, random_state=random_state, stratify=dataset["anomaly"])
    return train_dataset, test_dataset
train_dataset, test_dataset = split_dataset(train_dataset)


# PREPROCESSING-1
# Component: Preprocess:TextPreprocessing
# Efficient Cause: Preprocess:TextPreprocessing is required in this pipeline since the dataset has ['feature:str_text_presence']. The relevant features are: ['processed_summary'].
# Purpose: Preprocess and normalize text.
# Form:
#   Input: array of strings
#   Key hyperparameters used: None
# Alternatives: Although  can also be used for this dataset, Preprocess:TextPreprocessing is used because it has more  than .
# Order: Preprocess:TextPreprocessing should be applied  
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
test_dataset = process_text(test_dataset)

# DETACH TARGET
TARGET_COLUMNS = ['anomaly']
feature_train = train_dataset.drop(TARGET_COLUMNS, axis=1)
target_train = train_dataset[TARGET_COLUMNS].copy()
if set(TARGET_COLUMNS).issubset(test_dataset.columns.tolist()):
    feature_test = test_dataset.drop(TARGET_COLUMNS, axis=1)
    target_test = test_dataset[TARGET_COLUMNS].copy()
else:
    feature_test = test_dataset

# PREPROCESSING-2
# Component: Preprocess:TfidfVectorizer
# Efficient Cause: Preprocess:TfidfVectorizer is required in this pipeline since the dataset has ['feature:str_text_presence']. The relevant features are: ['processed_summary'].
# Purpose: Convert a collection of raw documents to a matrix of TF-IDF features.
# Form:
#   Input: raw_documents
#   Key hyperparameters used: 
#		 "max_features: int, default=None" :: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None.
# Alternatives: Although  can also be used for this dataset, Preprocess:TfidfVectorizer is used because it has more  than .
# Order: Preprocess:TfidfVectorizer should be applied  
from sklearn.feature_extraction.text import TfidfVectorizer
TEXT_COLUMNS = ['processed_summary']
temp_train_data = feature_train[TEXT_COLUMNS]
temp_test_data = feature_test[TEXT_COLUMNS]
# Make the entire dataframe sparse to avoid it converting into a dense matrix.
feature_train = feature_train.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
feature_test = feature_test.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
for _col in TEXT_COLUMNS:
    tfidfvectorizer = TfidfVectorizer(max_features=3000)
    vector_train = tfidfvectorizer.fit_transform(temp_train_data[_col])
    feature_names = ['_'.join([_col, name]) for name in tfidfvectorizer.get_feature_names_out()]
    vector_train = pd.DataFrame.sparse.from_spmatrix(vector_train, columns=feature_names, index=temp_train_data.index)
    feature_train = pd.concat([feature_train, vector_train], axis=1)
    vector_test = tfidfvectorizer.transform(temp_test_data[_col])
    vector_test = pd.DataFrame.sparse.from_spmatrix(vector_test, columns=feature_names, index=temp_test_data.index)
    feature_test = pd.concat([feature_test, vector_test], axis=1)

# PREPROCESSING-3
# Component: Preprocess:StandardScaler
# Efficient Cause: Preprocess:StandardScaler is required in this pipeline since the dataset has ['feature:max_skewness', 'feature:max_normalized_stddev', 'feature:max_normalized_mean']. The relevant features are: all columns in the dataset.
# Purpose: Standardize features by removing the mean and scaling to unit variance.
# Form:
#   Input: {array-like, sparse matrix} of shape (n_samples, n_features)
#   Key hyperparameters used: 
#		 "with_mean: bool, default=True" :: If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
# Alternatives: Although  can also be used for this dataset, Preprocess:StandardScaler is used because it has more  than .
# Order: Preprocess:StandardScaler should be applied  
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler(with_mean=False)
feature_train = pd.DataFrame.sparse.from_spmatrix(standard_scaler.fit_transform(feature_train), columns=feature_train.columns, index=feature_train.index)
feature_test = pd.DataFrame.sparse.from_spmatrix(standard_scaler.transform(feature_test), columns=feature_test.columns, index=feature_test.index)

# PREPROCESSING-4
# Component: Preprocess:SMOTE
# Efficient Cause: Preprocess:SMOTE is required in this pipeline since the dataset has ['feature:target_imbalance_score', 'feature:target_imbalance_score', 'feature:target_imbalance_score']. The relevant features are: all columns in the dataset.
# Purpose: Perform over-sampling
# Form:
#   Input: Dictionary containing the information to sample the dataset. The keys corresponds to the class labels from which to sample and the values are the number of samples to sample.
#   Key hyperparameters used: None
# Alternatives: Although  can also be used for this dataset, Preprocess:SMOTE is used because it has more  than .
# Order: Preprocess:SMOTE should be applied  
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
y_pred = model.predict(feature_test)

#EVALUATION
if set(TARGET_COLUMNS).issubset(test_dataset.columns.tolist()):
    from sklearn import metrics
    f1 = metrics.f1_score(target_test, y_pred, average='macro')
    print('RESULT: F1 Score: ' + str(f1))

# Confusion Matrix
if set(TARGET_COLUMNS).issubset(test_dataset.columns.tolist()):
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(target_test, y_pred)

# OUTPUT PREDICTION
prediction = pd.DataFrame(y_pred, columns=TARGET_COLUMNS, index=feature_test.index)
prediction.to_csv("./prediction_result.csv")

# PERMUTATION IMPORTANCE
from sklearn.inspection import permutation_importance
if len(feature_train.columns) <= 100:
    perm = permutation_importance(model, feature_train.sparse.to_dense(), target_train[TARGET_COLUMNS[0]],
                                    n_repeats=5,
                                    random_state=0)
    perm_df = pd.DataFrame({"feature": feature_train.columns, "importance": perm.importances_mean})
    perm_df.to_csv("./permutation_importance.csv", index=False)


# Models are restricted because of execution time.
models_for_shap = ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor']
if model.__class__.__name__ in models_for_shap:
    import shap
    feature_shap = feature_train.sample(1000) if feature_train.shape[0] > 1000 else feature_train
    explainer = shap.Explainer(model)
    shap_values = explainer(feature_shap)
    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values)
    #bar plots
    shap.plots.bar(shap_values)
