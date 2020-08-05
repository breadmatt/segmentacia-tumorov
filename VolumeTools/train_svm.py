import pandas as pd
import numpy as np
from sklearn import svm
from joblib import dump, load
TRAIN_CSV = "../train_data.csv"

print("Loading csv.")
train_df = pd.read_csv(TRAIN_CSV, delimiter=",")
feature_names = train_df.columns[:-1]
features = train_df[feature_names].values
labels = train_df[["label"]].values.flatten()

classifiers = [
    [svm.SVC(kernel="rbf", gamma='scale', cache_size=200, class_weight="balanced"), "rbf_model.joblib"],
    [svm.SVC(kernel="poly", gamma='scale', cache_size=200, class_weight="balanced", degree=3), "poly_d3_model.joblib"],
    [svm.SVC(kernel="poly", gamma='scale', cache_size=200, class_weight="balanced", degree=5), "poly_d5_model.joblib"],
    [svm.SVC(kernel="sigmoid", gamma='scale', cache_size=200, class_weight="balanced", degree=5), "sigmoid_model.joblib"],
]

for idx, clf in enumerate(classifiers):
    print(f"Fitting model: {clf[1]}")
    clf[0].fit(features, labels)
    print("Fitting done.")
    dump(clf[0], clf[1])
    print("Model saved.")
print("Finished.")