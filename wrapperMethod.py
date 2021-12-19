import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import plot_confusion_matrix,classification_report,plot_precision_recall_curve,plot_roc_curve



def report(model):
    preds = model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,X_test,y_test)
    plot_precision_recall_curve(model,X_test,y_test)
    plot_roc_curve(model,X_test,y_test)

df = pd.read_csv("emails.csv")


X = df.iloc[:, 1:-1].values # Except the first and last column
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
# Split dataset into random trainset and testset


m = DecisionTreeClassifier(min_samples_leaf=20)
# Use DecisionTree as learning module

sfs = SFS(m, forward=True, cv=0, k_features = 5, scoring='accuracy', verbose=1, n_jobs=-1)
# Sequential feature selection of wrapper method, direction = forward

sfs.fit(X_train, y_train)

print(f"Best score achieved: {sfs.k_score_}, Feature's names: {sfs.k_feature_names_}")

m.fit(X_train, y_train)
print("sfs result")
report(m)





