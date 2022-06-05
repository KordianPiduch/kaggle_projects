import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def check_model(model, x_train, x_test, y_train, y_test, rounds=2):
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    print("\t\t", "train\t|", "test")
    print("precision\t", np.round(precision_score(y_train,y_pred_train),rounds),"\t|", np.round(precision_score(y_test,y_pred_test),rounds))
    print("recall\t\t", np.round(recall_score(y_train,y_pred_train),rounds), "\t|", np.round(recall_score(y_test,y_pred_test),rounds))
    print("f1-score\t", np.round(f1_score(y_train,y_pred_train),rounds), "\t|", np.round(f1_score(y_test,y_pred_test),rounds))
    print("accuracy\t", np.round(accuracy_score(y_train,y_pred_train),rounds), "\t|", np.round(accuracy_score(y_test,y_pred_test),rounds))