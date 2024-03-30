from django.shortcuts import render, redirect

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# Create your views here.
def index(request):
    return render(request, 'index.html', {})


def PredictForm(request):
    return render(request, 'form.html', {})


def result(request):
    if request.method == 'POST':
        dataset = pd.read_csv('breast_cancer.csv')

        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LogisticRegression(random_state=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = (accuracy_score(y_test, y_pred)) * 100

        #predictions = model.predict(X_test)

        val1 = float(request.POST['val1'])
        val2 = float(request.POST['val2'])
        val3 = float(request.POST['val3'])
        val4 = float(request.POST['val4'])
        val5 = float(request.POST['val5'])
        val6 = float(request.POST['val6'])
        val7 = float(request.POST['val7'])
        val8 = float(request.POST['val8'])
        val9 = float(request.POST['val9'])
        val10 = float(request.POST['val10'])

        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10]])
        if pred[0] == 1:
            result = "There is high chances of Breast Cancer."
        else:
            result = "You are safe but stil checking is better."

    return render(request, 'form.html', {'result':result, 'accuracy':accuracy})
