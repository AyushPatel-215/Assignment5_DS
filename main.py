import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = None
X = None
y = None
X_train, X_test, y_test, y_train = None, None, None, None


def fetchdata_from_csv():
    global data
    data = pd.read_csv('HR_comma_sep.csv')
    print("duplicate rows:", len(data[data.duplicated()]))
    data = data.drop_duplicates()


def get_data_and_make_plot():
    global X, y, X_train, X_test, y_test, y_train
    features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
                'Work_accident', 'promotion_last_5years', 'Department', 'salary']
    target = 'left'

    data_subset = data[features + [target]]
    sns.pairplot(data_subset, hue=target, palette='coolwarm')

    X = data[features]
    y = data[target]
    X = pd.get_dummies(X, columns=['Department', 'salary'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


def Make_model():
    global X, y, X_train, X_test, y_test, y_train

    # use normal logisticRegression
    model = LogisticRegression()
    y_pred = give_model(model)
    find_performance(y_test, y_pred)

    # use minmaxScaler and logistic regression
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_pred = give_model(LogisticRegression())
    print("minmaxScaler and logistic regression")
    find_performance(y_test, y_pred)

    # use StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_pred = give_model(LogisticRegression())
    print("Use StandardScaler")
    find_performance(y_test, y_pred)

    # use decision tree classifier
    y_pred = give_model(DecisionTreeClassifier())
    print("Use Decision Tree")
    find_performance(y_test, y_pred)

    # use random forest classifier
    y_pred = give_model(RandomForestClassifier())
    print("Use Random Forest")
    find_performance(y_test, y_pred)

    # use KNN classified
    y_pred = give_model(KNeighborsClassifier())
    print("Use KNN classifire")
    find_performance(y_test, y_pred)


def give_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def find_performance(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", confusion)


if __name__ == '__main__':
    fetchdata_from_csv()
    get_data_and_make_plot()
    Make_model()
