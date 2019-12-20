import pandas as pd
import numpy as np

wine = pd.read_csv('trainingdataxx.csv', names = ["1", "2", "3", "4", "5", "6","7","8", "9",
                                                "10","11", "12", "13","14","15","16","17","18",
                                                "name"])

x_train = wine.drop('name', axis=1)
y_train = wine['name']

#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x,y)

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#scaler.fit(x_train)

#StandardScaler(copy=True, with_mean=True, with_std=True)

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import classification_report,confusion_matrix

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,
                    learning_rate_init=0.001, momentum=0.9, random_state=0)

model = mlp.fit(x_train, y_train)

'''MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)'''

x_test = pd.read_csv('testingdata.csv', names = ["1", "2", "3", "4", "5", "6","7","8", "9",
                                                "10","11", "12", "13","14","15","16","17","18"])

predictions = model.predict_log_proba(x_test)
prediction_prob = model.predict_proba(x_test)
predict = model.predict(x_test)

print(predict)
print((prediction_prob)+1)

#prediction = mlp.predict(x_test)
#print(classification_report(y_test,prediction))
