import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

train = pd.read_csv('trainingdataft.csv', names = ["1", "2", "3", "4", "5", "6","7","8", "9",
                                                "10","11", "12", "13","14","15","16","17","18",
                                                "name"])

x_train = train.drop('name', axis=1)
y_train = train['name']

test = pd.read_csv('testingdataft.csv', names = ["1", "2", "3", "4", "5", "6","7","8", "9",
                                                "10","11", "12", "13","14","15","16","17","18",
                                                "name"])

x_test = test.drop('name', axis=1)
y_test = test['name']

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#scaler = StandardScaler()
#scaler.fit(x_train)

#StandardScaler(copy=True, with_mean=True, with_std=True)

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,
                    learning_rate_init=0.001,
                    momentum=0.9, random_state=0)

mlp.fit(x_train, y_train)

'''MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)'''

predictions = mlp.predict_log_proba(x_test)
prediction_prob = mlp.predict_proba(x_test)
predict = mlp.predict(x_test)

foo = [np.amax(x) for x in (prediction_prob+1)]

print(classification_report(y_test,predict))
print(predict)
print (foo)
#print(np.amax((prediction_prob)+1)[i])
