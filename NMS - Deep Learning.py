import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import psutil
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical


print("Deep Learning Classifier")
# Reads the train and test data
#train = pd.read_csv('1Million.csv', header=None)
#train = pd.read_csv('500K.csv', header=None)
#train = pd.read_csv('./125K.csv', header=None)
#)train = train.iloc[:,0:42]
train = pd.read_csv('./KDDTrain+.csv', header=None)
test = pd.read_csv('./KDDTest+.csv', header=None)
print("Train and Test Data read...")


# Reads "Field Names.csv". Use this to set the names of train and test data columns
columns = pd.read_csv('Field Names.csv', header=None)
columns.columns = ['name', 'type']
train.columns = columns['name']
test.columns = columns['name']


# Read Service Types.csv
# Use this to create a mapping from service types to final labels (MAIL, WWW, FTP, DATABASE, v.b.)
serviceType = pd.read_csv('Service Types.csv', header=None)
serviceType.columns = ['Name', 'Type']
serviceMap={}



# Creates attackMap map which contains a mapping between service type and the final label
for i in range(len(serviceType)):
    serviceMap[serviceType['Name'][i]] = serviceType['Type'][i]
print("Service type mapping created...")



# Add a new variable called 'label' which contains the final label
train['label'] = train['service'].map(serviceMap)
test['label'] = test['service'].map(serviceMap)



#find count of classes from serviceType
classesCount = len(train['label'].drop_duplicates())
classesName = train['label'].drop_duplicates().values.tolist()
print('There are ' +  str(classesCount) + ' classes to be monitored')
print('Classes are:')
print(classesName)



# Transform the existing nominal variables into the integer coded variables using the LabelEncoder
for col in ['protocol_type', 'flag', 'attack_type', 'label']:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    le1 = LabelEncoder()
    le1.fit(test[col])
    test[col] = le1.transform(test[col])



# The variable 'label' is stored in different variables
# This is required to keep the dependent variable separate from the independent variable
trainLabel = train['label']
testLabel = test['label']



#Transform dependent variable into categorical columns as much as number of classes
trainLabel = to_categorical(trainLabel, classesCount)
testLabel = to_categorical(testLabel, classesCount)



# service_type and label variables are removed from the train and test data
# so only features are remained in train and test data
train.drop(['service', 'label'], axis=1, inplace=True)
test.drop(['service', 'label'], axis=1, inplace=True)
print("Train and Test data labels created...")



# for Stadart Scaler   
scaler = MinMaxScaler()   #scale between 0 and 1
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)      
total = np.concatenate([train, test] )



# Decomposition features are generated for both train and test data
pca = PCA(n_components=41, random_state=100)
pca.fit(total)
train = pca.transform(train)
test = pca.transform(test)



print("Decomposed features created...")
print("Number of features used : %d" % train.shape[1])


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)



# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(110, input_dim=41, activation='relu'))
	model.add(Dense(110, activation='relu'))
	model.add(Dense(classesCount, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


DLC = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)

'''
# Performing k-fold Cross validation
startTime = time.clock()
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(DLC, train, trainLabel, cv=kfold)
endTime = time.clock()
print("Time taken to perform 5-fold cross validation : %f" % (endTime - startTime))
print("Mean score of 5-fold cross validation: %.2f%% " % (results.mean()*100))
'''


# Final Testing and Evaluate Performance
# Train the KNN classifier model by original train data and got optimized parameter
startTime = time.clock()
DLC.fit(train, trainLabel, epochs =10)
endTime = time.clock()
print("Time taken to train final model : %f" % (endTime - startTime))
print("Predictions made using final model...")


# Predictions for test data and evaluate its performance
startTime = time.clock()
pred = DLC.predict(test)
endTime = time.clock()
cpuUsage = psutil.cpu_percent()
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0] / 2. ** 30
print("Time taken to make predictions on test data : %f" % (endTime - startTime))
print("Memory used : %f GB  CPU usage : %f" % (memoryUse, cpuUsage))



pred = to_categorical(pred,classesCount)
pred = np.array(pred)
testLabel = np.array(testLabel)


#calculate confusion matrix
con_matrix = confusion_matrix(pred.argmax(axis=1), testLabel.argmax(axis=1)) #, labels=Classes
print("Confusion matrix : ")
print(con_matrix)


# Print accuracy and detection rate
#acc = accuracy_score(y_pred=pred, y_true=testLabel)
acc = accuracy_score(pred.argmax(axis=1), testLabel.argmax(axis=1))
print("Accuracy score on test data is : %f" % acc)

sumDr = 0
for i in range(con_matrix.shape[0]):
    det_rate = 0
    for j in range(con_matrix.shape[1]):
        if i != j :
            det_rate += con_matrix[i][j]
    if con_matrix[i][i] != 0 or (det_rate + con_matrix[i][i])  != 0:
        det_rate =100* con_matrix[i][i]/(det_rate + con_matrix[i][i])
        sumDr += det_rate
        print("For " + classesName[i] + ", Detection Rate is % " + str(det_rate))
    else:
        print("For " + classesName[i] + ", Detection Rate is % 0")

DR = sumDr/classesCount
print("DR is % " + str(DR))
