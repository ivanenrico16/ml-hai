# classification mlp model
from numpy import unique
from numpy import argmax
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

df_train=read_csv('train.csv')
dataset = df_train.values
X, y = dataset[:, 1:-1], dataset[:, -1]

print("------------------")
print(X)
print("------------------")
print(y)

#Change to float, create variable that save number of features (input) 
# and create variable that save number of class (output)
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]
y = LabelEncoder().fit_transform(y)
n_class = len(unique(y))

print("Print again after change to float and labeling the output")
print("------------------")
print(X)
print("------------------")
print(y)

print("------------------")
print(f"n_features {n_features}")
print("------------------")
print(f"n_class {n_class}")

df_test=read_csv('test.csv')
dataset_test = df_test.values
X_test, y_test = dataset_test[:, 1:-1], dataset_test[:, -1]
X_test, y_test = X_test.astype('float'), y_test.astype('float')
y_test = LabelEncoder().fit_transform(y_test)

print("Print Test variables")
print("------------------")
print(X_test)
print("------------------")
print(y_test)

# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(n_class, activation='softmax'))
# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=32, verbose=2)

# evaluate on test set
yhat = model.predict(X_test)
yhat = argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)

f1 = f1_score(y_test, yhat, average='weighted')
print('F1 score: %f' % f1)