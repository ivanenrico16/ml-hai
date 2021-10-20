Steps: (Run this Program in Jupyter Notebook or Google Collab)
1. Import all of required libraries
2. Load and Split the data training set into input(X) and output(y) variables. 
X variable consists all of data from column 2-10 and the y variable consists all of data from column 11.
Note if you are using Jupyter Notebook, please run row number 2. However if you are using Google collab please
run row number 3
3. Change variable X and y into float, create variable that save number of features (input),
transform y variable with Label Encoder and create variable that save number of class (output)
4. Load and Split the data testing set into input(X_test) and output(y_test) variables. 
5. Change variable X_test and y_test into float datatype, and transform y_test with LabelEncoder
6. Define the keras Model 
- Input layer -> 9 nodes just like the number of column of X variable (store it in n_features variable)
	Column Details: 
	1. Acceleration value from accelerometer 1 in x axis. (acc1_x)
	2. Acceleration value from accelerometer 1 in y axis. (acc1_y)
	3. Acceleration value from accelerometer 1 in z axis. (acc1_z)
	4. Gyro value from gyroscope 1 in x axis. (gyro1_x)
	5. Gyro value from gyroscope 1 in y axis. (gyro1_y)
	6. Gyro value from gyroscope 1 in z axis. (gyro1_z)
	7. Acceleration value from accelerometer 2 in x axis. (acc2_x)
	8. Acceleration value from accelerometer 2 in y axis. (acc2_y)
	9. Acceleration value from accelerometer 2 in z axis. (acc2_z)
- First layer -> 20 nodes (activation: "ReLu", kernel_initializer: "he_normal")
- Second layer -> 10 nodes (activation: "ReLu", kernel_initializer: "he_normal")
- Output Layer -> 6 nodes (activation: "ReLu", kernel_initializer: "he_normal"), 
there are 6 classes in y(output) variables:
	1. Walking activity
	2. Walking up the stairs activity
	3. Walking down the stairs activity
	4. Sitting activity
	5. Standing activity
	6. Laying down activity
7. Predict the model with test set (X_test variable), check the accuracy score by comparing 
the predict variable (yhat) with output test variables (y_test)
8. check the f1 score. Parameters (y_test -> real output from dataset, yhat -> predict output, average:"weighted")