from numpy import mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Reading the CSV file named 'wineQuality_dataset.csv' into a Pandas DataFrame called wineDataSet
wineDataSet = pd.read_csv('wineQuality_dataset.csv')

#Extracting the features from the DataFrame wineDataSet. It selects all rows except the first one (index 1 onwards) and the first 11 columns.
x = wineDataSet.iloc[1:, :11]

#Extracting the target variable (dependent variable) from the DataFrame wineDataSet. It selects all rows except the first one (index 1 onwards) and only the 12th column.
y = wineDataSet.iloc[1:, 11]

#Mapping the values in the target column ('good' and 'bad') to numerical values (1 and 0) in order to convert it to a binary classification problem
target_column_index = 11
wineDataSet.iloc[:, target_column_index] = wineDataSet.iloc[:, target_column_index].map({'good': 1, 'bad': 0})

#Converting the data types of the variables y and x to integers
y=y.astype('int')
x=x.astype('int')

#Splitting the dataset into Training and Testing sets. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=16)

# Creating a KFold object with 5 splits (folds)
c = KFold(n_splits=3, random_state=2, shuffle=True)


#Analysis using the Decision Tree Classifier

# Creating and Initializing a Decision Tree Classifier
dt_model = DecisionTreeClassifier()
# Training the Decision Tree model on the training data
dt_model.fit(x_train, y_train)
# Making predictions on the test set
dt_prediction = dt_model.predict(x_test)

# Calculating and printing the accuracy of the Decision Tree model on the test set
dt_accuracy = accuracy_score(dt_prediction, y_test)
print("Accuracy for the Decision Tree Model: ", dt_accuracy)

# Performing cross-validation and printing the mean accuracy
dt_crossVal_scores = cross_val_score(dt_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('Mean Accuracy of Cross-Validation Results for the Decision Tree Model: ',mean(dt_crossVal_scores))

# Calculating ROC curve and AUC for the Decision Tree model
dt_fpr, dt_tpr, thresholds = roc_curve(y_test, dt_prediction)
auc_dt = auc(dt_fpr, dt_tpr)


#Analysis using the Logistic Regression Model

# Creating and Initializing a Logistic Regression Model 
lr_model=LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
# Training the Logistic Regression model on the training data
lr_model.fit(x_train,y_train)
# Making predictions on the test set
lr_prediction=lr_model.predict(x_test)

# Calculating and printing the accuracy of the Logistic Regression model on the test set
lr_accuracy = accuracy_score(lr_prediction, y_test)
print("\nAccuracy of the Logistic Regression Model: ", lr_accuracy)

# Performing cross-validation and printing the mean accuracy
lr_crossVal_scores = cross_val_score(lr_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('Mean Accuracy of Cross-Validation Results for the Logistic Regression Model: ',mean(lr_crossVal_scores))

# Calculating ROC curve and AUC for the Logistic Regression Model
lr_fpr, lr_tpr, threshold =roc_curve(y_test,lr_prediction)
auc_lr = auc(lr_fpr, lr_tpr)

lr_confusionMatrix = confusion_matrix(y_test, lr_prediction)
print("Confusion Matrix - \n",lr_confusionMatrix)


#Analysis using the k-Nearest Neighbors Model

# Initialize lists to store k values and corresponding accuracy scores
k_values = []
accuracy_scores = []

# Define a range of k values to test
k_range = range(1, 50)  

# Looping through different values of k
for k in k_range:
    # Creating and Initializing the kNN Model
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Training the kNN model on the training data
    knn_model.fit(x_train, y_train)
    # Making predictions on the test set
    knn_prediction = knn_model.predict(x_test)
    # Calculating and printing the accuracy of the kNN model on the test set
    knn_accuracy = accuracy_score(knn_prediction, y_test)
    k_values.append(k)
    accuracy_scores.append(knn_accuracy)

# Finding the value of k with the highest accuracy
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]

#Repeating the above steps
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_model.fit(x_train, y_train)
knn_prediction = knn_model.predict(x_test)

# Calculating and printing the best accuracy of the kNN model on the test set
knn_accuracy = accuracy_score(knn_prediction, y_test)
print("\nAccuracy of the k-NN Model: ", knn_accuracy)

# Performing cross-validation and printing the mean accuracy
knn_crossVal_scores = cross_val_score(knn_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('Mean Accuracy of Cross-Validation Results for the k-NN Model: ',mean(knn_crossVal_scores))

# Calculating ROC curve and AUC for the KNN Model
knn_fpr, knn_tpr, threshold =roc_curve(y_test,knn_prediction)
auc_knn = auc(knn_fpr, knn_tpr)

# Plotting the accuracy versus k values
plt.figure(figsize=(10, 6))
plt.plot([0] + list(k_values), [0] + accuracy_scores, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. k Value for k-NN')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks([0] + list(k_range))  # Converting k_range to a list
plt.ylim(0, 1)  # Setting the y-axis limit from 0 to 1
plt.grid()
plt.show()



#Analysis using the Gaussian Naive Bayes Model

# Creating and Initializing a Naive Bayes Model 
nb_model = GaussianNB()
# Training the Naive Bayes model on the training data
nb_model.fit(x_train, y_train)
# Making predictions on the test set
nb_prediction = nb_model.predict(x_test)

# Calculating and printing the accuracy of the Naive Bayes model on the test set
nb_accuracy = accuracy_score(y_test, nb_prediction)
print("\nAccuracy Score of the GaussianNB Model: ",nb_accuracy)

# Performing cross-validation and printing the mean accuracy
nb_crossVal_scores = cross_val_score(nb_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('Mean Accuracy of Cross-Validation Results for the Naive Bayes Model: ',mean(nb_crossVal_scores))

# Calculating ROC curve and AUC for the Naive Bayes Model
nb_fpr, nb_tpr, threshold =roc_curve(y_test,nb_prediction)
auc_nb = auc(nb_fpr, nb_tpr)



#Analysis using the Linear Discriminant Analysis Model

# Creating and Initializing a Linear Discriminant Analysis Model 
lda_model = LinearDiscriminantAnalysis()
# Training the Linear Discriminant Analysis model on the training data
lda_model.fit(x_train, y_train)
# Making predictions on the test set
lda_prediction = lda_model.predict(x_test)

# Calculating and printing the accuracy of the Linear Discriminant Analysis model on the test set
lda_accuracy = accuracy_score(y_test, lda_prediction)
print("\nAccuracy Score of the Linear Discriminant Analysis Model: ",lda_accuracy)

# Performing cross-validation and printing the mean accuracy
lda_crossVal_scores = cross_val_score(lda_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('Mean Accuracy of Cross-Validation Results for the Linear Discriminant Analysis Model: ',mean(lda_crossVal_scores))

# Calculating ROC curve and AUC for the Linear Discriminant Analysis Model
lda_fpr, lda_tpr, threshold =roc_curve(y_test,lda_prediction)
auc_lda = auc(lda_fpr, lda_tpr)

lda_confusionMatrix = confusion_matrix(y_test, lda_prediction)
print("Confusion Matrix - \n",lda_confusionMatrix)

#Plotting the ROC-Curve & AUC 
plt.plot(dt_fpr, dt_tpr, linestyle='-', label='Decision Tree (auc=%0.3f)' % auc_dt)
plt.plot(lr_fpr, lr_tpr, marker='.',label='Logistic Regression (auc=%0.3f)' % auc_lr)
plt.plot(knn_fpr, knn_tpr, marker='.',label='KNN (auc=%0.3f)' % auc_knn)
plt.plot(nb_fpr, nb_tpr, marker='.',label='Naive Bayes (auc=%0.3f)' % auc_nb)
plt.plot(lda_fpr, lda_tpr, marker='.',label='Linear Discriminant Analysis (auc=%0.3f)' % auc_lda)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='50-50 Baseline')