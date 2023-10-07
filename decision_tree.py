import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset into a pandas dataframe
data = pd.read_csv('medical_data.csv')

# Preprocess data to select relevant features and encode categorical variables
X = data[['age', 'gender', 'symptom_1', 'symptom_2', 'symptom_3']]
y = data['diagnosis']
X = pd.get_dummies(X, columns=['gender', 'symptom_1', 'symptom_2', 'symptom_3'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix to get the percentages
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a heatmap to visualize the accuracy
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix heatmap (Accuracy = {:.2f}%)'.format(accuracy*100))
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))
