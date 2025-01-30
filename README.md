Random Forest Classification and Evaluation

This Python script demonstrates how to use a Random Forest Classifier to perform binary classification on a synthetic dataset, followed by evaluating the model using various metrics. The dataset is created using the make_classification function, and the Random Forest model is trained and tested on the dataset. The evaluation metrics include accuracy, precision, recall, F1 score, and ROC-AUC.

Requirements
To run this script, you'll need the following Python packages:

scikit-learn
numpy (implicitly used by scikit-learn)
You can install them using pip if you don't have them already:

pip install scikit-learn numpy

Script Breakdown
Data Generation:
A synthetic dataset is generated with 1000 samples and 10 features using make_classification from sklearn.datasets.

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
Data Splitting:
The dataset is split into training (80%) and testing (20%) sets using train_test_split.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model Training:
A RandomForestClassifier is created and trained on the training dataset.

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
Prediction:
The trained model predicts the labels of the test data.

y_pred = model.predict(X_test)
Evaluation:
The model's performance is evaluated using the following metrics:

Accuracy: Measures the proportion of correct predictions.

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
Precision: The proportion of true positive predictions out of all positive predictions.

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
Recall: The proportion of true positive predictions out of all actual positive instances.

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")
F1 Score: The harmonic mean of precision and recall.

f1 = f1_score(y_test, y_pred)
print(f"f1: {f1:.2f}")
ROC-AUC: Measures the area under the ROC curve, evaluating the model's ability to discriminate between classes.

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc:.2f}")
Output Example
When you run the script, you will get an output similar to the following:

License
This project is licensed under the MIT License.
