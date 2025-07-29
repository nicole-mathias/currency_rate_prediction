import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os


def svm_funct():
    # Load training data from the CSV file
    train_file_path = os.getcwd() + '/datasets/categorical_data/train_data.csv'
    train_df = pd.read_csv(train_file_path)

    # Load testing data from the CSV file
    test_file_path = os.getcwd() + '/datasets/categorical_data/test_data.csv'
    test_df = pd.read_csv(test_file_path)


    X_train = train_df.drop(['market_movement','date'], axis=1)
    y_train = train_df['market_movement']

    X_test = test_df.drop(['market_movement','date'], axis=1)
    y_test = test_df['market_movement']

    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)


    # Define the hyperparameter grid for the SVM
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear','rbf', 'poly'],  # Kernel functions to try
        'degree': [2, 3, 4]  # Degree for 'poly' kernel
    }

    # Create an SVM classifier
    svm_classifier = SVC(probability=True)


    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(svm_classifier, param_grid, cv = 10, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # printing the best hyperparameters, which will be used for testing
    print("Best Hyperparameters:", best_params)

    # Train the SVM model with the best hyperparameters on the training data
    best_svm = SVC(**best_params, probability=True)
    best_svm.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = best_svm.predict(X_test)

    print("predictions",predictions)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions,labels=best_svm.classes_)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate and display the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test.apply(lambda x: 1 if x == 'Up' else 0), best_svm.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)


    # Determine the point on the ROC curve where the classifier lies
    best_threshold_index = (tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_index]
    best_tpr = tpr[best_threshold_index]

    # Print the results
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Classifier Threshold Point - False Positive Rate: {fpr[best_threshold_index]:.2f}, True Positive Rate: {best_tpr:.2f}")


    # Calculate a simplified Equal Opportunity Score
    # Here, we use the TPR for the 'UP' class as the TPR for the positive group
    positive_group_tpr = best_tpr

    # The TPR for the 'Down' class is equivalent to 1 - TNR (True Negative Rate)
    negative_group_tpr = 1 - fpr[best_threshold_index]

    # Calculate the Equal Opportunity Score
    equal_opportunity_score = abs(positive_group_tpr - negative_group_tpr)
    print(f"Equal Opportunity Score: {equal_opportunity_score:.2f}")


    # Display consfusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=best_svm.classes_)
    disp.plot()
    plt.savefig(os.getcwd() + '/plots/svm_cm.png')
    plt.show()


if __name__ == "__main__":
    svm_funct()
