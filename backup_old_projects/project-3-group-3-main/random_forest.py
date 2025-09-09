import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def rf_funct():
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


    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'criterion': ['gini', 'entropy']
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf_classifier, param_grid, cv = 10, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # printing the best hyperparameters, which will be used for testing
    print("Best Hyperparameters:", best_params)

    # Train the Random Forest model with the best hyperparameters on the training data
    best_rf = RandomForestClassifier(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = best_rf.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions,labels=best_rf.classes_)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate and display the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test.apply(lambda x: 1 if x == 'Up' else 0), best_rf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Determine the point on the ROC curve where the classifier lies
    best_threshold_index = (tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_index]
    best_tpr = tpr[best_threshold_index]

    # Print the results
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Classifier Threshold Point - False Positive Rate: {fpr[best_threshold_index]:.2f}, True Positive Rate: {best_tpr:.2f}")

    # Calculate the Equal Opportunity Difference (EOD)
    positive_group_tpr = best_tpr
    negative_group_tpr = 1 - fpr[best_threshold_index]
    equal_opportunity_difference = positive_group_tpr - negative_group_tpr
    print(f"Equal Opportunity Score: {equal_opportunity_difference:.2f}")


    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=best_rf.classes_)
    disp.plot()
    plt.savefig(os.getcwd() + '/plots/random_forest_cm.png')
    plt.show()

if __name__ == "__main__":
    rf_funct()