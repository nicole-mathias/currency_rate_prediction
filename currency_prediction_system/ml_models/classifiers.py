import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

models = {
    'knn': KNeighborsClassifier(),
    'naive_bayes': GaussianNB(),
    'logistic': LogisticRegression()
}

hyperparams = {
    'knn': {'n_neighbors': [3, 5, 7]},
    'naive_bayes': {},
    'logistic': {'C': [0.1, 1, 10]}
}

def prepare_dataset(stockm_file, economic_file):
    stock_data = pd.read_csv(stockm_file)
    economic_data = pd.read_csv(economic_file)

    merged_data = pd.merge(stock_data, economic_data, left_on='Date', right_on='date')

    # Create categorical variable which tells whether the market moves up or down on that dat
    merged_data['market_movement'] = merged_data.apply(lambda row: 'Up' if row['Close'] >= row['Open'] else 'Down', axis=1)

    X = merged_data.drop(['Date', 'date', 'market_movement'], axis=1)
    y = merged_data['market_movement']
    
    # Clean data
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)
    
    X = StandardScaler().fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


def train_models(X_train, y_train):
    best_models = {}
    for model_name in models:
        # Fit model whhilst performing grid search
        grid_search = GridSearchCV(models[model_name], hyperparams[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_

        print("Best Parameters : ", grid_search.best_params_)

    return best_models

def evaluate_model(model, model_name, X_test, y_test, plot_roc=False):
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='Up')

    # ROC Curve
    if plot_roc:
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'area = {roc_auc:.2f}')
        plt.savefig(f'/content/{model_name}_roc.png')

    # Equal Opportunity Difference
    best_t_idx = (tpr - fpr).argmax()
    positive_tpr = tpr[best_t_idx]
    negative_tpr = 1 - fpr[best_t_idx]
    eod = positive_tpr - negative_tpr
    print(f"{model_name} Equal Opportunity Difference : {eod:.2f}")

    # Confusion matrix
    cf = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=model.classes_)
    disp.plot()

    plt.savefig(f'./plots/{model_name}.png')
    plt.show()


def main():
    X_train, y_train, X_test, y_test = prepare_dataset('./datasets/stock_market/nikkei_225.csv', './datasets/new_combined_clean.csv')

    best_models = train_models(X_train, y_train)
    
    evaluate_model(best_models['knn'], 'knn', X_test, y_test)
    evaluate_model(best_models['naive_bayes'], 'naive_bayes', X_test, y_test)
    evaluate_model(best_models['logistic'], 'logistic', X_test, y_test, True)

if __name__ == '__main__':
    main()