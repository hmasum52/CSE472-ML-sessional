# author: Hasan Masum(1805052)
# pip install scikit-learn
# pip install pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


# create custom metrics functions
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    return np.sum(y_true * y_pred) / np.sum(y_pred)

def recall(y_true, y_pred):
    return np.sum(y_true * y_pred) / np.sum(y_true)

def specificity(y_true, y_pred):
    return np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true)

def false_discovery_rate(y_true, y_pred):
    return 1 - precision(y_true, y_pred)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def scale_data(X_train, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled


# custom logistic regression class
class MyLogisticRegression:

    def __init__(self, n_features, 
                        lr=0.05, 
                        n_iters=2000, 
                        threshold=0,
                        show_loss=False):
        self.n_features = n_features
        self.lr = lr
        self.n_iters = n_iters
        self.weights = np.random.randn(n_features+1)
        # Early terminate Gradient Descent if error in the training set becomes less than threshold
        self.threshold = threshold 
        self.show_loss = show_loss

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _information_gain(self, X, y, feature):
        original_entropy = self._entropy(y)

        # Get the values and counts for the feature
        values, counts = np.unique(X[:, feature], return_counts=True)

        # Calculate the remainder
        remainder = 0
        for value, count in zip(values, counts):
            remainder += count / counts.sum() * self._entropy(y[X[:, feature] == value])

        # Calculate the information gain
        info_gain = original_entropy - remainder
        return info_gain

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _cost(self, X, y):
        epsilon = 1e-10
        y_pred = self._sigmoid(X @ self.weights)
        cost = -(np.dot(y,np.log(y_pred+epsilon)) + np.dot((1 - y) , np.log(1 - y_pred+epsilon)) ) / len(y)
        return cost

    def _gradient(self, X, y):
        y_pred = self._sigmoid(X @ self.weights)
        gradient = (X.T @ (y_pred - y)) / len(y)
        return gradient

    def fit(self, X, y):
        # check shape
        if X.shape[0] != y.shape[0]:
            raise ValueError("shape of X and y do not match")

        # check shape len
        if len(X.shape) != 2:
            raise ValueError("X must be 2 dimensional")

        # print(X.shape, y.shape)

        # Calculate information gain for each feature
        info_gains = [self._information_gain(X, y, feature) for feature in range(X.shape[1])]

        # Get the indices of the features sorted by information gain
        indices = np.argsort(info_gains)[::-1]

        # Select the top n features
        self.selected_features = indices[:self.n_features]

        # freate a new array with only the selected features
        X = X[:, self.selected_features]

        # print(X.shape, y.shape)

        # Add column for bias
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # apply gradient descent
        for i in range(self.n_iters):
            self.weights -= self.lr * self._gradient(X, y)
            loss = self._cost(X, y)
            if self.show_loss:
                print(f"epoch {i+1}, loss: {loss}")
            # early terminate if mse is less than threshold
            if loss < self.threshold:
                break
        
        # print(self._cost(X,y))


    def predict(self, X):

        if self.selected_features is None:
            raise Exception("model must be trained before prediction")

        # select the features
        X = X[:, self.selected_features]
        
        # Add column for bias
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # predict
        y_pred = self._sigmoid(X @ self.weights)

        # convert probabilities to 0 or 1
        y_pred = np.round(y_pred).astype(int)
        return y_pred
        


def report(y_true, y_pred):
    print(f"accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"recall: {recall(y_true, y_pred):.4f}")
    print(f"specificity: {specificity(y_true, y_pred):.4f}")
    print(f"precision: {precision(y_true, y_pred):.4f}")
    print(f"fdr: {false_discovery_rate(y_true, y_pred):.4f}")
    print(f"f1: {f1(y_true, y_pred):.4f}")


class AdaBoost:
    def __init__(self, num_classifiers, n_features=10, threshold=0):
        self.num_classifiers = num_classifiers
        self.n_features = n_features
        self.threshold = threshold
        self.alphas = None
        self.classifiers = None

    def resample(self, X, y, weights):
        indices = np.random.choice(len(X), len(X), p=weights)
        return X[indices], y[indices]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples
        alphas = []
        classifiers = []

        for k in range(self.num_classifiers):
            X_resampled, y_resampled = self.resample(X, y, weights)
            classifier = MyLogisticRegression(
                n_features=self.n_features, n_iters=1000, threshold=self.threshold)
            classifier.fit(X_resampled, y_resampled)

            predictions = classifier.predict(X)
            error = np.sum(weights * (predictions != y))

            if error > 0.5:
                continue

            for i in range(n_samples):
                if predictions[i] == y[i]:
                    weights[i] *= error/(1-error)

            alpha = np.log((1 - error) / error)
           # weights = weights * np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            alphas.append(alpha)
            classifiers.append(classifier)

        self.alphas = np.array(alphas)
        self.classifiers = classifiers

    def predict(self, X):
        predictions = np.zeros(len(X))

        # normalize alpha
        self.alphas = self.alphas/np.sum(self.alphas)

        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * classifier.predict(X)

    
        predictions = (predictions >= 0.5).astype(int)

        # print(f"hi {len(predictions[predictions < 0])}")
        # print(predictions[predictions==0])
        # print(predictions[predictions==1])

        return predictions

    def weighted_majority(self, X):
        return self.predict(X)



def run_model(X_train, X_test, y_train, y_test, n, n_adaboost=20):
    # create the model
    # model = MyLogisticRegression(n_features=X_train.shape[1])
    print("Running Logistic Regression")
    n = int(X_train.shape[1] * 0.8)
    model = MyLogisticRegression(n_features=n, show_loss=False, threshold=0)

    # train the model
    model.fit(X_train, y_train)

    # predict train data
    y_pred = model.predict(X_train)
    print("train data")
    report(y_train, y_pred)

    print()

    # predict test data
    y_pred = model.predict(X_test)
    print("test data")
    report(y_test, y_pred)

    print()

    print("Running AdaBoost")

    K = [5, 10, 15, 20]
    for k in K:
        print(f"num_classifiers: {k}")
        model = AdaBoost(num_classifiers=k, n_features=n_adaboost, threshold=0.5)
        model.fit(X_train, y_train)

        # predict train data
        y_pred = model.predict(X_train)
        print(f"train: k={k}, accuracy: {accuracy(y_train, y_pred):.4f}")

        # predict test data
        y_pred = model.predict(X_test)
        print(f"test: k={k}, accuracy: {accuracy(y_test, y_pred):.4f}\n")


def preprocess_telco_customer_churn_dataset():
    # dataset 1: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/
    csv_path = "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)

    # drop customerID
    df.drop('customerID', axis=1, inplace=True)

    columns = df.columns
    # drop customer id, tenure, monthly charges, total charges
    columns = columns.drop(['tenure', 'MonthlyCharges', 'TotalCharges'])

    # preprocess data
    for column in columns:
        if df[column].dtype == 'object' and column != 'Churn':
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)

    # convert churn to 0 or 1
    df['Churn'] = df['Churn'].astype('category').cat.codes

    # convert total charges to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # drop rows with missing values
    df.dropna(inplace=True)

    # split the data into 80% training and 20% testing using sklearn
    # churn is the target
    X = df.drop(['Churn'], axis=1).values
    y = df['Churn'].values

    # split the data into 80% training and 20% testing using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test


def preprocess_creditcard_dataset():
    # read credit dataset from datasets/creditcard.csv
    all_data_df = pd.read_csv('datasets/creditcard.csv')
    # all_data_df.info()

    # take all the rows with class 1
    fraud_df = all_data_df[all_data_df['Class'] == 1]

    # take 20000 rows with class 0
    non_fraud_df = all_data_df[all_data_df['Class'] == 0].sample(20000)

    df = pd.concat([fraud_df, non_fraud_df])

    X = df.drop(['Class'], axis=1).values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # scale the data
    X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test

def preprocess_adult_dataset():
    # import adult dataset from datasets/adult folder
    # https://archive.ics.uci.edu/ml/datasets/adult

    # preprocess the data
    # convert categorical data to numerical data
    # split the data into 80% training and 20% testing using sklearn
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income']

    train_df = pd.read_csv('datasets/adult/adult.data', names=columns)
    train_df['train'] = 1

    test_df = pd.read_csv('datasets/adult/adult.test', names=columns, skiprows=1)
    test_df['train'] = 0

    # concatenate train and test data
    df = pd.concat([train_df, test_df])

    # replace icome with 0 or 1
    df['income'] = df['income'].str.replace('.', '')
    df['income'].unique()

    # education and education-num are the same
    # drop education
    df.drop('education', axis=1, inplace=True)

    # 
    df['income'] = df['income'].astype('category').cat.codes

    # convert categorical data to numerical data
    for column in columns:
        if column in [
            'workclass', 'education-num', 
            'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country'  ]:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)

    # train test split
    train_df = df[df['train'] == 1]
    test_df = df[df['train'] == 0]

    X_train = train_df.drop(['income', 'train'], axis=1).values
    y_train = train_df['income'].values


    X_test = test_df.drop(['income', 'train'], axis=1).values
    y_test = test_df['income'].values

    # scale data
    X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test




# main function
if __name__ == "__main__":
    
    # get args from command line
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="all", help="dataset to use")
    args = parser.parse_args()

    if args.dataset == "telco":
        print("-----telco_customer_churn_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_telco_customer_churn_dataset()
        n = int(X_train.shape[1] * 0.8)
        run_model(X_train, X_test, y_train, y_test, n, X_train.shape[1])
    elif args.dataset == "adult":
        print("\n-----adult_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_adult_dataset()
        n = 20
        run_model(X_train, X_test, y_train, y_test, n, n)
    elif args.dataset == "credit":
        print("\n-----creditcard_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_creditcard_dataset()
        n = int(X_train.shape[1] * 0.8)
        run_model(X_train, X_test, y_train, y_test, n, X_train.shape[1])
    else:
        print("-----telco_customer_churn_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_telco_customer_churn_dataset()
        n = int(X_train.shape[1] * 0.8)
        run_model(X_train, X_test, y_train, y_test, n, X_train.shape[1])

        print("\n-----adult_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_adult_dataset()
        n = 20
        run_model(X_train, X_test, y_train, y_test, n, n)

        print("\n-----creditcard_dataset-----")
        X_train, X_test, y_train, y_test = preprocess_creditcard_dataset()
        n = int(X_train.shape[1] * 0.8)
        run_model(X_train, X_test, y_train, y_test, n, X_train.shape[1])

