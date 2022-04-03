# This project predicts concrete compressive strengths 

import warnings
warnings.filterwarnings('ignore')


# Import Libraries

from numpy import mean,std
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score,RepeatedKFold,RandomizedSearchCV
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,VotingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# Importing dataset
# Importing or Loading dataset
data = "C:/Users/Gabriel/Desktop/Concrete/concrete_dataset.csv"
df = pd.read_csv(data, delimiter=",")
# Inspect data
print('Inspect Data')
print('Normal Data Shape:',df.shape)
#print(df.head(50).to_string())
df = df.sample(frac=0.7)
df = df.sort_index()
#print(df.head(50).to_string())
# Check new data Shape
print('New data shape:',df.shape)


# Statistical Summary
print(df.describe())

# Data Types
print(df.info())

# Each feature summary
for i in df:
    print(df[i].describe())

# Check Missing Values: To delete columns having missing values more than 30% or to input values--------
# Check missing values
df3 = df.isnull().sum()
print('Missing values in each feature \n:-------------------------------')
print(df3)


# Check feature relevance to the target through correlation matrix
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


df_corr = df.corr()
print('Feature Correlation Table')
print(df_corr)

# Through inspection, the concrete compressive strength dataset is a multivariate dataset, having float and integer feature values
# with no missing values. Each feature has different scale, looking at minimum and maximum values for each of variables.
# To obtain a better scale, it is good to normalize the data because it makes distributions better.

# Separate feature vectors from target labels and convert to numpy arrays
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# Put X and y in pandas dataframe to view them
X = pd.DataFrame(X)
y = pd.DataFrame(y)
print(X.count())
print(y.value_counts())

# Visualize the data using Histogram plots
# plot the histograms of all features or variable in the data
X.hist(sharex=False, sharey=False,  xlabelsize=1, ylabelsize=1, figsize=(6,6))
plt.show()
# Note: In the plots, The shape of the each graph can be Gaussian’, skewed or even has an exponential distribution.
# Density plots
X.plot(kind='density', subplots=True, layout=(5,5), sharex=False, legend=True, fontsize=1, figsize=(8,8))
plt.show()



# Convert the Dataframe to Numpy Arrays
X = X.values
y = np.ravel(y.values)


# Check Outliers-------------------------------------------
# identify outliers using Isolation Forest in the training dataset
# The data has been transformed between a range of 0 and 1. So there is no need to check for outliers
# identify outliers using Isolation Forest in the training dataset
iso = IsolationForest(contamination=0.1)
# Contamination argument is used to help estimate the number of outliers in the dataset.
# This is a value between 0.0 and 0.5 and by default is set to 0.1.
outl = iso.fit_predict(X)
# select all rows that are not outliers
remove_outl = outl != -1
X, y = X[remove_outl, :], y[remove_outl]
# summarize the shape of the updated training dataset
print('New data without outliers \n')
print(X.shape, y.shape)


# Performing feature normalization or standardization-----------------------------------
scale = MinMaxScaler()# The MinMax Scaler normalizes the data to a range 0 and 1.
X = scale.fit_transform(X)
#print(X)


# DATA PREPARATION ENDS HERE---------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

print('\n')
# MODEL DEVELOPMENT BEGINS
print('# MODEL DEVELOPMENT BEGINS')
# Cross validation of 10 folds and 5 runs
cv_method = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# Hyperparameter Optimization using RandomSearch and CrossValidation to get the best model hyperparamters

# naive Bayes Classifier: Naive Bayes algorithm is not considered for regression because it is basically a classification
# algorithm. Hence, it is excluded as part of the base learners used for regression problems

# kNN Classifier
nearest_neighbour = KNeighborsRegressor()
# Create a dictionary of KNN parameters
# K values between 1 and 9 are used to avoid ties and p values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski)
param_kNN = {'n_neighbors': [1,3,5,7,9],'p':[1,2,5]} # Distance Metric: Manhattan (p=1), Euclidean (p=2) or
# Minkowski (any p larger than 2). Technically p=1 and p=2 are also Minkowski distances.
# Define the kNN model using RandomSearch and optimize accuracy
kNN_grid = RandomizedSearchCV(nearest_neighbour,param_kNN,scoring='r2',cv=cv_method)
kNN_grid.fit(X_train,y_train)
# Print the best parameter values for KNN
print('kNN Best Parameter values =',kNN_grid.best_params_)
#kNN = KNeighborsRegressor(**kNN_grid.best_params_)
kNN = kNN_grid.best_estimator_


# Decision Tree Classifier
Decision_Tree = DecisionTreeRegressor()
# Create a dictionary of DT hyperparameters
params_DT = {'criterion':['mse','mae'],
             'max_depth':[1,2,3,4,5,6,7,8],
             'splitter':['best','random']}

# Using Random Search to explore the best parameter for the a decision tree model
DT_Grid = RandomizedSearchCV(Decision_Tree,params_DT,scoring='r2',cv=cv_method)
# Fitting the parameterized model
DT_Grid.fit(X_train,y_train)
# Print the best parameter values
print('DT Best Parameter Values:', DT_Grid.best_params_)
#DT = DecisionTreeRegressor(**DT_Grid.best_params_)
DT = DT_Grid.best_estimator_

# Support Vector Machines
SVM_clasf = SVR()
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_SVR = {'kernel':['rbf'],'C':np.linspace(0.1,1.0),
              'gamma':['scale','auto']} #np.linspace(0.1,1.0)}

# Using Random Search to explore the best parameter for the a SVM model
SVR_Grid = RandomizedSearchCV(SVM_clasf,params_SVR,scoring='r2',cv=cv_method)
# Fitting the parameterized model
SVR_Grid.fit(X_train,y_train)
# Print the best parameter values
print('SVR Best Parameter Values:', SVR_Grid.best_params_)
#SVR = SVR(**SVR_Grid.best_params_)
SVR = SVR_Grid.best_estimator_


# Neural Network
mlp = MLPRegressor()
parameter_MLP = {
    'hidden_layer_sizes': [(25,25,25),(50,50,50)],
    'activation': ['relu','tanh'],
    'solver': ['adam'],'max_iter':[500,1000],
    'learning_rate': ['constant','adaptive']}

mlp_Grid = RandomizedSearchCV(mlp, parameter_MLP, scoring='r2',cv=cv_method)
mlp_Grid.fit(X_train, y_train) # X is train samples and y is the corresponding labels

# Check best hyperparameters
print('ANN Best parameter values:\n', mlp_Grid.best_params_)
#MLP = MLPRegressor(**mlp_Grid.best_params_)
MLP = mlp_Grid.best_estimator_
print('\n')

#for bagsize in range(5,41,5):
#    print("Ensemble Results for Bag Size of ", bagsize, "\n")
for feature_size in np.arange(2, 18, 2):
    print("Ensemble Results for Feature Size of ", feature_size, "\n")
    # Developing homogeneous ensembles of each classifier
    kNN_ensemble = BaggingRegressor(base_estimator=kNN, bootstrap=False, max_features=feature_size)
    DT_ensemble = BaggingRegressor(base_estimator=DT, bootstrap=False, max_features=feature_size)
    Rand_forest = RandomForestRegressor(bootstrap=False, max_features=feature_size)
    SVM_ensemble = BaggingRegressor(base_estimator=SVR, bootstrap=False, max_features=feature_size)
    MLP_ensemble = BaggingRegressor(base_estimator=MLP, bootstrap=False, max_features=feature_size)


    def get_HTRGN_ensemble():
        models = list()
        models.append(('kNN_ensemble', kNN_ensemble))
        models.append(('DT_ensemble', DT_ensemble))
        models.append(('RF', Rand_forest))
        models.append(('SVM_ensemble', SVM_ensemble))
        models.append(('MLP_ensemble', MLP_ensemble))
        HTE = VotingRegressor(estimators=models)
        return HTE

    # get a list of models to evaluate
    def get_models():
        models = dict()
        models['kNNR_HE'] = kNN_ensemble
        models['DTR_HE'] = DT_ensemble
        models['RF'] = Rand_forest
        models['SVR_HE'] = SVM_ensemble
        models['ANNR_HE'] = MLP_ensemble
        models['HTE'] = get_HTRGN_ensemble()
        return models


    # Cross validate the models
    def evaluate_model(model, X_train, y_train):
        # evaluate the model and collect the results
        scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv_method, n_jobs=-1)
        return scores

    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    print('Cross Validation R-squared values of each ensemble on test set:------------------------------------------------------')
    for name, model in models.items():
        scores = evaluate_model(model, X_test, y_test)
        results.append(scores)
        names.append(name)
        print('>%s %.3f' % (name, mean(scores)), u"\u00B1", '%.3f' % std(scores))

    # plot model performance for comparison
    plt.figure()
    plt.boxplot(results, labels=names, showfliers=False)
    #plt.title('Cross validation R-squared comparison of ensembles')
    plt.xlabel("Ensembles")
    plt.ylabel("R-squared score of Ensembles")
    plt.savefig("concrete_without_featuresize_{}".format(feature_size))
    #plt.show()
    print('\n')

    print('Cross validation R-squared values of ensemble on train set:----------------------------------------------------')
    for name, model in models.items():
        # evaluate the model
        scores = evaluate_model(model, X_train, y_train)
        # store the results
        results.append(scores)
        names.append(name)
        # summarize the performance along the way
        print('>%s %.3f' % (name, mean(scores)), u"\u00B1", '%.3f' % std(scores))

    print('\n')
    # Train and evaluate each Ensemble
    for name,model in models.items():
        # fit the model
        model.fit(X_train,y_train)
        # then predict on the test set
        y_pred= model.predict(X_test)
        # Evaluate the models
        mse_test = mean_squared_error(y_test, y_pred)
        root_mse_test = np.sqrt(mse_test)
        r2_test= r2_score(y_test,y_pred)
        print('Performance Result of',name,':-----------------------------------------------------------------')
        print(name, 'root mean squared error of test set:', root_mse_test)
        print(name, 'r_squared coefficient of test set:', r2_test)
        y_pred1 = model.predict(X_train)
        mse_train = mean_squared_error(y_train,y_pred1)
        root_mse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_pred1)
        print(name, 'root mean squared error of train set:', root_mse_train)
        print(name, 'r_squared coefficient of train set:',r2_train)
        gen_factor = mse_test/mse_train
        print('Generalization Factor to determine Ensemble Overfitting', gen_factor)
        # NOTE: if the gen_factor > 1, then the ensemble overfits else it is desirable
        print('\n')

        # Evaluate Bias-Variance Tradeoff
        avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(model, X_train, y_train
                                                                            , X_test, y_test, loss='mse',
                                                                            num_rounds=10,
                                                                            random_seed=20)
        # Summary of Results
        print('Average Expected loss for', name, '%.2f' % avg_expected_loss2)
        print('Average Expected Bias error for', name, '%.2f' % avg_bias2)
        print('Average Expected Variance error for', name, '%.2f' % avg_variance2)
        print('\n')
