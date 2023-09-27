        # -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:23:32 2021

@author: tomas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score

# for Box-Cox Transformation
from scipy import stats

# reading csv files
data_2018 = pd.read_csv("data/Example_DATASET.csv", na_values=['NA'])
data_2019 = pd.read_csv("data/Example_2019_price_var.csv", na_values=['NA'])

# named undefined column with "Company Name"
data_2018.columns.values[0] = "Comp. Name"
data_2019.columns.values[0] = "Comp. Name"

# merging data
all_data = pd.merge(data_2018, data_2019, on='Comp. Name', how='inner')

# Create correlation matrix
corr_matrix = all_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
all_data.drop(to_drop, axis=1, inplace=True)

# correlation matrix visualization
# =============================================================================
# heatMap=sb.heatmap(corr_matrix, annot=True,  cmap="YlGnBu", annot_kws={'size':12})
# heatmap=plt.gcf()
# heatmap.set_size_inches(120,120)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# =============================================================================

# drop unnecessary columns
all_inputs = all_data.drop(['Comp. Name', '2019 PRICE VAR [%]'], axis='columns')
# print(all_inputs)
all_labels = all_data['class'].values
# print(all_labels)

# scale data
# scaler = preprocessing.MinMaxScaler()
# scale_data = pd.DataFrame(scaler.fit_transform(all_inputs.values), columns=all_inputs.columns, index=all_inputs.index)


# Show plots of the data to compare the measurement distributions of the classes
# for column1 in scale_data.columns[2:]:
#     print("\n" + column1)
#     sb.violinplot(x='class', y=column1, data=scale_data)
#     plt.show()
    
    
# standardization
train_split, test_split = train_test_split(all_inputs, test_size=0.25, random_state=1, stratify=all_inputs['class'])
x_training = train_split.iloc[:, :-1].values
y_training = train_split.iloc[:, -1].values
x_testing = test_split.iloc[:, :-1].values
y_testing = test_split.iloc[:, -1].values

standardize = preprocessing.StandardScaler()
standardize.fit(x_training)
x_training = standardize.fit_transform(x_training)
x_testing = standardize.fit_transform(x_testing)

# decision tree
decision_tree_classifier = DecisionTreeClassifier(random_state=1)

parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': ["auto", "sqrt", "log2"],
                  "criterion" : ["gini", "entropy"],
                  "splitter" : ["best", "random"]}

cross_validation = StratifiedKFold(n_splits=10)

grid_search = GridSearchCV(decision_tree_classifier,
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            scoring="precision_weighted")

grid_search.fit(x_training, y_training)
dt_classifier = grid_search.best_estimator_

print(53 * '=')
print("TRAINING")
predict_dt_train = dt_classifier.predict(x_training)
print('Precision score: {}'.format(precision_score(y_training, predict_dt_train)))
print('\nConfusion Matrix: ')
print(confusion_matrix(y_training, predict_dt_train, labels=np.unique(predict_dt_train)))
print('\nClassification Report: ')
print(classification_report(y_training, predict_dt_train, labels=np.unique(predict_dt_train)))


print(53 * '=')
print("TESTING")
predict_dt_test = dt_classifier.predict(x_testing)
print('Precision score: {}'.format(precision_score(y_testing, predict_dt_test)))
print('Best parameters: {}'.format(grid_search.best_params_))
print('\nConfusion Matrix: ')
print(confusion_matrix(y_testing, predict_dt_test, labels=np.unique(predict_dt_test)))
print('\nClassification Report: ')
print(classification_report(y_testing, predict_dt_test, labels=np.unique(predict_dt_test)))


# Neural Newtwok Model
# An advantage of the ANN algorithm is that it can process a large amount of data.
mlpc = MLPClassifier(random_state=1, early_stopping=False)


tuned_parameters = {'hidden_layer_sizes': [(30, 15, 7), (30, 15), (30,) ],
                    'activation': ['identity', 'logistic','tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'learning_rate': ['constant','adaptive', 'invscaling'],
                    'alpha': [0.0001, 0.05],
                    'max_iter': [1000]}

grid_search = GridSearchCV(mlpc, 
                    tuned_parameters,
                    scoring='precision_weighted',
                    n_jobs=-1,
                    cv=10)


grid_search.fit(x_training, y_training)
mlp_classifier = grid_search.best_estimator_

print(53 * '=')
print("TRAINING")
predict_mlpc_train = mlp_classifier.predict(x_training)
print('Precision score: {}'.format(precision_score(y_training, predict_mlpc_train)))
print('Best parameters: {}'.format(grid_search.best_params_))
print('\nConfusion Matrix: ')
print(confusion_matrix(y_training, predict_mlpc_train, labels=np.unique(predict_mlpc_train)))
print('\nClassification Report: ')
print(classification_report(y_training, predict_mlpc_train, labels=np.unique(predict_mlpc_train)))


print(53 * '=')
print("TESTING")
predict_mlpc_test = mlp_classifier.predict(x_testing)
print('Precision score: {}'.format(precision_score(y_testing, predict_mlpc_test)))
print('Best parameters: {}'.format(grid_search.best_params_))
print('\nConfusion Matrix: ')
print(confusion_matrix(y_testing, predict_mlpc_test, labels=np.unique(predict_mlpc_test)))
print('\nClassification Report: ')
print(classification_report(y_testing, predict_mlpc_test, labels=np.unique(predict_mlpc_test)))


# classifiers = {}
# classifiers['Decision Trees'] = dtgs.best_estimator_
# classifiers['Neural Networks'] = mlpgs.best_estimator_
# classifiers['Support Vector Machine'] = svmgs.best_estimator_
# classifiers['K Nearest Neighbours'] = knngs.best_estimator_
# clf_df = pd.DataFrame()
# for name, clf in classifiers.items():
#     clf_df = clf_df.append(pd.DataFrame({'precision_weighted': cross_val_score(clf, inputs_train, training_classes, cv=StratifiedKFold(n_splits=10)),
#                        'classifier': [name] * 10}))

# ax = sb.boxplot(x='classifier', y='precision_weighted', data=clf_df)
# ax.set_title('Classifiers Accuracy')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right')


# for column1 in x_training:
#     print("\n" + column1)
#     sb.violinplot(x=x_training['class'], y=column1, data=x_training)
#     plt.show()
    
# Show plots of the normalize data to compare the measurement distributions of the classes
# for column_index, column in enumerate(all_inputs.columns):
#     if column == 'class':
#         continue
    
#     # get the index of all positive pledges (Box-Cox only takes postive values)
#     index_positive_pledges = all_inputs[column] > 0
    
#     # get only positive pledges (using their indexes)
#     positive_pledges = all_inputs[column].loc[index_positive_pledges]
#     aux_class = all_inputs["class"].loc[index_positive_pledges]
    
#     # normalize the pledges (w/ Box-Cox)
#     normalized_pledges = stats.boxcox(positive_pledges)[0]
    
#     sb.violinplot(x=aux_class, y=normalized_pledges)
#     plt.show()

# for column1 in all_data.columns[2:]:
#     print("\n" + column1)
#     sb.stripplot(x=all_data["class"], y=all_data[column1], data=all_data)
#     plt.show()




#  grid visualization
# grid_visualization = grid_search.cv_results_['mean_test_score']
# grid_visualization.shape = (5, 4)
# sb.heatmap(grid_visualization, cmap='Blues', annot=True)
# plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
# plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'])
# plt.xlabel('max_features')
# plt.ylabel('max_depth')
