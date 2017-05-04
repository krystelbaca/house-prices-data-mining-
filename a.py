from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import logging

logger = logging.getLogger(__name__)

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(1314))

    #print("Numerical Information:\n")
    #numerical_info = data.iloc[:, :data.shape[1]]
    #print(numerical_info.describe())

def get_feature_subset(data, *args):
    featureDict = []
    for arg in args:
        featureDict.append(arg)

    subset = data[featureDict]

    return subset

def delete_column(data, *args):
    for arg in args:
        data = data.drop(arg, 1)

    return data

def delete_missing_objects(data, type):
    type = 0 if type == 'instance' else 1

    data = data.dropna(axis = type)

    return data

def replace_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp

    return data

    plt.show()

def drop_garage_features(data):
    dtd = data.drop('GarageYrBlt', 1)
    dtd = dtd.drop('GarageType', 1)
    dtd = dtd.drop('GarageFinish', 1)
    dtd = dtd.drop('GarageQual', 1)
    dtd = dtd.drop('GarageCond', 1)
    return dtd

def replace_mv_poolqc(data):
    data['PoolQC'] = data['PoolQC'].fillna('NoPool')
    return data

def replace_mv_fence(data):
    data['Fence'] = data['Fence'].fillna('NoFence')
    return data

def replace_missing_values_with_constant(data):
    data['LotFrontage'] = data['LotFrontage'].fillna(-1)
    return data

def replace_missing_values_with_constant_alley(data):
    data['Alley'] = data['Alley'].fillna('None')
    return data

def replace_mv_sotano(data):
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('NoBsmt')

    return data

def replace_mv_fireplaces(data):
    data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFP')
    return data

def replace_mv_misc(data):
    data['MiscFeature'] = data['MiscFeature'].fillna('NoMisc')
    return data

def replace_mv_MasVnrType(data):
    data['MasVnrType'] = data['MasVnrType'].fillna('NA')
    return data

def replace_mv_MasVnrArea(data):
    data['MasVnrArea'] = data['MasVnrArea'].fillna(-1)
    return data
def replace_missing_values_electrical(data):
    data['Electrical'] = data['Electrical'].fillna("NA")
    return data


def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        print(numpy_data[:,i])
        dict = np.unique(numpy_data[:,i])
        print("---------------------------------------------")
        print(i)
        print(dict)
        print("---------------------------------------------")
        if type(dict[0]) == str:
            for j in range(len(dict)):
                temp[np.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:,i] = temp
    return numpy_data

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def create_whisker_plot(data):
    print(data.size)
    data.plot(kind='box', subplots=True, layout=(3,13), sharex=False, sharey=False)
    plt.show()

def z_score_normalization(data):
    # import data
    """num_features = len(data.columns) - 1
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))
    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])
    features = data[list(range(1, num_features))]
    target = data[[num_features]]"""

    features = data[:, 0:-1]
    target = data[:, -1]

    # First 10 rows
    print('Training Data:\n\n' + str(features))
    print('\n')
    print('Targets:\n\n' + str(target))

    # Data standarization
    standardized_data = preprocessing.scale(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])
    print('\n\n')

    new_data = np.append(standardized_data, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def principal_components_analysis(data, n_components):
    features = data[:, 0:-1]
    target = data[:, -1]
    X = features
    Y = target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(X)

    # Model transformation
    new_feature_vector = pca.transform(X)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    return new_feature_vector

def attribute_subset_selection_with_trees(data, type):
    features = data[:, 0:-1]
    target = data[:, -1]
    feature_vector = features
    targets = target

    """
    This function use gain info to select the best features of a numpy array
    :param type: Classification or Regression
    :param feature_vector: Numpy array to be transformed
    :param targets: feature vector targets
    :return: Numpy array with the selected features
    """

    if type == 'Regression':
        extra_tree = ExtraTreesRegressor()
    else:
        # Model declaration
        extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(feature_vector, targets)

    # Model information:
    logger.debug('Model information:')

    # display the relative importance of each attribute
    logger.debug('Importance of every feature: %s', extra_tree.feature_importances_)

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit=True)

    # Model transformation
    new_feature_vector = model.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])

    return new_feature_vector


def min_max_scaler(data):
    features = data[:, 0:-1]
    target = data[:, -1]
    X = features
    Y = target
    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(X)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    print("---------------------- TERMINA NORMALIZACION")
    return new_feature_vector

def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
    :param data: numpy array
    :return: four subsets that represents data train and data test
    """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test

def decision_tree_training(data,targets):

    feature_vector = data[:, 0:-1]
    #targets = data[:, -1]



    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(feature_vector,
                         targets,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))


if __name__ == '__main__':
    filePath = "training_data.csv"

    data = open_file(filePath)
    targets = np.array(data)[:, -1]
    #PRIMERA ITERACION
    #temp = drop_garage_features(data)
    #temp = replace_missing_values_with_constant(temp)
    #temp = replace_missing_values_with_constant_alley(temp)
    '''temp = replace_mv_fence(temp)
    temp = replace_mv_poolqc(temp)
    temp = replace_mv_sotano(temp)
    temp = replace_mv_misc(temp)
    temp = replace_mv_fireplaces(temp)
    temp = replace_mv_MasVnrType(temp)
    temp = replace_mv_MasVnrArea(temp)
    temp = replace_missing_values_electrical(temp)'''
    #temp['MSSubClass'] = reject_outliers(temp['MSSubClass'])
    #temp['OverallQual'] = reject_outliers(temp['OverallQual'])
    #temp['OverallCond'] = reject_outliers(temp['OverallCond'])
    #temp['1stFlrSF'] = reject_outliers(temp['1stFlrSF'])
    #temp['2ndFlrSF'] = reject_outliers(temp['2ndFlrSF'])
    #temp['BsmtFullBath'] = reject_outliers(temp['BsmtFullBath'])

    #temp = convert_data_to_numeric(temp)
    #temp = z_score_normalization(temp)
    #temp = principal_components_analysis(temp, .90)
    #decision_tree_training(temp)

    # Segunda ITERACION
    temp = drop_garage_features(data)
    temp = replace_missing_values_with_constant(temp)
    temp = replace_missing_values_with_constant_alley(temp)
    temp = replace_mv_fence(temp)
    temp = replace_mv_poolqc(temp)
    temp = replace_mv_sotano(temp)
    temp = replace_mv_misc(temp)
    temp = replace_mv_fireplaces(temp)
    temp = replace_mv_MasVnrType(temp)
    temp = replace_mv_MasVnrArea(temp)
    temp = replace_missing_values_electrical(temp)
    # temp['MSSubClass'] = reject_outliers(temp['MSSubClass'])
    # temp['OverallQual'] = reject_outliers(temp['OverallQual'])
    # temp['OverallCond'] = reject_outliers(temp['OverallCond'])
    # temp['1stFlrSF'] = reject_outliers(temp['1stFlrSF'])
    # temp['2ndFlrSF'] = reject_outliers(temp['2ndFlrSF'])
    # temp['BsmtFullBath'] = reject_outliers(temp['BsmtFullBath'])'''

    temp = convert_data_to_numeric(temp)
    temp = min_max_scaler(temp)
    temp = attribute_subset_selection_with_trees(temp, "Regression")
    decision_tree_training(temp, targets)

    # Tercera ITERACION
    '''temp = drop_garage_features(data)
    temp = replace_missing_values_with_constant(temp)
    temp = replace_missing_values_with_constant_alley(temp)
    temp = replace_mv_fence(temp)
    temp = replace_mv_poolqc(temp)
    temp = replace_mv_sotano(temp)
    temp = replace_mv_misc(temp)
    temp = replace_mv_fireplaces(temp)
    temp = replace_mv_MasVnrType(temp)
    temp = replace_mv_MasVnrArea(temp)
    temp = replace_missing_values_electrical(temp)
    # temp['MSSubClass'] = reject_outliers(temp['MSSubClass'])
    # temp['OverallQual'] = reject_outliers(temp['OverallQual'])
    # temp['OverallCond'] = reject_outliers(temp['OverallCond'])
    # temp['1stFlrSF'] = reject_outliers(temp['1stFlrSF'])
    # temp['2ndFlrSF'] = reject_outliers(temp['2ndFlrSF'])
    # temp['BsmtFullBath'] = reject_outliers(temp['BsmtFullBath'])

    temp = convert_data_to_numeric(temp)
    temp = z_score_normalization(temp)
    temp = attribute_subset_selection_with_trees(temp, "Regression")
    decision_tree_training(temp)'''

