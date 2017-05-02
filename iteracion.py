
import pandas as pd
import numpy



def replace_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp
    return data

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0])); #shape da la forma de la medida de los atributos y las instancias [0] -> numero de instancias
    print("Number of fetures: " + str(
        data.shape[1]));  # shape da la forma de la medida de los atributos y las instancias [1] -> numero de campos
    print("----------------------------------------------------------------")
    print(data.head(10)) #Te muestra los titulos de cada una de las caracteristicas y un número de instancias
    print("Atributos numericos:")
    numerical_info = data.iloc[:, : data.shape[1]]
    print(numerical_info.describe())

    # """
    # This function returns four subsets that represents training and test data
    # :param data: numpy array
    # :return: four subsets that represents data train and data test
    # """
def data_splitting(data_features, data_targets, test_size):
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)
    return data_features_train, data_features_test, data_targets_train, data_targets_test

def decision_tree_training(data_without_target, target):
    data_features = data_without_target
    data_targets = target

    #Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        data_splitting(data_features, data_targets, 0.25)

    #Model declaration
    """
    Parameters to select:
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """
    dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dec_tree.fit(data_features_train, data_targets_train)
    #Model evaluation
    test_data_predicted = dec_tree.predict(data_features_test)
    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)
    print("Model error: " + str(error))
    print("Probability of each class: \n")
    #Measure probability of each class
    prob_class = dec_tree.predict_proba(data_features_test)
    print(prob_class)
    print("Feature Importance: \n")
    print(dec_tree.feature_importances_)

if _name_ == '_main_':

    # PRIMERA ITERACIÓN
    # CSV que salio de la limpieza de datos que hice en mongo
    # data = pd.read_csv('cleanReadyForPca.csv')

    # TUVE QUE SEPARAR EL TARGET DEL DATASET PARA LOS ALGORITMOS Y LA NORMALIZACIÖN
    # target = data['SalePrice']
    # data_without_target = data.drop("SalePrice", 1)

    # LO QUE HICE PARA SUSITTUIR LOS NAN QUE SALIAN
    # data_without_target = replace_missing_values_with_constant(data, "Alley", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MasVnrType", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtQual", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtCond", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtExposure", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtFinType1", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtFinType2", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "Electrical", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MiscFeature", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "FireplaceQu", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageYrBlt", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageQual", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageFinish", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageCond", "NA")
    #
    # data_only_numeric = convert_data_to_numeric(data_without_target)
    #
    # data_normalizado = z_score_normalization(data_only_numeric, target)
    # data_after_pca = principal_components_analysis(data_normalizado, target, .90)
    #
    # decision_tree_training(data_after_pca, target)

    # Segunda iteración
    # CSV que salio de la limpieza de datos que hice en mongo
    data = pd.read_csv('cleanReadyForPcaSegundaIteracion.csv')

    # TUVE QUE SEPARAR EL TARGET DEL DATASET PARA LOS ALGORITMOS Y LA NORMALIZACIÖN
    target = data['SalePrice']
    data_without_target = data.drop("SalePrice", 1)

    data_without_target = replace_missing_values_with_constant(data, "Alley", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtExposure", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "MiscFeature", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "PoolQC", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "MasVnrArea", -1)

    show_data_info(data_without_target)
    data_only_numeric = convert_data_to_numeric(data_without_target)
    data_normalizado = min_max_scaler(data_only_numeric, target)
    #NO SE UTILIZO PCA POR QUE BORRE MANUALMENTE LAS COLUMNAS EN MONGO
    # data_after_pca = principal_components_analysis(data_normalizado, target, .90)
    decision_tree_training(data_normalizado, target)