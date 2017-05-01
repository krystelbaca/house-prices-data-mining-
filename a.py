import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def create_histogram(data):
    data.hist(column = 'bedrooms')
    plt.show()

def create_density_plot(data):
    data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    plt.show()

def create_whisker_plots(data):
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(10))

    print("Numerical Information:\n")
    numerical_info = data.iloc[:, :data.shape[1]]
    print(numerical_info.describe())

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

def replace_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp

    return data

def replace_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp

    return data

    plt.show()



def lotArea_influye_salePrice(data):
    
    lotArea = data['LotArea'].value_counts()
    lotAreaKeys = lotArea.keys()
    priceArray=[]
    keyArray = []
    for number in lotAreaKeys:
        subset = data.loc[data['LotArea'] == number]
        #print('Area del lote:' + str(number))
        #print(subset['SalePrice'])
        keyArray.append(str(number))
        priceArray.append(subset["SalePrice"].mean())

    #print(numbHabKeys)
    #print(priceArray)
    

    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('precio')
    plt.xlabel('Tamano de terreno')
    plt.title('Tamano terreno influye precio final')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def poolArea_inluye_precio(data):
    pool = data['PoolArea'].value_counts()
    poolKeys = pool.keys()
    priceArray = []
    keyArray = []
    for number in poolKeys:
        subset = data.loc[data['PoolArea'] == number]
        keyArray.append(str(number))
        priceArray.append(subset["SalePrice"].mean())

    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('precio')
    plt.xlabel('Tamano pool')
    plt.title('El tamano de la piscina influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
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
    data['LotFrontage'] = data['LotFrontage'].fillna('-1')
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











if __name__ == '__main__':
    filePath = "train.csv"

    data = open_file(filePath)
    


    #headers = [x for x in data]
    #print(headers)
    #for head in headers:
    #    if head != 'description' and head != 'features' and head != 'photos':
    #        print(data[head].value_counts())
    #print(data.head)
    #show_data_info(data)
    #print(data[0:10])
    

    #lotArea_influye_salePrice(data)
    #poolArea_inluye_precio(data)
    drop_garage_features(data)
    replace_missing_values_with_constant(data)
    replace_missing_values_with_constant_alley(data)
    replace_mv_fence(data)
    replace_mv_poolqc(data)
    replace_mv_sotano(data)
    replace_mv_misc(data)
    replace_mv_fireplaces(data)
    print(data['FireplaceQu'])
    show_data_info(data)
    #print(data['BsmtCond'])

    #create_histogram(data)
    #create_density_plot(data)
    #create_whisker_plots(data)