import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix


def open_file(fileName):
    data = pd.read_csv(fileName)
    return data


def create_histogram(data):
    data.hist(column='LotFrontage')
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

    data = data.dropna(axis=type)

    return data


def replace_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp

    return data

    plt.show()


def lotArea_influye_salePrice(data):
    lotArea = data['LotArea'].value_counts()
    lotAreaKeys = lotArea.keys()
    priceArray = []
    keyArray = []
    for number in lotAreaKeys:
        subset = data.loc[data['LotArea'] == number]
        # print('Area del lote:' + str(number))
        # print(subset['SalePrice'])
        keyArray.append(str(number))
        priceArray.append(subset["SalePrice"].mean())

    # print(numbHabKeys)
    # print(priceArray)


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

def create_whisker_plot(data):
    print(data.size)
    data.plot(kind='box', subplots=True, layout=(3,13), sharex=False, sharey=False)
    plt.show()




if __name__ == '__main__':
    filePath = "train.csv"

    data = open_file(filePath)

    # headers = [x for x in data]
    # print(headers)
    # for head in headers:
    #    if head != 'description' and head != 'features' and head != 'photos':
    #        print(data[head].value_counts())
    # print(data.head)
    # show_data_info(data)
    # print(data[0:10])


    #lotArea_influye_salePrice(data)
    #poolArea_inluye_precio(data)
    #show_data_info(data)
    # print(data['BsmtCond'])
    create_whisker_plot(data)

    # create_histogram(data)
    # create_density_plot(data)
    # create_whisker_plots(data)