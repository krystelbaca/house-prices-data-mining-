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

def numero_banios_influye_precio(data):
    
    numbBath = data['bathrooms'].value_counts()
    numbBathKeys = numbBath.keys()

    priceArray = []
    for number in numbBathKeys:
        subset = data.loc[data['bathrooms'] == number]
        print('Numero de banios:' + str(number))
        print(subset['price'])
        priceArray.append(subset["price"].mean())

    print(numbBathKeys)
    print(priceArray)

    width = .2
    plt.bar(numbBathKeys, priceArray, width, color="blue")

    plt.ylabel('precio')
    plt.xlabel('#banios')
    plt.title('banios inlfuye precio')
    plt.xticks(np.arange(0, max(numbBathKeys), .5))
    plt.yticks(np.arange(0, 60000, 5000))
    

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
    
    #numero_banios_influye_precio(data)
    lotArea_influye_salePrice(data)
    
    #create_histogram(data)
    #create_density_plot(data)
    #create_whisker_plots(data)