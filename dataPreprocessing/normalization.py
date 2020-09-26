import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

# runable server
sys.path.append(os.path.abspath("."))

from configuration.Configuration import Configuration


min_max = pd.DataFrame()


# calculation of local min and max if necessary and execute the normalisation
def normalise(df, missing, config: Configuration):
    print('-------------------------------')
    print('Save local minimal and maximal values')
    print('-------------------------------')
    global min_max
    extrema = min_max

    # determine local minimal and maximal values for values without extreme levels in the configuration
    min_local = 0
    max_local = 0
    for item in missing:
        min_local = df[item[0]].min()
        max_local = df[item[0]].max()
        extrema[item[0]] = [min_local, max_local]

    # delete attributes which are not used and sort the attributes
    extrema = extrema[config.featuresAll]
    extrema = extrema.reindex(sorted(df.columns), axis=1)
    print('-------------------------------')
    print('-------------------------------')
    print('Normalise Data')
    print('-------------------------------')

    data_normalised = pd.DataFrame()

    # min-max normalisation
    for item in config.featuresAll:
        min_val = float(extrema.filter(items=[item]).values.min())
        max_val = float(extrema.filter(items=[item]).values.max())
        y = np.empty(0)
        for x in df.filter(items=[item]).values:
            j = [(float(x) - min_val) / (max_val - min_val)]
            y = np.append(y, j)

        data_normalised[item] = pd.Series(y)
    print('-------------------------------')
    return data_normalised.reindex(sorted(df.columns), axis=1)


# load the fused data framen
def unpickle_data(path_to_file):
    print('-------------------------------')
    print('Unpickle dataset')
    print('-------------------------------')
    # read the imported dataframe from the saved file
    df: pd.DataFrame = pd.read_pickle(path_to_file)
    print('-------------------------------')
    return df

# calcuation of the minmal and maximal values
def get_min_max(config=Configuration()):
    print('-------------------------------')
    print('Save global minimal and maximal values')
    print('-------------------------------')
    axis = []
    df = pd.DataFrame()
    array = []

    # for attributes of config.zeroOne the minmal value is 0 and maximal value is 1
    for item in config.zeroOne:
        df[item] = ['0', '1']

    # minimal and maximal values from configuration if they are given
    # otherwise append the attribute to a array and use local values
    for item in config.intNumbers:
        if len(item) > 1:
            df[item[0]] = item[1:3]
        else:
            for j in config.featuresAll:
                if j == item[0]:
                    array.append(item)

    # minimal and maximal values from configuration if they are given
    # otherwise append the attribute to a array and use local values
    for item in config.realValues:
        if len(item) > 1:
            df[item[0]] = item[1:3]
        else:
            for j in config.featuresAll:
                if j == item[0]:
                    array.append(item)

    # for attributes of config.bools the minmal value is 0 and maximal value is 1
    for item in config.bools:
        df[item] = ['0', '1']

    print('-------------------------------')
    return df, array


def normalize_data(i, missing, config=Configuration()):
    path_to_file = os.path.abspath(".") + config.datasets[i][0][2::]
    unpickeled_data = unpickle_data(path_to_file + config.filename_pkl)
    print(unpickeled_data)
    normalised_data = normalise(unpickeled_data, missing, config)
    print(normalised_data)
    print('\nSaving data frame as pickle file in', path_to_file)
    normalised_data.to_pickle(path_to_file + config.filename_pkl_normalised)
    print('Saving finished')

# configuration of minimal and maximal values and start of the normalisation for every fused data frame
def main():
    config = Configuration()
    nbr_datasets = len(config.datasets)
    global min_max
    min_max, missing = get_min_max(config)

    for i in range(0, nbr_datasets):
        print('-------------------------------')
        print('Start of normalisation of dataset: ' + str(i))
        print('-------------------------------')
        normalize_data(i, missing, config)
        print('-------------------------------')

if __name__ == '__main__':
    main()
