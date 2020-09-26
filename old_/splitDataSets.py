import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))

from configuration.Configuration import Configuration

def unpickle_data(path):
    print('-------------------------------')
    print('Unpickle dataset')
    print('-------------------------------')
    # read the imported dataframe from the saved file
    df: pd.DataFrame = np.load(path)
    return df

def main():
    config = Configuration()
    nbr_datasets = len(config.datasets)
    print(nbr_datasets)

    for i in range(0, nbr_datasets):
        path_to_file = os.path.abspath(".") + config.datasets[i][0][2::] + 'matrix_data/'
        for file in os.listdir(path_to_file):
            matrix = unpickle_data(path_to_file + file)
            length = matrix.shape[0]
            i = 0
            while i < length:
                if (length > (i+9000)):
                    new_matrix = matrix[i:(i+9000)]
                else:
                    new_matrix = matrix[i:length]
                path_to_directory = path_to_file + 'start' + str(i) + '/'
                if not os.path.exists(path_to_directory):
                    os.makedirs(path_to_directory)
                print(path_to_directory)
                np.save(path_to_directory + file, new_matrix)
                i = i + 4000


if __name__ == '__main__':
    main()