import os
import sys
import pandas as pd
import json


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration

def convert_dataset(dataset_to_import=0):

    config = Configuration(dataset_to_import=dataset_to_import)

    df15: pd.DataFrame = import_txt(config.bmx055_HRS_gyr, 'hrs_gyr')
    df16: pd.DataFrame = import_txt(config.txt16, 'txt16')
    df17: pd.DataFrame = import_txt(config.txt17, 'txt17')
    df18: pd.DataFrame = import_txt(config.txt18, 'txt18')
    df19: pd.DataFrame = import_txt(config.txt19, 'txt19')

def import_txt(filename: str, prefix: str):
    with open(filename) as f:
        print(f)
        fobj = open(filename, "r")
        line = fobj.readline()
        conv = ''
        count = 0
        if line[0:2] != '[[':
            for l in fobj:
                i = l.find("]")
                if i == (-1):
                    conv = conv + l
                else:
                    count = (count+1)
                    conv = conv + (l[0:i+1] + ',' + l[i+1:(len(l) - 1)])
                line = l
        if count > 1:
            conv = '['+ conv[0:len(conv)-1]+ ']'
            name = filename[0:(len(filename) - 4)] + '_conv.txt'
            fobj_out = open(name, 'w')
            fobj_out.write(conv)
            with open(name) as n:
                content = json.load(n)
        else:
            content = json.load(f)


def to_jason(line, filename, fobj):
    name = filename[0:(len(filename)-4)] + '_conv.txt'
    fobj_out = open(name, 'w')

    for l in fobj:
        fobj_out.write('[' + line)

        i = line.find("]")
        if i == (-1):
            fobj_out.write(line)
        else:
            fobj_out.write(line[0:i] + ',' + line[i:(len(line)-1)])
        line = l
    fobj_out.write(line + ']')
    fobj_out.close()
    with open(name) as n:
        content = json.load(n)
    return content


def main():
    config = Configuration()
    nbr_datasets = len(config.datasets)
    print(nbr_datasets)

    for i in range(0, nbr_datasets):
        print('-------------------------------')
        print('Importing dataset', i)
        print('-------------------------------')

        convert_dataset(i)

        print('-------------------------------')
        print()


if __name__ == '__main__':
    main()