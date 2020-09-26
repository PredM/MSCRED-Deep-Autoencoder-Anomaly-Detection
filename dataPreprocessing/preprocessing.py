import sys
import os
import json
import gc
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from pandas.plotting import register_matplotlib_converters

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath("."))
print(os.path.abspath("."))
print(sys.path)
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

#parentdir = os.path.dirname(currentdir)

#sys.path.insert(0,parentdir)
from configuration.Configuration import Configuration


# global variables
# start_time to define the start of a data sequence
start_time = ''
# interval to define the interval for the fusion process
interval = 0

stats = ''


# creation of a dataframe with the start value, which is defined at the config.json
# the connection of the data frame and the raw data enables the fusion process
def start_dataframe():
    global start_time
    start_time = start_time + '0000'
    df = pd.DataFrame({'timestamp': [start_time]})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(df)
    return df


# chances the frequency during time points of a data frame to enable the data fusion
def frequency_df(df):
    global stats
    string = str(df.count().values)
    stats = stats + string[1:len(string) - 2]

    df_start = start_dataframe()

    start = str(df_start.index[0])
    print(start)
    i = start.find(':')
    j = start.find('.')
    if j != (-1):
        if start[j - 2:j] != '59':
            start = start[i - 2:j - 2] + str(int(start[j - 2:j]) + 1)
        elif start[j - 5:j - 3] != '59':
            start = start[i - 2:j - 5] + str(int(start[j - 5:j - 3]) + 1) + ':00'
        elif start[j - 8:j - 6] != '23':
            start = str(int(start[i - 2:i]) + 1) + ':00:00'
        else:
            start = '00:00:00'
    else:
        start = start[i - 2:len(start)]
    end = str(df.index[df.shape[0] - 1])
    print(df)
    i = end.find(':')
    j = end.find('.')
    if j != (-1):
        print(end[j - 2:j])
        if end[j - 2:j] != '59':
            end = end[i - 2:j - 2] + str(int(end[j - 2:j]) + 1)
        elif end[j - 5:j - 3] != '59':
            end = end[i - 2:j - 5] + str(int(end[j - 5:j - 3]) + 1) + ':00'
        elif end[j - 8:j - 6] != '23':
            end = str(int(end[i - 2:i]) + 1) + ':00:00'
        else:
            end = '00:00:00'
    else:
        end = end[i - 2:len(end)]
    print(start)
    print(end)
    df_between = df.between_time(start, end)
    df = df_start.append(df_between, sort=True)

    df = df.loc[~df.index.duplicated(keep='first')]
    print(df)

    # determine the interval to fuse the data
    freq = interval + 'ms'

    # change frequency of single sensors to enable fusion
    fused = df.asfreq(freq=freq, method='ffill')
    return fused


# creation of txt files which can be loaded as a json file
def to_json(filename: str):
    with open(filename) as f:
        fobj = open(filename, "r")
        line = fobj.readline()
        c = 0
        count = 0
        conv = line
        i = conv.find("]")
        if i != (-1) and conv[i + 1] != ",":
            count = (count + 1)
            conv = conv[0:i + 1] + ',' + conv[i + 1:(len(conv) - 1)]
        if line[0:2] != '[[':
            for l in fobj:
                i = l.find("]")
                if i == (-1):
                    conv = conv + l
                else:
                    count = (count + 1)
                    conv = conv + (l[0:i + 1] + ',' + l[i + 1:(len(l) - 1)])

        if count > 1:
            conv = '[' + conv[0:len(conv) - 1] + ']'
            name = filename[0:(len(filename) - 4)] + '_conv.txt'
            fobj_out = open(name, 'w')
            fobj_out.write(conv)
            with open(name) as n:
                content = json.load(n)
        elif count == 1:
            conv = conv[0:len(conv) - 1]
            name = filename[0:(len(filename) - 4)] + '_conv.txt'
            fobj_out = open(name, 'w')
            fobj_out.write(conv)
            with open(name) as n:
                content = json.load(n)
        else:
            content = json.load(f)
        return content


# importation of txt files and transformation into a fused data frame
def import_txt(filename: str, prefix: str):
    # load from file and transform into a json object
    content = to_json(filename)

    # transform into data frame
    df = pd.DataFrame.from_records(content)

    # special case for txt controller 18 which has a sub message containing the position of the crane
    # split position column into 3 columns containing the x,y,z position
    if '18' in prefix:
        pos = df['currentPos'].apply(lambda x: dict(eval(x.strip(','))))
        df['vsg_x'] = pos.apply(lambda r: (r['x'])).values
        df['vsg_y'] = pos.apply(lambda r: (r['y'])).values
        df['vsg_z'] = pos.apply(lambda r: (r['z'])).values
        df = df.drop('currentPos', axis=1)

    # add the prefix to every column except the timestamp
    prefix = prefix + '_'
    df = df.add_prefix(prefix)
    df = df.rename(columns={prefix + 'timestamp': 'timestamp'})

    # Fix wrong data types
    if '15' in prefix:
        transformFinishedFromStringToNumeric("txt15_m1.finished", df)
    elif '16' in prefix:
        transformFinishedFromStringToNumeric("txt16_m1.finished", df)
        transformFinishedFromStringToNumeric("txt16_m2.finished", df)
        transformFinishedFromStringToNumeric("txt16_m3.finished", df)
    elif '17' in prefix:
        transformFinishedFromStringToNumeric("txt17_m1.finished", df)
        transformFinishedFromStringToNumeric("txt17_m2.finished", df)
    elif '18' in prefix:
        transformFinishedFromStringToNumeric("txt18_m1.finished", df)
        transformFinishedFromStringToNumeric("txt18_m2.finished", df)
        transformFinishedFromStringToNumeric("txt18_m3.finished", df)
        df["txt18_currentTask"] = (df["txt18_currentTask"]).astype('category').cat.codes
        df["txt18_currentTask"] = pd.to_numeric(df["txt18_currentTask"])
        transformFinishedFromStringToNumeric("txt18_isContainerReady", df)
    elif '19' in prefix:
        transformFinishedFromStringToNumeric("txt19_m1.finished", df)
        transformFinishedFromStringToNumeric("txt19_m2.finished", df)
        transformFinishedFromStringToNumeric("txt19_m3.finished", df)
        transformFinishedFromStringToNumeric("txt19_m4.finished", df)
        df["txt19_currentTask"] = (df["txt19_currentTask"]).astype('category').cat.codes
        df["txt19_currentTask"] = pd.to_numeric(df["txt19_currentTask"])
        transformFinishedFromStringToNumeric("txt19_getState", df)
        df["txt19_currentTask"] = (df["txt19_getSupply"]).astype('category').cat.codes
        df["txt19_getSupply"] = pd.to_numeric(df["txt19_getSupply"])
        df["txt19_isContainerReady"] = (df["txt19_isContainerReady"]).astype('category').cat.codes
        df["txt19_isContainerReady"] = pd.to_numeric(df["txt19_isContainerReady"])

    # Remove lines with duplicate timestamps, keep first appearance
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    # determine the timestamp as datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    pd.set_option('display.expand_frame_repr', False)
    # print(df.describe(include='all'))
    pd.set_option('display.expand_frame_repr', True)
    print(prefix + str(df.shape))
    df = frequency_df(df)

    return df


# transformation of boolean attributes into numeric values
def transformFinishedFromStringToNumeric(attributeToTransoform, df):
    try:
        df[attributeToTransoform] = df[attributeToTransoform].replace("True", 1)
        df[attributeToTransoform] = df[attributeToTransoform].replace("False", 0)
        df[attributeToTransoform] = pd.to_numeric(df[attributeToTransoform])
    except:
        print("could not convert: " + attributeToTransoform)


# plot of the txt data
def plot_export_txt(df: pd.DataFrame, file_name: str, config: Configuration):
    if not (config.export_plots or config.plot_txts):
        return

    print('Creating plot for ' + file_name)
    df = df.query(config.query)
    df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)

    xmarks = df.index.values[::3000]
    plt.xticks(xmarks)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=200)
    if config.plot_txts:
        plt.show()


# start of the import of the raw data from the TXT controllers
def import_all_txts(config: Configuration):
    print('Importing TXT controller data')

    # import the single txt sensors
    df15: pd.DataFrame = import_txt(config.txt15, 'txt15')
    df16: pd.DataFrame = import_txt(config.txt16, 'txt16')
    df17: pd.DataFrame = import_txt(config.txt17, 'txt17')
    df18: pd.DataFrame = import_txt(config.txt18, 'txt18')
    df19: pd.DataFrame = import_txt(config.txt19, 'txt19')

    # plot the data if enabled
    plot_export_txt(df15, 'txt_15', config)
    plot_export_txt(df16, 'txt_16', config)
    plot_export_txt(df17, 'txt_17', config)
    plot_export_txt(df18, 'txt_18', config)
    plot_export_txt(df19, 'txt_19', config)

    # df15 = df15.query(config.query)

    # combine into a single dataframe
    df_txt = df15.join(df16, how='outer')
    df_txt = df_txt.join(df17, how='outer')
    df_txt = df_txt.join(df18, how='outer')
    df_txt = df_txt.join(df19, how='outer')

    df_txt.query(config.query, inplace=True)

    return df_txt


# import and transformation of the txt files from the pressure sensors
def import_single_pressure_sensor(content, c_name: str, module):
    record_list = []

    # add prefixes to message components
    for i in content:
        temp = i[c_name]
        temp['hPa_' + module] = temp.pop('hPa')
        temp['tC_' + module] = temp.pop('tC')
        temp['timestamp'] = i['meta']['time']
        record_list.append(temp)

    df = pd.DataFrame.from_records(record_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index(df['timestamp'])
    df.drop('timestamp', 1, inplace=True)
    print(c_name + module + str(df.shape))
    df = frequency_df(df)
    return df


# start of the import of the raw data from the pressure sensors
def import_pressure_sensors(config: Configuration):
    print('\nImporting pressure sensors')
    content = to_json(config.topicPressureSensorsFile)

    # import the single components of the message
    df_pres_18 = import_single_pressure_sensor(content, 'VSG', '18')
    df_pres_17 = import_single_pressure_sensor(content, 'Oven', '17')
    df_pres_15 = import_single_pressure_sensor(content, 'Sorter', '15')

    # combine into a single data frame
    df_sensor_data = df_pres_18.merge(df_pres_17, left_on='timestamp', right_on='timestamp', how='inner')
    df_sensor_data = df_sensor_data.merge(df_pres_15, left_on='timestamp', right_on='timestamp', how='inner')

    df_sensor_data.query(config.query, inplace=True)
    # df_sensor_data.drop('timestamp', 1, inplace=True)

    if config.plot_pressure_sensors or config.export_plots:
        df_sensor_data.plot(subplots=True, sharex=True, figsize=(20, 20), title="Pressure Sensors")

        xmarks = df_sensor_data.index.values[::7500]
        plt.xticks(xmarks)

        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/pressure_sensors.png', dpi=200)
        if config.plot_pressure_sensors:
            plt.show()

    # df_sensor_data = frequency_df(df_sensor_data)
    return df_sensor_data


# start of the import of the raw data from the acc sensors
def import_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # load from file and transform into a json object
    content = to_json(filename)

    entry_list = []

    # extract single messages and add prefixes to the message entries
    for m in content:
        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')

            # partly different naming
            if 'timestamp' not in e.keys():
                e['timestamp'] = e.pop('time')

            entry_list.append(e)

    df = pd.DataFrame.from_records(entry_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    print(prefix + str(df.shape))
    fused_df = frequency_df(df)
    return fused_df


# plot of the acc sensors
def plot_export_acc(df: pd.DataFrame, file_name: str, config: Configuration):
    df = df.set_index('timestamp')
    df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=200)
    if config.plot_acc_sensors:
        plt.show()


# import and transformation of the txt files from the scc sensors
def import_acc_sensors(config: Configuration):
    print('\nImport acceleration sensors')

    # import each acceleration sensor
    acc_txt15_m1 = import_acc(config.acc_txt15_m1, 'a_15_1')
    acc_txt15_comp = import_acc(config.acc_txt15_comp, 'a_15_c')
    acc_txt16_m3 = import_acc(config.acc_txt16_m3, 'a_16_3')
    acc_txt18_m1 = import_acc(config.acc_txt18_m1, 'a_18_1')

    # plot if enabled
    if config.plot_acc_sensors or config.export_plots:
        plot_export_acc(acc_txt15_m1, 'acc_txt15_m1.png', config)
        plot_export_acc(acc_txt15_comp, 'acc_txt15_comp.png', config)
        plot_export_acc(acc_txt16_m3, 'acc_txt16_m3.png', config)
        plot_export_acc(acc_txt18_m1, 'acc_txt18_m1.png', config)

    # combine into a single data frame
    df_accs = acc_txt15_m1.join(acc_txt15_comp, how='outer')
    df_accs = df_accs.join(acc_txt16_m3, how='outer')
    df_accs = df_accs.join(acc_txt18_m1, how='outer')
    # acc_txt15_m1['timestamp'] = pd.to_datetime(acc_txt15_m1['timestamp'])
    # df_accs = acc_txt15_m1.set_index('timestamp').join(acc_txt15_comp.set_index('timestamp'), how='outer')
    # df_accs = df_accs.join(acc_txt16_m3.set_index('timestamp'), how='outer')
    # df_accs = df_accs.join(acc_txt18_m1.set_index('timestamp'), how='outer')
    df_accs.query(config.query, inplace=True)

    return df_accs


# start of the import of the raw data from the acc sensors
def import_bmx_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # load from file and transform into a json object
    content = to_json(filename)
    # with open(filename) as f:
    #    content = json.load(f)

    entry_list = []

    # extract single messages and add prefixes to the message entries
    for m in content:
        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')
            e[prefix + '_t'] = e.pop('t')
            if 'timestamp' not in e.keys():
                e['timestamp'] = e.pop('time')
            entry_list.append(e)

    # transform into a data frame and return
    df = pd.DataFrame.from_records(entry_list)

    df = df.loc[~df['timestamp'].duplicated(keep='first')]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    print(prefix + str(df.shape))
    fused_df = frequency_df(df)
    return fused_df


# import and transformation of the txt files from the bmx sensors
def import_bmx_sensors(config: Configuration):
    print('\nImport bmx sensors')
    # all datasets dont contain the hrs acceleration sensors data, so some lines needed to be changed

    # import single components
    # df_hrs_acc = import_bmx_acc(config.bmx055_HRS_acc, 'hrs_acc')
    df_hrs_gyr = import_acc(config.bmx055_HRS_gyr, 'hrs_gyr')
    df_hrs_mag = import_acc(config.bmx055_HRS_mag, 'hrs_mag')

    # combine into a single data frame
    df_hrs = df_hrs_gyr.join(df_hrs_mag, how='outer')
    # df_hrs_gyr['timestamp'] = pd.to_datetime(df_hrs_gyr['timestamp'])
    # df_hrs = df_hrs_gyr.set_index('timestamp').join(df_hrs_mag.set_index('timestamp'), how='outer')
    # df_hrs_gyr['timestamp'] = pd.to_datetime(df_hrs_gyr['timestamp'])
    # df_hrs = df_hrs.set_index('timestamp').join(df_hrs_mag.set_index('timestamp'), how='outer')
    df_hrs.query(config.query, inplace=True)

    # import single components
    df_vsg_acc = import_bmx_acc(config.bmx055_VSG_acc, 'vsg_acc')
    df_vsg_gyr = import_acc(config.bmx055_VSG_gyr, 'vsg_gyr')
    df_vsg_mag = import_acc(config.bmx055_VSG_mag, 'vsg_mag')

    # combine into a single dataframe
    df_vsg = df_vsg_acc.join(df_vsg_gyr, how='outer')
    df_vsg = df_vsg.join(df_vsg_mag, how='outer')
    # df_vsg_acc['timestamp'] = pd.to_datetime(df_vsg_acc['timestamp'])
    # df_vsg = df_vsg_acc.set_index('timestamp').join(df_vsg_gyr.set_index('timestamp'), how='outer')
    # df_vsg = df_vsg.join(df_vsg_mag.set_index('timestamp'), how='outer')
    df_vsg.query(config.query, inplace=True)

    # plot if enabled
    if config.plot_bmx_sensors or config.export_plots:
        df_vsg.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX VSG")
        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/bmx_vsg.png', dpi=100)
        if config.plot_acc_sensors:
            plt.show()

        df_hrs.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX HRS")
        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/bmx_hrs.png', dpi=100)
        if config.plot_acc_sensors:
            plt.show()

    return df_hrs, df_vsg


# debugging method to check data frame for duplicates
def check_duplicates(df):
    df = df.loc[df.index.duplicated(keep=False)]
    print('Current index dublicates:', df.shape)


# start the import of the raw data
def import_dataset(dataset_to_import=0):
    register_matplotlib_converters()

    config = Configuration(dataset_to_import=dataset_to_import)

    # import each senor type
    df_txt_combined = import_all_txts(config)
    df_press_combined = import_pressure_sensors(config)
    df_accs_combined = import_acc_sensors(config)

    # bmx sensor are not used because of missing data in some datasets
    df_hrs_combined, df_vsg_combined = import_bmx_sensors(config)
    gc.collect()

    if not (config.save_pkl_file or config.plot_all_sensors or config.print_column_names):
        sys.exit(0)

    print("\nCombine all data frames...")
    print("Step 1/4")
    df_combined: pd.DataFrame = df_txt_combined.join(df_press_combined, how='outer')
    print("Step 2/4")
    df_combined = df_combined.join(df_accs_combined, how='outer')
    print("Step 3/4")
    df_combined = df_combined.join(df_hrs_combined, how='outer')
    print("Step 4/4")
    df_combined = df_combined.join(df_vsg_combined, how='outer')

    # del df_vsg_combined, df_hrs_combined
    del df_press_combined, df_accs_combined
    gc.collect()

    print('\nGarbage collection executed')

    print('\nDelete unnecessary streams')
    print('Number of streams before:', df_combined.shape)
    # df_combined = df_combined.drop(config.unused_attributes, 1, errors='ignore')
    try:
        # df_combined = df_combined[config.featuresBA]
        df_combined = df_combined[config.featuresAll]
        # df_combined = df_combined[config.all_features_configured]
    except:
        raise AttributeError('Relevant feature not found in current dataset.')

    print('Number of streams after:', df_combined.shape)

    print('\nSort streams by name to ensure same order like live data')
    df_combined = df_combined.reindex(sorted(df_combined.columns), axis=1)

    print('Delete NaN Values at the beginning')
    df_combined = df_combined.dropna()
    print(df_combined)

    if config.save_pkl_file:
        print('\nSaving data frame as pickle file in', config.pathPrefix)
        df_combined.to_pickle(config.pathPrefix + config.filename_pkl)
        print('Saving finished')

    if config.print_column_names:
        print(*list(df_combined.columns.values), sep="\n")

    # not tested, formatting not lucid
    if config.plot_all_sensors:
        # Interpolate and add missing data
        print('Interpolate data for plotting')
        df_combined = df_combined.apply(pd.Series.interpolate, args=('linear',))
        # df_combined.plot(subplots=True, sharex=True, figsize=(10,10))
        df_combined = df_combined.fillna(method='backfill')

        # ax = plt.gca()

        # df_combined.plot(subplots=True, sharex=True, figsize=(10,10), linewidth=0.5, legend=False)
        print('Creating full plot')
        df_combined.plot(subplots=True, sharex=True, figsize=(40, 40),
                         title="All sensors")
        plt.show()

    print('\nImporting of datset', dataset_to_import, 'finished')


# import for all datasets which are configured in the config.json
def main():
    config = Configuration()
    nbr_datasets = len(config.datasets)
    print(nbr_datasets)

    for i in range(0, nbr_datasets):
        print('-------------------------------')
        print('Importing dataset', i)
        print('-------------------------------')
        if (config.defined_interval == True):
            global start_time
            start_time = config.datasets[i][1]
            global interval
            interval = config.interval
            import_dataset(i)

        print('-------------------------------')
        print()
    global stats
    with open(os.path.abspath(".") + '/data/datasets/' + 'test.csv', 'w') as wf:
        wf.write(stats)


if __name__ == '__main__':
    main()
