import json

import pandas as pd
import os

class Configuration:

    def __init__(self, dataset_to_import=0):

        # self.load_config_json('../configuration/config.json')
        #self.load_config_json(os.path.abspath(".") + '/configuration/config.json')
        self.load_config_json('../configuration/config.json')

        ###
        # name of the directories and files
        ###

        # data preprocessing
        self.filename_pkl = 'imported_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'
        self.filename_pkl_normalised = 'normalised_data.pkl'

        # create input of NN
        self.filename_matrix_data = "matrix_win_"
        self.split_prefix_filename = 'start'
        self.directoryname_matrix_data = '../data/matrix_data_2_test/'
        self.directoryname_training_data = 'training_data/'
        self.directoryname_eval_data = 'evaluation_data/'
        self.reconstruction_error = 'reconstruction_error.npy'
        self.step_size_reconstruction = 1

        # NN
        self.pretrained_model_name = 'pertrained-model'
        self.directoryname_NNresults = 'resultsNN/'
        self.filename_reconstruction_error = 'reconstruction_error.npy'

        # anomaly detection and diagnosis
        #NotUsed self.filename_diagnosis = 'Anomaly_Detection_Diagnosis.txt'

        ###
        # import and data visualisation
        ###
        '''
        self.plot_txts: bool = False
        self.plot_pressure_sensors: bool = False
        self.plot_acc_sensors: bool = False
        self.plot_bmx_sensors: bool = False
        self.plot_all_sensors: bool = False

        self.export_plots: bool = False

        self.print_column_names: bool = False
        self.save_pkl_file: bool = True
        '''
        ###
        # preprocessing
        ###
        '''
        # define the interval to fuse
        self.defined_interval: bool = True
        self.interval = '10'
        self.fusion_interval = '10'

        # select specific dataset with given parameter
        # preprocessing however will include all defined datasets
        self.pathPrefix = os.path.abspath(".") + str(self.datasets[dataset_to_import][0])[2::]
        #print("self.pathPrefix: ", self.pathPrefix)
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # define file names for all topics
        self.txt15 = self.pathPrefix + 'raw_data/txt15.txt'
        self.txt16 = self.pathPrefix + 'raw_data/txt16.txt'
        self.txt17 = self.pathPrefix + 'raw_data/txt17.txt'
        self.txt18 = self.pathPrefix + 'raw_data/txt18.txt'
        self.txt19 = self.pathPrefix + 'raw_data/txt19.txt'

        self.topicPressureSensorsFile = self.pathPrefix + 'raw_data/pressureSensors.txt'

        self.acc_txt15_m1 = self.pathPrefix + 'raw_data/TXT15_m1_acc.txt'
        self.acc_txt15_comp = self.pathPrefix + 'raw_data/TXT15_o8Compressor_acc.txt'
        self.acc_txt16_m3 = self.pathPrefix + 'raw_data/TXT16_m3_acc.txt'
        self.acc_txt18_m1 = self.pathPrefix + 'raw_data/TXT18_m1_acc.txt'

        self.bmx055_HRS_acc = self.pathPrefix + 'raw_data/bmx055-HRS-acc.txt'
        self.bmx055_HRS_gyr = self.pathPrefix + 'raw_data/bmx055-HRS-gyr.txt'
        self.bmx055_HRS_mag = self.pathPrefix + 'raw_data/bmx055-HRS-mag.txt'

        self.bmx055_VSG_acc = self.pathPrefix + 'raw_data/bmx055-VSG-acc.txt'
        self.bmx055_VSG_gyr = self.pathPrefix + 'raw_data/bmx055-VSG-gyr.txt'
        self.bmx055_VSG_mag = self.pathPrefix + 'raw_data/bmx055-VSG-mag.txt'
        '''
        ###
        # NN
        ###

        # attribute for signature matrix generation

        # maximum step in ConvLSTM
        #self.step_max = 5
        self.step_max = 5
        # gap time between each segment in time steps
        #self.gap_time = 125 #10#
        self.gap_time = 500
        # window size / length of each segment
        #self.win_size = [125, 250, 375, 500, 625, 750, 875, 1000]  # [10, 30, 60]#
        self.win_size = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000] #[10, 30, 60]#
        # train start point
        #NotUsed self.train_start_point = 0
        # train end point
        #NotUsed self.train_end_point = 8000
        # test start point
        #NotUsed self.test_start_point = 8000
        # test end point
        #NotUsed self.test_end_point = 20000
        # define names for txt controler to create filter matrix
        #NotUsed self.controler_names = [['15'], ['16'], ['17'], ['18', 'vsg'], ['19', 'hrs']]
        # length to split matrix
        #NotUsed self.training_length = 9000
        #NotUsed self.training_overlapping = 4000

        # data to train NN and calculate threshold
        self.no_failure = 0
        self.training_data = 0
        self.validation_data = 0

        # NN parameter
        self.batch_size = 128
        self.epochs = 10000
        self.learning_rate = 0.001
        self.length_training = 275818
        self.length_eval = 47657
        self.saver_step = 10
        self.filter_dimension_encoder = [128, 256]#[64, 128, 256, 512]
        self.stride_encoder = [1, 2]#[1, 2, 2, 2]
        self.filter_size_encoder = [[3, 3], [3, 3]]#[[3, 3], [3, 3], [2, 2], [2, 2]]
        self.dimension_lstm = [128, 256]# self.filter_dimension_encoder
        self.filter_dimension_decoder = [128, 32, 3]#[256, 128, 64, 32, 3] #[128, 64, 32, 3]
        self.stride_decoder = [1, 2, 1]
        self.filter_size_decoder = [[3, 3], [3, 3], [1,1]]#[[2, 2], [2, 2], [3, 3], [3, 3], [3, 3]]
        self.filter = False

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.datasets = data['datasets']
        self.prefixes = data['prefixes']
        self.error_descriptions = data['error_descriptions']
        # self.subdirectories_by_case = data['subdirectories_by_case']

        features_all_cases = data['relevant_features']
        # self.featuresBA = data['featuresBA']
        self.featuresAll = data['featuresAll']
        self.cases_used = None
        if self.cases_used is None or len(self.cases_used) == 0:
            self.relevant_features = features_all_cases
        else:
            self.relevant_features = {case: features_all_cases[case] for case in self.cases_used if
                                      case in features_all_cases}

        # sort feature names to ensure that the order matches the one in the list of indices of the features in
        # the case base class
        for key in self.relevant_features:
            self.relevant_features[key] = sorted(self.relevant_features[key])

        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.bools = data['bools']

        def flatten(l):
            return [item for sublist in l for item in sublist]

        self.all_features_configured = sorted(list(set(flatten(features_all_cases.values()))))

    # return the error case description for the passed label
    def get_error_description(self, error_label: str):
        return self.error_descriptions[error_label]


# import the timestamps of each dataset
def import_timestamps(self):
    datasets = []
    number_to_array = {}

    with open('../configuration/cases.csv', 'r') as file:
        for line in file.readlines():
            parts = line.split(',')
            parts = [part.strip(' ') for part in parts]
            dataset, case, start, end = parts

            timestamp = (gen_timestamp(case, start, end))

            if dataset in number_to_array.keys():
                number_to_array.get(dataset).append(timestamp)
            else:
                ds = [timestamp]
                number_to_array[dataset] = ds

    for key in number_to_array.keys():
        datasets.append(number_to_array.get(key))

    self.cases_datasets = datasets


def gen_timestamp(label: str, start: str, end: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time
