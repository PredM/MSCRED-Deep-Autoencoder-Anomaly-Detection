import json

import pandas as pd
import os


class Configuration:

    def __init__(self, dataset_to_import=0):

        self.load_config_json('../configuration/config.json')

        ###
        # name of the directories and files
        ###

        self.filename_matrix_data = "matrix_win_"

        self.step_size_reconstruction = 1

        ###
        # NN
        ###

        # attribute for signature matrix generation

        # maximum step in ConvLSTM
        self.step_max = 5
        # gap time between each segment in time steps
        # self.gap_time = 125 #10#
        self.gap_time = 125
        # window size / length of each segment
        # self.win_size = [125, 250, 375, 500, 625, 750, 875, 1000]  # [10, 30, 60]#
        # self.win_size = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000] #[10, 30, 60]#
        # self.win_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,140, 150, 175, 200, 225, 250,
        #                 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        self.win_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150, 200, 250]

        # NN parameter
        self.num_datastreams = 61
        self.batch_size = 128
        self.dim_of_dataset = 18  # 1: 8 2: 17 3:18
        self.epochs = 1000
        self.learning_rate = 0.001
        self.filter_dimension_encoder = [32, 64, 128, 256]  # [16, 32, 64, 128] #[32, 64, 128, 256] #[64, 128, 256, 512]
        self.memory_size = 100
        # self.stride_encoder = [1, 2]#[1, 2, 2, 2]
        # self.filter_size_encoder = [[3, 3], [3, 3]]#[[3, 3], [3, 3], [2, 2], [2, 2]]
        # self.dimension_lstm = [128, 256]# self.filter_dimension_encoder
        # Not used, because equal to encoder: self.filter_dimension_decoder = [256, 128, 64, 32]#[256, 128, 64, 32, 3] #[128, 64, 32, 3]
        # self.stride_decoder = [1, 2, 1]
        # self.filter_size_decoder = [[3, 3], [3, 3], [1,1]]#[[2, 2], [2, 2], [3, 3], [3, 3], [3, 3]]
        # self.filter = False

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

