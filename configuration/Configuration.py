import json

import pandas as pd
import os


class Configuration:

    def __init__(self, dataset_to_import=0):

        self.load_config_json('../configuration/config.json')

        ###
        # General Configuration
        ###
        self.curr_run_identifier = "model_23"
        self.use_data_set_version = 3
        self.train_model = False
        self.test_model = True

        ###
        # Autoencoder Configuration
        ###
        # Variants of the MSCRED
        self.guassian_noise_stddev = None       # MSCRED default: None, None für nichts oder Wert: 0.1 / denoising autoencoder
        self.use_attention = True               # MSCRED default: True, Deaktivierung der Attention, erfordert ConvLSTM
        self.keras_attention_layer_instead_of_own_impl = True  # MSCRED default: False, ansonsten andere Attention als im peper beschrieben
        self.use_convLSTM = True                # MSCRED default: True, Deaktivierung des ConvLSTM und Attention mittels False
        self.use_memory_restriction = False     # MSCRED default: False, Restricts the output only on previously seen examples

        self.use_loss_corr_rel_matrix = True   # MSCRED default: False, Reconstruction error is only based on correlations that are manually defined as relevant
        self.loss_use_batch_sim_siam = False    # MSCRED default: False,
        self.use_corr_rel_matrix_for_input = True  # MSCRED default: False,  input contains only relevant correlations, others set to zero
        self.use_corr_rel_matrix_for_input_replace_by_epsilon = True  # MSCRED default: False,  meaningful correlation that would be zero, are now near zero

        # NN parameter
        self.num_datastreams = 61
        self.batch_size = 128
        if self.use_data_set_version == 1:
            self.dim_of_dataset = 8
        elif self.use_data_set_version == 2:
            self.dim_of_dataset = 17
        elif self.use_data_set_version == 3:
            self.dim_of_dataset = 18
        #self.dim_of_dataset = 18  # 1: 8 2: 17 3:18
        self.epochs = 1
        self.learning_rate = 0.001
        self.early_stopping_patience = 3
        self.split_train_test_ratio = 0.1
        self.filter_dimension_encoder = [32, 64, 128, 256] # [16, 8, 4, 1] [64, 128, 256, 512]  #  # [16, 32, 64, 128] #[32, 64, 128, 256] #[64, 128, 256, 512]
        self.strides_encoder = [1, 2, 2, 2]
        self.kernel_size_encoder = [3, 3, 2, 2]
        self.memory_size = 100

        ###
        # Test Evaluation
        ###
        self.use_corr_rel_matrix_in_eval = self.use_loss_corr_rel_matrix
        self.threshold_selection_criterium = '90%'  # 'max', '99%' #[.25, .5, .75, 0.9, 0.95, 0.97, 0.99]
        self.num_of_dim_over_threshold = 0  # normal: 0
        self.num_of_dim_under_threshold = 20  # normal: 20 (higher as max. dim value) # 3: 20
        self.num_of_dim_considered = self.dim_of_dataset # MSCRED: 1, only first dimension is used for anomaly detection
        self.print_att_dim_statistics = False
        self.generate_deep_encodings = False
        self.plot_heatmap_of_rec_error = False
        self.remove_hard_to_detect_stuff = ['low_wear']                         # not implemented
        self.use_attribute_anomaly_as_condition = False                          # MSCRED default: True
        self.use_dim_for_anomaly_detection = range(14,self.dim_of_dataset)         # MSCRED default: range(1) // only first dimension 0, range(1,2): dim 1
        self.print_all_examples = True

        self.use_mass_evaulation = False
        # Mass evaluation parameters
        self.threshold_selection_criterium_list = ['99%', '99%', '99%', '99%', '97%', '97%', 'max', 'max', 'max', '90%', '97%', '97%']
        self.num_of_dim_over_threshold_list = [1, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1]
        self.num_of_dim_under_threshold_list = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        self.use_corr_rel_matrix_in_eval_list = [False, True, False, True, False, True, False, True, True, True, False, True]
        self.use_attribute_anomaly_as_condition_list = [True, True, True, True, True, True, True, True, True, False, False, False]
        self.print_pandas_statistics_for_validation = False #self.use_mass_evaulation

        ###
        # Data Generation Configuration
        ###
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

        self.filename_matrix_data = "matrix_win_"
        self.step_size_reconstruction = 1

        ###
        # Data Set Configuration
        ###
        path = "../../../../data/pklein/MSCRED_Input_Data/"
        if self.use_data_set_version == 1:
            self.training_data_set_path = path + "training_data_set.npy"
            self.valid_split_save_path = path + "training_data_set_test_split.npy"
            self.test_matrix_path = path + "training_data_set_failure.npy"
            self.test_labels_y_path = path + "training_data_set_failure_labels_test.npy"
            self.test_matrix_path = path + "test_data_set.npy"
            self.test_labels_y_path = path + "test_data_set_failure_labels.npy"
        elif self.use_data_set_version == 2:
            self.training_data_set_path = path + "training_data_set_2_trainWoFailure.npy"
            self.valid_split_save_path = path + "training_data_set_2_test_split.npy"
            self.test_matrix_path = path + "training_data_set_2_trainWFailure.npy"
            self. test_labels_y_path = path + "training_data_set_2_failure_labels.npy"
            self.test_matrix_path = path + "test_data_set_2.npy"
            self.test_labels_y_path = path + "test_data_set_2_failure_labels.npy"
        elif self.use_data_set_version == 3:
            self.training_data_set_path = path + "training_data_set_3_trainWoFailure.npy"
            self.valid_split_save_path = path + "training_data_set_3_test_split.npy"
            self.test_matrix_path = path + "training_data_set_3_trainWFailure.npy"
            self.test_labels_y_path = path + "training_data_set_3_failure_labels.npy"
            self.test_matrix_path = path + "test_data_set_3.npy"
            self.test_labels_y_path = path + "test_data_set_3_failure_labels.npy"

        self.feature_names_path = "../data/feature_names.npy"

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

