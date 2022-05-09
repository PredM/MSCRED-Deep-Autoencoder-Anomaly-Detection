import json

import pandas as pd
import os


class Configuration:

    def __init__(self, dataset_to_import=0):

        self.load_config_json('../configuration/config.json')

        ###
        # General Configuration
        ###
        self.curr_run_identifier = "Fin_DS_MANW_2022_wAdjMat_" #Fin_Standard_wAdjMat_newAdj_2 #"Fin_Standard_wAdjMat_newAdj_2_wAdjMatwFM_"#"Fin_Standard_wAdjMat_newAdj_2_wAdjMatShuffled"
        self.use_data_set_version = 2022_2
        self.train_model = True
        self.test_model = True
        self.save_results_as_file = True
        self.use_train_FaF_in_eval = False

        ###
        # Autoencoder Configuration
        ###
        # Variants of the MSCRED
        self.guassian_noise_stddev = None       # MSCRED default: None, None für nichts oder Wert: 0.1 / denoising autoencoder (Dropout zusätzlich ergänzt)
        self.l1Reg = 0.0                       # MSCRED default: 0.0 , not recommended: 0.001 (Leads to high loss for reconstruction)
        self.use_attention = True               # MSCRED default: True, Deaktivierung der Attention, erfordert ConvLSTM
        self.keras_attention_layer_instead_of_own_impl = False  # MSCRED default: False, ansonsten andere Attention als im peper beschrieben
        self.use_convLSTM = True                # MSCRED default: True, Deaktivierung des ConvLSTM und Attention mittels False
        self.use_memory_restriction = False     # MSCRED default: False, Restricts the output only on previously seen examples
        self.use_MemEntropyLoss = False
        self.use_filter_for_memory = False      # Convolutional filters instead of feature matrices
        self.use_graph_conv = False
        self.normalize_residual_matrices = False     # Normalization of reconstruction error

        self.use_loss_corr_rel_matrix = True   # MSCRED default: False, Reconstruction error loss is only based on correlations that are manually defined as relevant
        self.loss_use_batch_sim_siam = False    # MSCRED default: False,
        self.use_corr_rel_matrix_for_input = True  # MSCRED default: False,  input contains only relevant correlations, others set to zero
        self.use_corr_rel_matrix_for_input_replace_by_epsilon = False  # MSCRED default: False,  meaningful correlation that would be zero, are now near zero
        self.use_corr_rel_matrix_on_masking_residual_matrices = False    # NOT USED; Masking out irrelevant correlations during the evaluation phase is already done with the activation of: use_loss_corr_rel_matrix

        # NN parameter
        self.num_datastreams = 78 if self.use_data_set_version == 2022_2 else 61
        self.batch_size = 128
        if self.use_data_set_version == 1:
            self.dim_of_dataset = 8
        elif self.use_data_set_version == 2:
            self.dim_of_dataset = 17
        elif self.use_data_set_version == 3:
            self.dim_of_dataset = 18
        elif self.use_data_set_version == 4:
            self.dim_of_dataset = 4
        elif self.use_data_set_version == 5:
            self.dim_of_dataset = 4
        elif self.use_data_set_version == 2022:
            self.dim_of_dataset = 4
        elif self.use_data_set_version == 2022_2:
            self.dim_of_dataset = 3
        #self.dim_of_dataset = 18  # 1: 8 2: 17 3:18
        self.epochs = 100                           # used in eval: 100
        self.learning_rate = 0.001                  # used in eval: 0.001
        self.early_stopping_patience = 3            # used in eval: 3
        self.split_train_test_ratio = 0.1
        self.use_strides = True
        self.threshold_step = 100                  # used in eval: 100, faster: 1000 or 10.000
        self.used_valid_split = [0.0] # [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97] # [0.0] #[0.0,0.25,0.50,0.75,0.90]

        if self.use_strides == True:
            if self.use_graph_conv == True:
                self.strides_encoder = [1, 1, 1, 1]  # [1, 2, 2, 2]
                self.dilation_encoder = [1, 1, 1, 1]
                self.output_dim = [61, 61, 61, 61]  # dependent on strides
                self.filter_dimension_encoder = [16, 8, 4, 1]  # [32, 64, 128, 256] #[32, 16, 8, 4] # [64, 32, 16, 8] # [32, 64, 128, 256]#[32, 64, 128, 256]  # [16, 8, 4, 1] [64, 128, 256, 512]  #  # [16, 32, 64, 128] #[32, 64, 128, 256] #[64, 128, 256, 512]
            else:
                self.strides_encoder = [1, 2, 2, 2] #[1, 2, 2, 2]
                self.dilation_encoder = [1, 1, 1, 1]
                if self.use_data_set_version == 2022_2:
                    self.output_dim = [8, 16, 31, 78]  #[8, 16, 31, 61]    #dependent on strides
                else:
                    self.output_dim = [8, 16, 31, 61]
                self.filter_dimension_encoder = [32, 64, 128, 256]# [32, 64, 128, 256] #[8, 16, 32, 64]  # [8, 16, 32, 64]#[32, 64, 128, 256] #[32, 16, 8, 4] # [64, 32, 16, 8] # [32, 64, 128, 256]#[32, 64, 128, 256]  # [16, 8, 4, 1] [64, 128, 256, 512]  #  # [16, 32, 64, 128] #[32, 64, 128, 256] #[64, 128, 256, 512]
        else:
            self.dilation_encoder = [1, 1, 1, 1]
            self.strides_encoder = [1, 2, 2, 2]
            self.output_dim = [8, 16, 31, 61]
            self.filter_dimension_encoder = [32, 64, 128, 256]
            '''
            self.dilation_encoder = [1, 2, 2, 2]
            self.strides_encoder = [1, 1, 1, 1]
            self.output_dim = [61, 61, 61, 61]
            self.filter_dimension_encoder = [16, 8, 4, 1]
            '''

        self.kernel_size_encoder = [3, 3, 2, 2] #[3, 3, 2, 2]
        self.memory_size = 100

        ###
        # Test Evaluation
        ###
        self.use_corr_rel_matrix_in_eval = self.use_loss_corr_rel_matrix
        self.threshold_selection_criterium = '99%'  # 'max', '99%' #[.25, .5, .75, 0.8, 0.9, 0.95, 0.97, 0.99]
        self.num_of_dim_over_threshold = 0  # normal: 0
        self.num_of_dim_under_threshold = 20  # normal: 20 (higher as max. dim value) # 3: 20
        self.use_dim_for_anomaly_detection = range(self.dim_of_dataset)         # MSCRED default: range(1) // only first dimension 0, range(1,2): dim 1
        self.num_of_dim_considered = self.dim_of_dataset # MSCRED: 1, only first dimension is used for anomaly detection
        self.print_att_dim_statistics = False
        self.generate_deep_encodings = False
        self.plot_heatmap_of_rec_error = False
        #self.remove_hard_to_detect_stuff = ['low_wear']                         # not implemented
        self.use_attribute_anomaly_as_condition = True                          # MSCRED default: True
        self.print_all_examples = False                                             # For inspection / debugging

        self.use_mass_evaulation = True
        # Mass evaluation parameters
        # SINCE NEW IMPLEMENTATION NOT MORE CONSIDERED
        self.threshold_selection_criterium_list = ['99%', '99%', '99%', '99%', '97%', '97%', 'max', 'max', 'max', '90%', '90%', '97%', '97%']
        self.num_of_dim_over_threshold_list = [1, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1]
        self.num_of_dim_under_threshold_list = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        self.use_corr_rel_matrix_in_eval_list =         [False, True, False, True, False, True, False, True, True, False, True, False, True]
        self.use_attribute_anomaly_as_condition_list =  [True, True, True, True, True, True, True, True, True, False, False, False, False]
        self.print_pandas_statistics_for_validation = False #self.use_mass_evaulation

        ###
        # Data Generation Configuration
        ###
        # maximum step in ConvLSTM
        if self.use_data_set_version == 2022_2:
            self.step_max = 4 #5
        else:
            self.step_max = 4  # 5
        # gap time between each segment in time steps
        # self.gap_time = 125 #10#
        if self.use_data_set_version == 2022_2:
            self.gap_time =125# 250 #250 #3/1: 125
        else:
            self.gap_time = 250
        # window size / length of each segment
        #1: self.win_size = [125, 250, 375, 500, 625, 750, 875, 1000]  # [10, 30, 60]#
        # self.win_size = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000] #[10, 30, 60]#
        # self.win_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,140, 150, 175, 200, 225, 250,
        #                 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        #self.win_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150, 200, 250]
        #self.win_size = [63, 125, 188, 250]
        if self.use_data_set_version == 2022_2:
            self.win_size = [40, 80, 125]
        else:
            self.win_size = [63, 125, 188, 250]

        self.filename_matrix_data = "matrix_win_"
        self.step_size_reconstruction = 1
        self.directoryname_matrix_data = "matrix_data_5"
        self.replace_zeros_with_epsilion_in_matrix_generation = False

        ###
        # Data Set Configuration
        ###
        if self.use_data_set_version == 2022:
            path = "../../../../data/pklein/MSCRED_Input_Data/"
        elif self.use_data_set_version == 2022_2:
            path = "../../../../data/pklein/MSCRED_Input_Data_2/"
        self.feature_names_path = path + "feature_names.npy"
        #path = '../data/'
        self.path = path
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
        elif self.use_data_set_version == 4:
            self.training_data_set_path = path + "training_data_set_4_trainWoFailure.npy"
            self.valid_split_save_path = path + "training_data_set_4_test_split.npy"
            self.test_matrix_path = path + "training_data_set_4_trainWFailure.npy"
            self.test_labels_y_path = path + "training_data_set_4_failure_labels.npy"
            self.test_matrix_path = path + "test_data_set_4.npy"
            self.test_labels_y_path = path + "test_data_set_4_failure_labels.npy"
        elif self.use_data_set_version == 5:
            self.training_data_set_path = path + "training_data_set_4_epsi_trainWoFailure.npy"
            self.valid_split_save_path = path + "training_data_set_4_epsi_test_split.npy"
            self.test_matrix_path = path + "training_data_set_4_epsi_trainWFailure.npy"
            self.test_labels_y_path = path + "training_data_set_4_failure_labels.npy"
            self.test_matrix_path = path + "test_data_set_4_epsi.npy"
            self.test_labels_y_path = path + "test_data_set_4_failure_labels.npy"
        elif self.use_data_set_version == 2022:
            self.training_data_set_path = path + "sig_mat_train_2.npy"
            self.valid_split_save_path = path + "training_data_set_test_split_2022.npy"
            self.valid_matrix_path_wF = path + "sig_mat_valid.npy"
            self.valid_labels_y_path_wF = path + "valid_labels.npy"
            self.test_matrix_path = path + "sig_mat_test.npy"
            self.test_labels_y_path = path + "test_labels.npy"
            self.train_faf_matrix_path = path + "sig_mat_train_failures4Test.npy"
            self.train_faf_labels_y_path = path + "sig_mat_train_failures4Test_labels.npy"
            self.graph_adjacency_matrix_attributes_file = path + "adjmat_new.csv"  # "adjmat_new_shuffled.csv" #"adjacency_matrix_v3_fullGraph_sparse.CSV"

        elif self.use_data_set_version == 2022_2:
            self.training_data_set_path = path + "sig_mat_train_2.npy"
            self.valid_split_save_path = path + "training_data_set_test_split_2022.npy"
            self.valid_matrix_path_wF = path + "sig_mat_valid.npy"
            self.valid_labels_y_path_wF = path + "valid_labels.npy"
            self.test_matrix_path = path + "sig_mat_test.npy"
            self.test_labels_y_path = path + "test_labels.npy"
            self.train_faf_matrix_path = path + "sig_mat_train_failures4Test.npy"
            self.train_faf_labels_y_path = path + "sig_mat_train_failures4Test_labels.npy"
            self.graph_adjacency_matrix_attributes_file = path + "adjmat_new.csv"  # "adjmat_new_shuffled.csv" #"adjacency_matrix_v3_fullGraph_sparse.CSV"



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

