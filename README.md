# MSCRED Anomaly Detection
The following section gives an overview of the packages, directories and included Python scripts in this repository. 

## Code

| Python script | Purpose |
| ---      		|  ------  |
| anomaly_detection/TrainAndTest.py   | Used to start a training and to evaluate a model from MscredModel.py. Opportunity for setting training and testing relevant configurations (in addition to Configurations.py)   |
| anomaly_detection/MscredModel.py   | Contains the Keras/TF models for anomaly detection   |
| data/GenerateNoFailureRuns.py      		| Transforms previously extracted 4 seconds windows of time series back into their original sequence (named run or trajectory) and removes overlapping sections. For generating the training data, time series windows labeled not as "no_failures" are removed.  Requires as Input: train_window_times.npy, train_labels.npy, train_features.npy. Generates Output: NoFailure_Train_runs.npz  |
| data/GenerateTestRuns.py     		|  Similar to GenerateNoFailureRuns.py for the test data set. Considers also time series windows / examples that are not labeled as "no_failure"  |
| data/MatrixGenerator2_TrainDataWOFailures.py     		|  Generates the correlation matrices (referred to as signature matrices within the MSCRED-AD framework) and the settings such as step_size, win_size(s) and output folder can be defined via Configuration.py. Input: previously generated NoFailure_Train_runs.npz (by GenerateNoFailureRuns.py). Ouput: signature matrices|
| data/MatrixGenerator2_TrainDataWFailures.py     		| Similar to MatrixGenerator2_TrainDataWOFailures.py but with the difference that only training examples labelled as failures are processed. |
| data/MatrixGenerator2_TestData.py     		|  Similar to MatrixGenerator2_TrainDataWOFailures.py but with the difference that only test examples are processed |
| data/Matrix_to_NN-Input_train_woFailure.py | Transforms all the signature matrices in one file so that this need not to be done during training and provides the probability for mini batch. Input: previously generated signature matrices (by MatrixGenerator2_TrainDataWOFailures.py). Output: generates training_data_set_3_trainWoFailure.npy  |
| data/Matrix_to_NN-Input_train_woFailure.py |  Similar to MatrixGenerator2_TrainDataWOFailures.py but with the difference that only training examples labelled as failures are processed  Output: training_data_set_3_trainWFailure.npy with the according labels training_data_set_3_failure_labels.npy    |
| data/Matrix_to_NN-Input_test.py | Similar to MatrixGenerator2_TrainDataWOFailures.py but with the difference that only test examples are processed  Output: test_data_set_3.npy with the according labels test_data_set_3_failure_labels.npy |
| data/Attribute_Correlation_Relevance_Matrix_v0.csv | Matrix with manually defined relevance for each correlation between data streams |
| configurations/Configuration.py | Provides mainly configuration settings for matrix signature generation as well as some model specific as well as training specific parameters |

## Data Set
### Data Set Generation
- The data sets are based on: https://seafile.rlp.net/d/69434c0df2c3493aac7f/.
- This data set can be transformed to signature matrices as follows:
    1. Transform them back in to the original sequence order by using: <i>data/GenerateNoFailureRuns.py </i>
    2. Generate the signature matrices by using: <i>data/MatrixGenerator2_TrainDataWOFailures.py </i>	
    3. Transform them into examples consisting of the step_size and gap_size by using: <i>data/Matrix_to_NN-Input_train_woFailure.py </i>

### Data Sets (with different Signature Matrix Configurations)
From preprocessed time series data (4ms sampling rate, 250 entries per second), the following signature matrix data sets are generated:

| No | Folder | Steps per Example | Gap Time in ms | Correlation Window Sizes in ms|
| ------ | ------ | ------ | ------ | ------|
| 1 | / data / pklein / MSCRED_Input_Data | 5 | 125 | [125, 250, 375, 500, 625, 750, 875, 1000]|
| 2 | / data / pklein / MSCRED_Input_Data | 5 | 4000 ? | [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000]|
| 3 | / data / pklein / MSCRED_Input_Data | 5 | 125 | [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150, 200, 250] |

The data can be found at the GPU server: <i>data/pklein/MSCRED_Input_Data/</i> 