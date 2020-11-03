import dataPreprocessing.preprocessing as prep
import dataPreprocessing.normalization as norm
import old_.splitDataSets as split
import old_.matrixGenerator as matrix
import old_.main as m
import old_.analyseNNresults.detectionAndDiagnosis as dd
#from configuration.Configuration import Configuration

if __name__ == '__main__':
    #dataPreprocessing.convertTXT()
    prep.main()
    norm.main()
    matrix.main()
    split.main()
    m.main()
    dd.main()