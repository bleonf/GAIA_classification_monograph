# GAIA_classification_monograph
See: raw

Python codes used for the monograph
Estadisticos.py: Python file with features based on numpy

DATA_CREATOR.py: Creates csv file with all features in Estadisticos.py. Must be in same folder as Estadisticos.py
  input : number of stars per class that you want to be processed
          Paths to folders which contain .dat files for each star class
  output: Correlation matrix for all features
          Saves .csv file with all chosen stars, loses star ID except for GAIA stars

Training_BestParams.py: Python file that uses GRIDSEARCH to find best parameters for RandomForest, SVM, DecsionTree and Kneighbours classifiers
  input: Parameters to be tested as lists
         paths to csv file created by DATA_CREATOR.py
  output: Best parameters among the ones stated

Train_final.py: Importances, score, recall, precision and accuracy generator for all classifiers
  input: Which list of features to use - edit list
         paths to csv file created by DATA_CREATOR.py
  output: Prints scores for all classifiers and importances to terminal
  
TRAIN_TEST_GAIA: Trains on LMC stars and shows resulting classification of GSEP stars
  input: paths to csv files generated with DATA_CREATOR.py
         Number of stars by star tyoe to be used, must be equal or less than number of stars
         Test size number between 0 and 1
         Custom list of features to use
  terminal input: a,b,c,d to chose classifier
         Number of top features to be used (list is organized by importance in Train_final.py). 0 applies custom features
         Generates confusion matrix on test data as figure 
            Save figure? Y->saves figure N->does not save figure
            Show figure? Y->shows figure N->dos not show figure
  output:Prints GSEP classification by star type.

Besura: Other files as histogram creators, specific trainings, histograms, deleted codes
