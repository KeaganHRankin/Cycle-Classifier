"""
This python file contains all of the functions and libraries required for the model to run.
Files are imported into the _model.ipynb file for ease of use. Keep this file organized
and comment every new function or class that is put in it.
"""

### THIRD PARTY LIBRARIES
# Import 3rd party libraries:
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

# SCIKIT Learn IMPORTS:
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from sklearn.utils.fixes import loguniform
from scipy.stats import uniform


### OUR FUNCTIONS

# Data Cleaning Functions


# Feature Engineering Functions
def ysign_droprows(data, keep):
    """
    Function drops rows that are not passed in 'keep' list.
    """
    
    #Drop
    data_drop = data[keep]
    
    return data_drop


def ysign_scale(data, scale_feats):
    """
    Scales the passed columns of the ysign data.
    """
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(data[scale_feats])
    
    scaled = data[scale_feats].copy()
    
    # Convert numeric features to standard units
    scaled = scaler.transform(scaled)
    
    data[scale_feats] = scaled
    
    return data

# Feature Selection Functions
def feature_selector(model, splits, X, y, i):
    """ 
    Selects features for the given model based on Kfold validation.
    inputs are splits, model + xy, and i-1# of best features.
    """
    
    # Initial setup
    f1s = []
    five_fold = StratifiedKFold(n_splits=splits)  #use stratified kfold due to calss imbalance.
    #could also use normal k-fold and remove y from five_fold.split()
    #five_fold = KFold(n_splits=splits)
    
    for train_index, val_index in five_fold.split(X, y):
        
        # Select i+1 best features
        select = SelectKBest(chi2, k=i).fit(X.iloc[train_index, :],
                                            y.iloc[train_index])
        
        # reduced feature set
        X_new = X.loc[:,select.get_support()]
        print('Selected features: {}'.format(X_new.columns.tolist()))
        
        # Fit model
        model.fit(X_new.iloc[train_index, :],
                  y.iloc[train_index])
        
        # Show prelim F1 score
        f1s.append(f1_score(y.iloc[val_index],
                            model.predict(X_new.iloc[val_index, :])
                           )
                  )
        
    print('Mean fold F1 Score: {}'.format(np.mean(f1s)))

# Hyper Parameter Tuning Functions


# Evaluation Functions














