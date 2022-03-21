"""
This python file contains all of the functions and libraries required for the model to run.
Files are imported into the _model.ipynb file for ease of use. Keep this file organized
and comment every new function or class that is put in it.
"""

### THIRD PARTY LIBRARIES
# Import 3rd party libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# SCIKIT Learn IMPORTS:
from sklearn.linear_model import LogisticRegression
from sklearn import svm

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
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from sklearn.utils.fixes import loguniform
from scipy.stats import uniform


### OUR FUNCTIONS

# Data Cleaning Functions


# Feature Engineering Functions


# Feature Selection Functions


# Hyper Parameter Tuning Functions


# Evaluation Functions














