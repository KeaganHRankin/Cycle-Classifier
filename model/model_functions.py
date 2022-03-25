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
from shapely import wkt

# SCIKIT Learn IMPORTS:
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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
def droprows(data, keep):
    """
    Function drops rows that are not passed in 'keep' list.
    """
    
    #Drop
    data_drop = data[keep]
    
    return data_drop


def scale(data, scale_feats):
    """ 
    Just scales feats
    """
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(data[scale_feats])
    
    scaled = data[scale_feats].copy()
    
    # Convert numeric features to standard units
    scaled = scaler.transform(scaled)
    data[scale_feats] = scaled
    
    return data


def dummy(data, dummy_feats):
    """
    Just dummies features
    """
    for s in dummy_feats:
        categorical = pd.get_dummies(data[s], prefix=s, drop_first=True)
        data = pd.concat((data, categorical), axis=1)
        
    data = data.drop(dummy_feats, axis=1)
    
    return data    
    
def scale_and_dummy(data, scale_feats, dummy_feats):
    """
    Scales the passed columns of data.
    Dummy encode categorical features passed.
    """
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(data[scale_feats])
    
    scaled = data[scale_feats].copy()
    
    # Convert numeric features to standard units
    scaled = scaler.transform(scaled)
    data[scale_feats] = scaled
    
    # Convert categorical features using dummy encoding. 
    #Drop the encoded features from the frame.
    for s in dummy_feats:
        categorical = pd.get_dummies(data[s], prefix=s, drop_first=True)
        data = pd.concat((data, categorical), axis=1)
        
    data = data.drop(dummy_feats, axis=1)
    
    return data


def add_regions(data, components, bins):
    """
    function extract PC of Toronto's geometry and bins it into a # of regions.
    
    takes: in data, # of components, and # of bins.
    returns: the training data with x_regions and y_regions labels for dummy encoding.
    string
    """
    # Extract the centroids of the LINESTRING geometry.
    # Also return the x y components of these centroids
    data_g = data.copy()
    data_g['geometry'] = data_g['geometry'].apply(wkt.loads)
        
    train_data_gpd = gpd.GeoDataFrame(data_g, crs="EPSG:26917")
    train_data_gpd.head()
    
    centroids = train_data_gpd['geometry'].centroid
    centroids_df = pd.DataFrame(data={'x':centroids.x, 'y':centroids.y})
    
    # Scale features.
    scaler = StandardScaler().fit(centroids_df)
    X_scaled = scaler.transform(centroids_df)
    
    # Create/fit PCA function.
    pca = PCA(n_components=components)
    X_transformed = pca.fit_transform(X_scaled)
    X_transformed_df = pd.DataFrame(X_transformed)
    
    # Print results.
    for i in range(components):
        print('Principal component', i)
        print('explains', (pca.explained_variance_ratio_[i] * 100), '% of the variance in "lon" and "lat".')
    
    # Return inversed values
    X_new = pca.inverse_transform(X_transformed)
    X_new = scaler.inverse_transform(X_new)
    X_new_df = pd.DataFrame(X_new)
    
    # Bin geographic regions based on components
    out = pd.qcut(X_transformed_df[0], bins, labels=np.arange(1,bins+1,1))
    out2 = pd.qcut(X_transformed_df[1], bins, labels=np.arange(1,bins+1,1))   
    
    #Append the bin labels to the geo data.
    data_g['x_region'] = out
    data_g['y_region'] = out2
    
    #return X_transformed_df, centroids_df
    return data_g

# Feature Selection Functions
def feature_selector(model, splits, X, y, i):
    """ 
    Selects features for the given model based on Kfold validation.
    inputs are splits, model + xy, and i# of best features.
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
                            model.predict(X_new.iloc[val_index, :]),
                            average='weighted',
                           )
                  )
        
    print('Mean fold F1 Score: {}'.format(np.mean(f1s)))

# Hyper Parameter Tuning Functions


# Evaluation Functions
def f1_threshold(y_true, y_pred_proba, threshold):
    
    """
    Returns the F-beta score.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels 
                                 np.array([0, 1, 0, 0, ..]).
        y_pred_proba (1D numpy array): 1D array of prediction probabilities 
                                       for the positive class
                                       (model.predict_proba(X)[:, 1])
                                       np.array([0.12, 0.56, 0.23, 0.89, ..]).
        threshold (float): The probability threshold, which is a number 
                           between 0 and 1.

    Returns:
        (float): The F1 score given a threshold.
    """
    
    # Calculate the binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return f1_score(y_true, y_pred, average='weighted')


def plot_f1_threshold(X, y, model):
    
    """
    Returns the F-beta score.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels 
                                 np.array([0, 1, 0, 0, ..]).
        y_pred_proba (1D numpy array): 1D array of prediction probabilities 
                                       for the positive class
                                       (model.predict_proba(X)[:, 1])
                                       np.array([0.12, 0.56, 0.23, 0.89, ..]).
        threshold (float): The probability threshold, which is a number 
                           between 0 and 1.
    """

    # Create threshold array
    thresholds = np.arange(0, 1, 0.01)
    
    # Compute F1 scores for each threshold
    f1_scores = np.array([f1_threshold(y.values.flatten(), model.predict_proba(X)[:, 1], threshold) 
                          for threshold in np.arange(0, 1, 0.01)])
    
    # Get finite values
    thresholds = thresholds[np.isfinite(f1_scores)]
    f1_scores = f1_scores[np.isfinite(f1_scores)]
    
    # Optimal values
    idx = np.argmax(f1_scores)
    f1_score = f1_scores[idx]
    threshold = thresholds[idx]

    # Plot
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(wspace=0.2, hspace=0)
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax1.set_title('Optimal F1 score is {:.2f} for a threshold of {:.2f}.'.format(f1_score, threshold), fontsize=12)
    ax1.plot(thresholds, f1_scores, '.-')
    ax1.plot([threshold, threshold], [0, 1], '-', label='Optimal F1')
    ax1.legend()
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Get PR curve
    prc_precisions, prc_recalls, prc_thresholds = precision_recall_curve(y.values.flatten(), model.predict_proba(X)[:, 1])
    precision_opt = precision_score(y.values.flatten(), (model.predict_proba(X)[:, 1] >= threshold).astype(int))
    recall_opt = recall_score(y.values.flatten(), (model.predict_proba(X)[:, 1] >= threshold).astype(int))
    
    # Plot
    ax2.set_title('Optimal Precision is {:.2f} and recall is {:.2f}.'.format(precision_opt, recall_opt), fontsize=12)
    ax2.plot(prc_recalls, prc_precisions)
    ax2.plot(recall_opt, precision_opt, 'o', label='Optimal PR')
    ax2.legend()
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    

def plot_roc(y_true, y_prob):
    """
    Plots the roc curve for given
    y_true {0,1}, y_predicted probabilities, both 1d array like.
    y_prob should = model.predict_proba(X_train)
    """
    # Get the false pos rate, true pos rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1], pos_label=1)
    
    # plot false positive rate vs true positive rate and compare to the 1:1 line (which is random guess)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(fpr, tpr, label='model')
    fig.suptitle('ROC Curve for Model')
    plt.xlabel('False Positivity Rate (1- Specificity)')
    plt.ylabel('Ture Positivity Rate (1- Specificity)') 
    plt.plot([0,1],[0,1], transform=ax.transAxes, label='random guess')
    plt.legend()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    plt.show()














