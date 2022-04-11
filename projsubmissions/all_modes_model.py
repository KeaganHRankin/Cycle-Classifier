"""
ALL MODES MODEL -> USES CENSUS DATA AND INTERSECTION VOLUME DATA TO PREDICT THE BIKE ACCESSIBILITY OF ROADS.

This .py file deploys the proof-of-concept model for classifying the bike accessibility of roads.
The model should be trained with appropriate data format (see example datasets in this repo).
It can then output the predicited stress of a road for any city.
1 = LOW STRESS (LTS 1 & 2)
0 = HIGH STRESS (LTS 3 & 4)
"""

# Import required functions. See model_functions for nuts & bolts of training and results.
from model_functions import *
import pickle



# Training the model
def train(train_data, model_name, save_model=False, metrics=False):
    """
    trains the model given data and what model to use.
    train_data = training data in the format of centrelinebike_train_spatial.csv.
    model_name = 'forest', 'log', 'svm', or 'boost' to train the model on either of these types of models.
    save_model = True/False whether to pickle the model.
    metrics = True/False whether to return training metrics.
    """
    
    print("[INFO] starting training.")

    # Perform feature engineering
    # fill NaN population densities with the median
    train_data['pop_density']=train_data['pop_density'].fillna(train_data['pop_density'].median())
    
    # Split features/target
    y_train_lts = train_data['LTS']
    y_train_access = train_data['high access']
    X_train = train_data.drop(['LTS','high access'], axis=1)
    
    # Engineer features using model functions.
    keep_rows = ['geometry', 'AREA_ID', 'bike lane', 'pop_density', 'car volume from', 'truck volume from',
             'ped volume from', 'car volume to', 'truck volume to', 'ped volume to']
    X_train = droprows(X_train, keep_rows)
    X_train = add_regions(X_train, 2, 3)
    X_train = dummy(X_train, dummy_feats=['x_region','y_region'])
    
    if model_name == 'svm':
        # Standard scaling if svm
        X_train = scale(X_train, scale_feats=['pop_density', 'car volume from','truck volume from','ped volume from',
                                                    'car volume to', 'truck volume to', 'ped volume to'])
    else: 
        # minmax scaling otherwise
        X_train = scale_minmax(X_train, scale_feats=['pop_density', 'car volume from','truck volume from','ped volume from',
                                                    'car volume to', 'truck volume to', 'ped volume to'])
    
    
    print("[INFO] finished feature engineering.")
     
    
    # train based on the given model
    if model_name == 'log':
        
        # define features
        features = ['bike lane', 'pop_density', 'car volume from', 'truck volume from', 
                    'ped volume from', 'car volume to', 'truck volume to', 'x_region_2', 
                    'x_region_3', 'y_region_3']
        
        model_features_log = LogisticRegression(C=16.71089673897469, class_weight='balanced')
        model = model_features_log.fit(X_train[features], y_train_access)

        print("[INFO] finished training logreg model.")

    elif model_name == 'svm':
        
        # define features
        features = ['bike lane', 
                    'pop_density', 'car volume from', 'truck volume from', 'car volume to', 'truck volume to', 'ped volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']
        
        model_features_svm = svm.SVC(C=16.19763, class_weight='balanced')
        model = model_features_svm.fit(X_train[features], y_train_access)
        
        print("[INFO] finished training support vector model.")
        
    elif model_name == 'forest':
        
        # define features
        features = ['bike lane', 'pop_density', 
                    'car volume from', 'truck volume from', 'ped volume from', 'car volume to', 'truck volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']
        
        model_features_rf = RandomForestClassifier(max_features=0.84174, max_samples=0.7380796)
        model = model_features_rf.fit(X_train[features], y_train_access)
        
        print("[INFO] finished training random forest model.")
        
    elif model_name == 'boost':
        
        from sklearn.ensemble import GradientBoostingClassifier
        # define features
        features = ['bike lane', 'pop_density', 
                    'car volume from', 'truck volume from', 'ped volume from', 'car volume to', 'truck volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']

        model_features_boost = GradientBoostingClassifier(learning_rate=0.36233394016439224, max_depth=10,
                           min_samples_leaf=2, min_samples_split=4)
        model = model_features_boost.fit(X_train[features], y_train_access)
        
        print("[INFO] finished training gradient boosting model.")
        
    else:
        print('[ERROR] not a valid model name: input "log" or "forest".')
    
    
    # Save the model if this option was specified
    if save_model:
        file_name = 'model_2.pkl'
        
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
        
        print(f'[INFO] saved model as {file_name}.')
        
        
    # Return training cross validation metrics
    if metrics:
        print('[INFO] returning training metrics using spatial cv...')

        # spatial_cv
        spatial_cv(model, grouper=X_train['AREA_ID'], splits=len(X_train['AREA_ID'].unique()), X=X_train[features], y=y_train_access)
        
    else:
        print('[INFO] skipping metrics.')
    
    print('[INFO] complete.')

    return model



# Before test predictions can be performed, the median from the training set must be extracted so it can be used when predicted
# this avoids data leakage
def extract_median_(train_data):
    """
    run this function before predicting to extract appropriate median for feature creation
    
    train_data = training data.
    """
    return train_data['pop_density'].median()


# Predicting results using the model.
def predict(data, train_median, model_type, model='', model_fname='model_2.pkl'):
    """
    Takes in a trained model (either just trained or from saved) and returns predictions + probabilities.
    
    data = test data in the format of centrelinebike_train_spatial.csv.
    model = trained model to use. Can be left empty.
    model_fname = model used if model name is not defined.
    model_type = type of model, 'log', 'svm', 'forest', 'boost'
    
    --RUN extract_median--
    train_median = median returned from extract_median function.
    """
    
    # Perform feature engineering, ensure medians come from training data.
    # fill NaN population densities with the median
    data['pop_density']=data['pop_density'].fillna(train_median)
    
    # Split features/target
    y_test_lts = data['LTS']
    y_test_access = data['high access']
    X_test = data.drop(['LTS','high access'], axis=1)
    
    # Engineer features using model functions.
    keep_rows = ['geometry', 'AREA_ID', 'bike lane', 'pop_density', 'car volume from', 'truck volume from',
                 'ped volume from', 'car volume to', 'truck volume to', 'ped volume to']
    X_test = droprows(X_test, keep_rows)
    X_test = add_regions(X_test, 2, 3)
    X_test = dummy(X_test, dummy_feats=['x_region','y_region'])
    
    if model_type == 'svm':
        # Standard scaling if svm
        X_test = scale(X_test, scale_feats=['pop_density', 'car volume from','truck volume from','ped volume from',
                                                    'car volume to', 'truck volume to', 'ped volume to'])
    else: 
        # minmax scaling otherwise
        X_test = scale_minmax(X_test, scale_feats=['pop_density', 'car volume from','truck volume from','ped volume from',
                                                    'car volume to', 'truck volume to', 'ped volume to'])
    
    print("[INFO] finished feature engineering")
    
    # Define features
    if model_type == 'log':

        # define features
        features = ['bike lane', 'pop_density', 'car volume from', 'truck volume from', 
                    'ped volume from', 'car volume to', 'truck volume to', 'x_region_2', 
                    'x_region_3', 'y_region_3']

    elif model_type == 'svm':
        
        # define features
        features = ['bike lane', 
                    'pop_density', 'car volume from', 'truck volume from', 'car volume to', 'truck volume to', 'ped volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']
        
    elif model_type == 'forest':
        
        # define features
        features = ['bike lane', 'pop_density', 
                    'car volume from', 'truck volume from', 'ped volume from', 'car volume to', 'truck volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']
        
    elif model_type == 'boost':
        
        # define features
        features = ['bike lane', 'pop_density', 
                    'car volume from', 'truck volume from', 'ped volume from', 'car volume to', 'truck volume to', 
                    'x_region_2', 'x_region_3', 'y_region_3']
    
    
    # Load Model
    if not model:
        
        print(f"[INFO] opening model from file {model_fname} ")
        
        with open(model_fname, 'rb') as f:
            model = pickle.load(f)
            
            
    # Predict
    y_pred = model.predict(X_test[features])
    
    try:
        y_prob = model.predict_proba_(X_test[features])
        
    except AttributeError:
        print('[ERROR] modeltype randomforest has no "predict_proba_". Not returning')
        y_prob = 0
    
    # Report Label
    print('[INFO] complete.')

    return y_pred, y_prob