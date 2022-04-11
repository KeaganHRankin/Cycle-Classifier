"""
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
    model_name = 'forest' or 'log' to train the model on either of these types of models.
    save_model = True/False whether to pickle the model.
    metrics = True/False whether to return training metrics.
    """
    
    print("[INFO] starting training.")
    
    # Perform feature engineering
    # Map the bikelanes in the correct way
    train_data = map_centreline_features(train_data)
    
    # Split features/target
    y_train_lts = train_data['LTS']
    y_train_access = train_data['high access']
    X_train = train_data.drop(['LTS','high access'], axis=1)
    
    # Engineer features usign model functions.
    keep_rows = ['FEATURE_CODE_DESC','geometry', 'AREA_ID', 'bikelane']
    X_train = droprows(X_train, keep_rows)
    X_train = add_regions(X_train, 2, 3)
    X_train = dummy(X_train, dummy_feats=['FEATURE_CODE_DESC','x_region','y_region'])
    
    print("[INFO] finished feature engineering.")
    
    # Define Features
    features = ['FEATURE_CODE_DESC_Arterial', 'FEATURE_CODE_DESC_Collector', 'FEATURE_CODE_DESC_Local', 'FEATURE_CODE_DESC_Trail',
                'x_region_2', 'x_region_3', 'y_region_2', 'y_region_3', 
                'bikelane',]
    
    
    # train based on the given model
    if model_name == 'log':
        model_log = LogisticRegression(C=0.9056753857212789, class_weight='balanced')
        model = model_log.fit(X_train[features], y_train_access)
        
        print("[INFO] finished training logreg model.")

    elif model_name == 'forest':
        model_forest = RandomForestClassifier(criterion='entropy', max_features=0.5384874461913766,
                                              max_samples=0.5862816454781242, class_weight='balanced')
        model = model_forest.fit(X_train[features], y_train_access) 
        
        print("[INFO] finished training random forest model.")
        
    else:
        print('[ERROR] not a valid model name: input "log" or "forest".')
    
    
    # Save the model if this option was specified
    if save_model:
        file_name = 'model_1.pkl'
        
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
        
        print(f'[INFO] saved model as {file_name}.')
        
        
    # Return training cross validation metrics
    if metrics:
        print('[INFO] returning training metrics using spatial cv...')

        # spatial_cv
        spatial_cv(model, grouper=X_train['AREA_ID'], splits=141, X=X_train[features], y=y_train_access)

        # Confusion matrix
        f, ax = plt.subplots(figsize=(10, 10))
        plot_confusion_matrix(model, X_train[features], y_train_access, ax=ax)
        ax.grid(False)

        #Weighted F1 score with optimal threshold if relevant
        plot_f1_threshold(X_train[features], y_train_access, model)

        #roc curve
        plot_roc(y_train_access, model.predict_proba(X_train[features]))
        
    else:
        print('[INFO] skipping metrics.')
    
    print('[INFO] complete.')

    return model



# Predicting results using the model.
def predict(data, model='', model_fname='model_1.pkl'):
    """
    Takes in a trained model (either just trained or from saved) and returns predictions + probabilities.
    data = test data in the format of centrelinebike_train_spatial.csv.
    model = trained model to use. Can be left empty.
    model_fname = model used if model name is not defined.
    """
    
    # Perform feature engineering to get in same format as trained data.
    # Map the bikelanes in the correct way
    data = map_centreline_features(data)
    
    # Split features/target
    #y_test_lts = data['LTS']
    y_test_access = data['high access']
    X_test = data.drop(['LTS','high access'], axis=1)
    
    # Engineer features usign model functions.
    keep_rows = ['FEATURE_CODE_DESC','geometry', 'AREA_ID', 'bikelane']
    X_test = droprows(X_test, keep_rows)
    X_test = add_regions(X_test, 2, 3)
    X_test = dummy(X_test, dummy_feats=['FEATURE_CODE_DESC','x_region','y_region'])
    
    print("[INFO] finished feature engineering")
    
    # Define Features
    features = ['FEATURE_CODE_DESC_Arterial', 'FEATURE_CODE_DESC_Collector', 'FEATURE_CODE_DESC_Local', 'FEATURE_CODE_DESC_Trail',
                'x_region_2', 'x_region_3', 'y_region_2', 'y_region_3', 
                'bikelane',]
    
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