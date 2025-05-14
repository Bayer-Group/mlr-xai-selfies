import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import KFold
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_features(data, CLOUMS):
    """
    Preprocesses the features according to theyr type.

    Keyword arguments:
    -- data: Table where the features are stored.
    -- CLOUMS: Colum where the feature is stored.

    Returns:
    -- X: Preprocessed feature.
    """
    features = []

    for i in CLOUMS:
        if type(data[i].values[0]) is np.ndarray:
            x = np.stack(data[i].values)
        else:
            x = data[i].values.reshape(-1, 1)
        features.append(x)

    X = np.concatenate(features, axis=1)
    return X

def get_morgan_fingerprint(smiles):
    """
    Calculates the Morgan Fingerprint (2028 bits, radius of 1) for the input smiles.

    Keyword arguments:
    -- smiles: Smiles for the which the fingerprint should be calculated.

    Returns:
    -- fingerprint: Morgan Fingerprint as array to the respective smiles.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol)#to keep the explicit hydrogens
    
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint = fingerprint.ToBitString()
        fingerprint = np.array(list(fingerprint))
        return fingerprint
    else:
        return None

def get_MACCS_fingerprint(smiles):
    """
    Calculates the MCCS Keys Fingerprint for the input smiles.

    Keyword arguments:
    -- smiles: Smiles for the which the fingerprint should be calculated.

    Returns:
    -- maccs_fp: MCCS Keys Fingerprint as array to the respective smiles.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol)#to keep the explicit hydrogens
    
    if mol is not None:
        maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
        maccs_fp = maccs_fp.ToBitString()
        maccs_fp = np.array(list(maccs_fp))
        return maccs_fp
    else:
        return None
        

def get_RDK_fingerprint(smiles):
    """
    Calculates the RDK Fingerprint (Maximal pathlenght of 7 and 2048 bits) for the input smiles.

    Keyword arguments:
    -- smiles: Smiles for the which the fingerprint should be calculated.

    Returns:
    -- rdkit_fp: RDK Fingerprint as array to the respective smiles.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol)#to keep the explicit hydrogens
    
    if mol is not None:
        rdkit_fp = AllChem.RDKFingerprint(mol, maxPath=7)
        rdkit_fp = rdkit_fp.ToBitString()
        rdkit_fp = np.array(list(rdkit_fp))
        return rdkit_fp
    else:
        return None
    
def hp_search_helper(model,df_train,target,feature):
    
    PARAM_GRID = {
    'SVC': {'model__C': [0.1, 1, 10, 100], 'model__kernel': ['rbf'], 'model__class_weight': ['balanced'], 'model__gamma': ['scale', 'auto', 1, 0.001, 0.01, 0.1]},
    'RandomForestClassifier': {'n_estimators': [400,700,1000], 'class_weight': ['balanced'], 'min_samples_leaf': [2,3]},

    'SVR': {'model__C': [0.1, 1, 10, 100], 'model__kernel': ['rbf'],  'model__gamma': ['scale', 'auto', 1, 0.001, 0.01, 0.1]},
    'RandomForestRegressor': {'model__n_estimators': [400,700,1000], 'model__max_depth':[30, 50], 'model__n_jobs':[32], 'model__random_state': [123]},
    
    'MLPRegressor': {'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],'model__activation': ['relu', 'tanh'],'model__solver': ['adam'],'model__alpha': [0.0001, 0.001, 0.01],'model__learning_rate': ['constant', 'adaptive'],'model__max_iter': [200, 500]},
    'BayesianRidge': {'model__alpha_1': [1e-7, 1e-6, 1e-5],'model__alpha_2': [1e-7, 1e-6, 1e-5],'model__lambda_1': [1e-7, 1e-6, 1e-5],'model__lambda_2': [1e-7, 1e-6, 1e-5]},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],'model__max_iter': [1000, 5000],'model__tol': [1e-4, 1e-3, 1e-2],'model__selection': ['cyclic', 'random']},
    'GradientBoostingRegressor': {'model__n_estimators': [100, 300, 500],'model__learning_rate': [0.01, 0.05, 0.1],'model__max_depth': [3, 5, 7],'model__subsample': [0.6, 0.8, 1.0],'model__min_samples_split': [2, 5, 10]},
    'LinearRegression': {},
    'GaussianProcessRegressor': {'model__alpha': [1e-10, 1e-5, 1e-2],'model__n_restarts_optimizer': [0, 2, 5, 10],'model__normalize_y': [True, False]}
    }

    SCORING = {
        'SVC': 'f1',
        'RandomForestClassifier': 'f1',
        'RandomForestRegressor': "r2",
        'SVR': "r2",
        'MLPRegressor': 'r2',
        'BayesianRidge': 'r2',
        'Lasso': 'r2',
        'GradientBoostingRegressor': 'r2',
        'LinearRegression': 'r2',
        'GaussianProcessRegressor': 'r2'
    }

    param_grid = PARAM_GRID[model.__class__.__name__]
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 
    pipe = HalvingRandomSearchCV(Pipeline([('scaler', StandardScaler()), ('model', model.__class__())]), param_distributions=param_grid, random_state=42, refit=True, cv=kf, scoring=SCORING[model.__class__.__name__])
    
    prep_train = get_features(df_train, feature)
    target_train = df_train[target].values

    pipe.fit(prep_train, target_train)

    splitted_data = kf.split(df_train)

    correlation_coff = []
    coef_of_determ = []
    mean_AE = []
    RMSE =[]

    for train_fold, test_fold in splitted_data:
        train_data = df_train.iloc[train_fold]
        test_data = df_train.iloc[test_fold]

        prep_train = get_features(train_data, feature)
        prep_test = get_features(test_data, feature)
        target_train = train_data[target].values
        target_test = test_data[target].values
            
        best_estimator = pipe.best_estimator_
        model = best_estimator.fit(prep_train, target_train)
            
        predictions = model.predict(prep_test)

        #stats
        r2 = np.corrcoef(target_test.flatten(), predictions.flatten())[0,1]**2
        correlation_coff.append(r2)
        
        RMSE_z = np.sqrt(np.mean((target_test.flatten() - predictions.flatten())**2))
        RMSE_n = np.sqrt(np.mean((target_test.flatten() - np.mean(target_test.flatten()))**2))
        R2 = 1 - RMSE_z**2/RMSE_n**2
        coef_of_determ.append(R2)
        
        MAE = mean_absolute_error(target_test.flatten(), predictions.flatten())
        mean_AE.append(MAE)
        
        mse = mean_squared_error(target_test.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        RMSE.append(rmse)

    mean_r2 = statistics.mean(correlation_coff)
    mean_R2 = statistics.mean(coef_of_determ)
    mean_MAE = statistics.mean(mean_AE)
    mean_RMSE = statistics.mean(RMSE)
    
    return pipe.best_estimator_, mean_r2, mean_R2, mean_MAE, mean_RMSE