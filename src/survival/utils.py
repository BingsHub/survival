import pandas as pd
from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle
import os
def show_all():
    return pd.option_context('display.max_rows', None, 'display.max_columns', None)

def lower_case(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.lower()
    return data
def geonames_cleaner(df, cols):
    for col in cols:
        df[col] = df[col].str.lower().str.replace('  ', ' ').str.replace('-', ' ').str.replace("'", '')\
            .str.replace('é', 'e').str.replace('ë', 'e').str.replace('ö', 'o').str.replace('ü', 'u')\
            .str.replace('ï', 'i').str.replace('î', 'i').str.replace('ç', 'c').str.replace('à', 'a')\
            .str.replace('â', 'a').str.replace('ê', 'e').str.replace('ô', 'o').str.replace('û', 'u')\
            .str.replace('è', 'e').str.replace('.', '').str.replace('(', '').str.replace(')', '')\
            .str.replace(',', '').str.replace('ú', 'u').str.strip()
    return df

def validate_imputation(df, features, target, missing_ratio=0.2, n_splits=5):
    complete_data = df[features + [target]].dropna()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(complete_data)):
        train_data = complete_data.iloc[train_idx]
        test_data = complete_data.iloc[test_idx]
        
        # Create missing values in test set
        mask = np.random.rand(len(test_data)) < missing_ratio
        data_with_holes = test_data.copy()
        true_values = data_with_holes.loc[mask, target].copy()
        data_with_holes.loc[mask, target] = np.nan
        
        # Fit imputer on train data
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=fold),
            random_state=fold
        )
        imputer.fit(train_data)
        
        # Impute test data
        imputed_data = imputer.transform(data_with_holes)
        imputed_values = imputed_data[mask, -1]
        
        # Calculate metrics
        results.append({
            'fold': fold,
            'RMSE': np.sqrt(mean_squared_error(true_values, imputed_values)),
            'MAE': mean_absolute_error(true_values, imputed_values),
            'R2': r2_score(true_values, imputed_values),
            'NRMSE': np.sqrt(mean_squared_error(true_values, imputed_values)) / np.mean(true_values)
        })
    
    return pd.DataFrame(results)

def save_model(model, name):
    """
    Save a model to a pickle file in src/model_pickles directory.
    
    Args:
        model: The model to save
        name: Name of the model file (without .pkl extension)
    """
    from datetime import datetime
    import os
    import pickle
    
    # Fixed path to model_pickles directory
    model_dir = 'src/model_pickles'
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{name}_{timestamp}.pkl'
    
    # Save the model
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved: {model_path}")
    return model_path