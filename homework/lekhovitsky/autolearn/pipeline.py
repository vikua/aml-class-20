from scipy.stats import loguniform, uniform

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


ESTIMATORS = {
    'svr': SVR(),
    'lr': ElasticNet(),
    'rf': RandomForestRegressor(),
    'gbm': GradientBoostingRegressor(),
}


def build_pipeline(model_type, dtypes):
    assert_supported(model_type)
    
    num_variables = filter_variables(dtypes, 'numeric')
    num_pipeline = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='constant', fill_value=-99999))
    ])
    if model_type in {'lr', 'svr'}:
        num_pipeline.steps.append(('scaler', StandardScaler()))
    
    cat_variables = filter_variables(dtypes, 'categorical')
    cat_pipeline = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    txt_variables = filter_variables(dtypes, 'text')
    txt_pipeline = Pipeline(steps=[
        ('tf-idf', TfidfVectorizer())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_variables),
        ('cat', cat_pipeline, cat_variables),
        #('txt', txt_pipeline, txt_variables)
    ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', ESTIMATORS[model_type])
    ])


def build_params_space(model_type):
    assert_supported(model_type)
    
    if model_type == 'svr':
        space = {
            'estimator__kernel': ['poly', 'rbf'],
            'estimator__C': loguniform(1e-3, 1e3),
            'estimator__degree': range(1, 8),
        }
    
    if model_type == 'lr':
        space = {
            'estimator__alpha': loguniform(1e-3, 1e1),
            'estimator__l1_ratio': uniform(0, 1),
        }
    
    if model_type == 'rf':
        space = {
            'estimator__n_estimators': range(50, 200),
            'estimator__max_depth': range(3, 10),
            'estimator__criterion': ['mse', 'mae'],
            'estimator__max_features': ['auto', 'sqrt', 'log2']
        }
    
    if model_type == 'gbm':
        space = {
            'estimator__n_estimators': range(25, 200),
            'estimator__max_depth': range(3, 8),
            'estimator__learning_rate': loguniform(1e-3, 1e-1)
        }
        
    return space
    

def filter_variables(dtypes: dict, dtype: str) -> dict:
    return [v for v, d in dtypes.items() if d == dtype]


def supported(model_type):
    return model_type in ESTIMATORS


def assert_supported(model_type):
    assert supported(model_type), f"{model_type} is not supported"
