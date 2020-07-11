from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


ESTIMATORS = {
    "svr": SVR(),
    "lr": LinearRegression(),
    "rf": RandomForestRegressor(),
    "gbr": GradientBoostingRegressor(),
}


def build_pipeline(model_type, dtypes):
    assert assert_supported(model_type)
    
    num_features = filter_variables(dtypes, 'numeric')
    cat_features = filter_variables(dtypes, 'categorical')
    txt_features = filter_variables(dtypes, 'text')
    
    steps = [
        ('num_imputer', SimpleImputer(strategy='constant', fill_value=-99999))
    ]
    if model_type in {'lr', 'svr'}:
        steps.append(('scaler', StandardScaler()))
    num_pipeline = Pipeline(steps=steps)
    
    cat_pipeline = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('fx_selection', SelectFromModel(ElasticNet(alpha=0.1), max_features=50, threshold=None))
    ])

    txt_pipeline = Pipeline(steps=[
        ('tf-idf', TfidfVectorizer())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('txt', txt_pipeline, txt_features)
    ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', ESTIMATORS[model_type])
    ])


def build_param_grid(model_type):
    assert_supported(model_type)

    
def filter_variables(dtypes: dict, dtype: str) -> dict:
    return [var for var, dtype in dtypes.items() if dtype == dtype]


def supported(model_type):
    return model_type in ESTIMATORS


def assert_supported(model_type):
    assert supported(model_type), f"{model_type} is not supported"
