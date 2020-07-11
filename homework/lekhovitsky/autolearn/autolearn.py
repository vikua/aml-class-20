import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from .dtypes import validate_dtypes, infer_dtypes
from .pipelines import build_pipeline, build_param_grid


def train_regressor(path_or_url, 
                    target,
                    explanatory=None,
                    scoring='mean_squared_error',
                    dtypes='infer',
                    test_size=0.2,
                    verbose=False,
                    random_state=42):
    """Automatically train a regression model for a given dataset.
    
    Parameters
    ----------
    path_or_url : str
        Filepath or URL to a valid CSV file.
    target : str
        Name of the column to use as target variable.
    explanatory : iterable of str, optional
        List of data frame columns to use as explanatory
        variables. 
        If None, use all variables except target.
    scoring : str, optional
        Any sklearn-compatible regression scoring.
        Default 'mean_squared_error'.
    dtypes : 'infer' or dict, optional
        If 'infer', try to automatically deduce the types of
        explanatory variables based on their values. 
        If dict, mapping from variable name to its type, which
        is one of 'numeric', 'categorical', 'date' and 'text'.
    test_size : float, optional
        Fraction of held-out test set. 
    verbose : bool, optional
        If True, display some messages.
    random_state : int, optional
        Set RNG seed to this number.
    
    Returns
    -------
    
    """
    np.random.seed(random_state)
    
    data = pd.read_csv(path_or_url)
    variables = list(data.columns)
    assert target in variables, f"{target} is not a name of any columns of the dataset"
    
    if explanatory is not None:
        assert len(explanatory), "List of explanatory variables must not be empty"
        assert all([var in variables for var in explanatory])
    else:
        explanatory = list(variables)
        explanatory.remove(target)
    
    X, y = data[explanatory], data[target]
    
    if dtypes == 'infer':
        dtypes = infer_dtypes(X)
        if verbose:
            print("Inferred types of variables:")
            for var, dtype in dtypes.items():
                print(f"    {var: dtype}")
    else:
        dtypes = validate_dtypes(X, dtypes)
    
    assert 0 < test_size < 1
    (X_train, X_test, 
     y_train, y_test
    ) = train_test_split(
        X, y, test_size=test_size, 
        random_state=random_state)
    
    # nested-CV parameters search
    models = ['gbm', 'lr', 'svr', 'rf']
    model_scores = {}
    for model in models:
        pipeline = build_pipeline(model)
        estimator = GridSearchCV(pipeline, param_grid=build_param_grid(model), cv=4)
        cv_score = cross_val_score(estimator, X_train, y_train, cv=4)
        model_scores[model] = cv_score.mean()
    
    # final model scoring
    best_model = max(model_scores, key=model_scores.get)
    estimator = build_pipeline(best_model)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test, y_test)
    
    # report scores
    