def validate_dtypes(df, dtypes):
    variables = df.columns
    assert all([var in dtypes for var in variables])
    
    supported = ["numeric", "categorical", "date", "text"]
    assert all([dtype in supported for dtype in dtypes.values()])
    
    return {var : dtype for var, dtype in dtypes.items() if var in variables}


def infer_dtypes(df) -> dict:
    raise NotImplementedError("Data types deduction is not supported.")
