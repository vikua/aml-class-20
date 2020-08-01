def validate_dtypes(df, dtypes):
    variables = df.columns
    assert all([var in dtypes for var in variables])
    
    supported = ["numeric", "categorical", "date", "text"]
    assert all([dtype in supported for dtype in dtypes.values()])
    
    return {var : dtype for var, dtype in dtypes.items() if var in variables}


def infer_dtypes(df) -> dict:
    raise NotImplementedError("Data types deduction is not supported.")


class Printer:
    enabled: bool
    
    def __init__(self, enabled: bool):
        self.enabled = enabled
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def __call__(self, x, *args, **kwargs):
        if self.enabled:
            print(x, *args, **kwargs)
