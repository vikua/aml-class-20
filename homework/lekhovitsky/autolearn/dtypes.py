from sklearn.base import TransformerMixin


def validate_dtypes(df, dtypes):
    variables = df.columns
    assert all([var in dtypes for var in variables])
    
    supported = ["numeric", "categorical", "date", "text"]
    assert all([dtype in supported for dtype in dtypes.values()])


def infer_dtypes(df) -> dict:
    raise NotImplementedError("Data types deduction is not supported.")

    
class CategoricalTransformer(TransformerMixin):
    pass


class NumericTransformer(TransformerMixin):
    pass


class TextTransformer(TransformerMixin):
    pass


class DateTransformer(TransformerMixin):
    pass
