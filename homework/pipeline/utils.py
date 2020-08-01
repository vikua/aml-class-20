
def get_columns_from_type(dict, dtype):
    return [k for k, v in dict.items() if v == dtype]
