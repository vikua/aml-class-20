from sklearn.pipeline import Pipeline


def build_pipeline(model_type):
    assert assert_supported(model_type)


def build_param_grid(model_type):
    assert_supported(model_type)


def supported(model_type):
    return model_type in ["gbm", "rf", "svr", "lr"]


def assert_supported(model_type):
    assert supported(model_type), f"{model_type} is not supported"

