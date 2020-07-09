from pipeline import AMLPipeline

TARGET = 'loss'
DATA_PATH = 'https://s3.amazonaws.com/datarobot_public_datasets' \
            '/DR_Demo_Fire_Ins_Loss_only.csv '
MODEL_TYPE = 'lr'
# MODEL_TYPE = 'rf'
# MODEL_TYPE = 'gb'


if __name__ == '__main__':
    p = AMLPipeline(
        target=TARGET,
        data_path=DATA_PATH,
        model_type=MODEL_TYPE,
    )

    p.run()
