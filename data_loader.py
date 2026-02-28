import pandas as pd

def load_and_preprocess_data(config):

    try:
        data = pd.read_excel(config["INPUT_PATH"], dtype='float64')
        X = data.iloc[:, 0:6]
        y = data.iloc[:, 6].values.ravel()
        return X, y
    except FileNotFoundError:
        print(f"Error: File not found - {config['INPUT_PATH']}")
        exit()