import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from utils.features import generate_features


def clean_data(df):
    x = df.copy()
    if not x.index.is_unique:
        x = x[~x.index.duplicated(keep='first')]
    if x.eq(0).any().any():
        x.replace(0, np.nan, inplace=True)
    x.dropna(inplace=True)
    return x


def generate_labels(df, min_return=0.005):
    df['target'] = np.where(df['ReturnPct'] >= min_return, 1, 0)
    return df


def filter_features_dates(df, dates):
    df = df[df.index.isin(dates)]
    df = df.iloc[:, 5:]
    df = generate_labels(df)
    return df


def get_features_col():
    original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    _ = pd.read_csv('dataset_eval.csv')
    features_columns = list(_.columns)
    features_columns.remove('target')
    selected_columns = original_columns + features_columns
    return selected_columns


def get_dates_in_out(df, ticker, min_date, max_date):
    dates_in = df.loc[(df['RIC'] == ticker) & (df['Action'] == 'IN')].index.tolist()
    dates_out = df.loc[(df['RIC'] == ticker) & (df['Action'] == 'OUT')].index.tolist()
    dates_in = [x.strftime('%Y-%m-%d') for x in dates_in]
    dates_out = [x.strftime('%Y-%m-%d') for x in dates_out]
    if len(dates_in) == 0:
        dates_in.append(min_date.strftime(format='%Y-%m-%d'))
    if (len(dates_out) == 0) or (len(dates_in) > len(dates_out)):
        dates_out.append(max_date.strftime(format='%Y-%m-%d'))
    return dates_in, dates_out


def filter_df_dates(df, dates_in, dates_out, ticker):
    x = pd.DataFrame()
    for f_in, f_out in zip(dates_in, dates_out):
        tmp = df.loc[(df.index >= f_in) & (df.index <= f_out)]
        if len(tmp) == 0:
            continue
        tmp = generate_features(tmp, ticker)
        x = x.append(tmp)
    if not x.index.is_unique:
        x = x[~x.index.duplicated(keep='first')]
    # x = clean_data(x)
    return x


def load_data(file, test_size=0.2):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    n_features = data.shape[1] - 1
    x = data[:, 0:n_features]
    y = data[:, -1].reshape(-1, 1)
    return train_test_split(x, y, test_size=test_size)


def load_dataset(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    n_features = data.shape[1] - 1
    x = data[:, 0:n_features]
    y = data[:, -1].reshape(-1, 1)
    return x, y


def create_train_test_data(file):
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    train_dir = os.path.join(os.getcwd(), 'data/train')
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(os.getcwd(), 'data/test')
    os.makedirs(test_dir, exist_ok=True)

    raw_dir = os.path.join(os.getcwd(), 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)

    dataset_dir = os.path.join(os.getcwd(), 'data/dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    x, y = load_dataset(file)
    x_train, x_test, y_train, y_test = load_data(file)

    np.save(os.path.join(dataset_dir, 'x.npy'), x)
    np.save(os.path.join(dataset_dir, 'y.npy'), y)
    np.save(os.path.join(raw_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(raw_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(test_dir, 'x_test.npy'), x_test)

    pickle.dump(scaler, open('models/scaler_models/scaler.pkl', 'wb'))
