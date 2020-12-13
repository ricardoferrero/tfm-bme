import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from model import create_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # raw_data directories
    parser.add_argument('--dataset', type=str, default='data/dataset/')
    parser.add_argument('--train', type=str, default='data/train/')
    parser.add_argument('--test', type=str, default='data/test/')

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_train_data(train_dir):
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape, 'y train', y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):
    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape, 'y test', y_test.shape)

    return x_test, y_test


def get_dataset(dataset_dir):
    x = np.load(os.path.join(dataset_dir, 'x.npy'))
    y = np.load(os.path.join(dataset_dir, 'y.npy'))
    print('x', x.shape, 'y', y.shape)
    return x, y


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == '__main__':
    args, _ = parse_args()

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)
    x, y = get_dataset(args.dataset)

    device = '/cpu:0'
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        model = create_model(x_train.shape[1])
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

        tscv = TimeSeriesSplit()

        for i, (train_index, test_index) in enumerate(tscv.split(x), 1):

            # Scaler
            scaler = StandardScaler()

            x_train = scaler.fit_transform(x[train_index])
            x_test = scaler.transform(x[test_index])

            y_train, y_test = y[train_index], y[test_index]

            os.makedirs(f'models/tf_model/checkpoints/model_{i}', exist_ok=True)

            checkpoint_path = os.path.join(f'models/tf_model/checkpoints/model_{i}')
            csvlog_path = os.path.join(f'models/tf_model/csv_logs/training_{i}.csv')

            calls = [EarlyStopping(monitor='val_accuracy', patience=10),
                     ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
                     LearningRateScheduler(scheduler, verbose=0),
                     CSVLogger(csvlog_path)]

            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_test, y_test), callbacks=calls)

            scores = model.evaluate(x_test, y_test, batch_size, verbose=2)

            pickle.dump(scaler, open(f'models/scaler_models/scaler_{i}.pkl', 'wb'))

            print("\nTest Accuracy :", scores)
