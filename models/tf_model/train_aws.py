import os
import argparse
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape, ))
    hidden_1 = tf.keras.layers.Dense(60)(inputs)
    batchnorm_1 = tf.keras.layers.BatchNormalization()(hidden_1)
    activation_1 = tf.keras.layers.Activation('relu')(batchnorm_1)
    dropout_1 = tf.keras.layers.Dropout(.2)(activation_1)

    hidden_2 = tf.keras.layers.Dense(30)(dropout_1)
    batchnorm_2 = tf.keras.layers.BatchNormalization()(hidden_2)
    activation_2 = tf.keras.layers.Activation('relu')(batchnorm_2)
    dropout_2 = tf.keras.layers.Dropout(.2)(activation_2)

    hidden_3 = tf.keras.layers.Dense(15, activation='relu')(dropout_2)
    batchnorm_3 = tf.keras.layers.BatchNormalization()(hidden_3)
    activation_3 = tf.keras.layers.Activation('relu')(batchnorm_3)
    dropout_3 = tf.keras.layers.Dropout(.2)(activation_3)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_3)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # raw_data directories
    parser.add_argument('--dataset', type=str, default='./data/dataset/')
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

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


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == "__main__":
    args, _ = parse_args()

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

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

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(x_test, y_test))

        # evaluate on test set
        scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
        print("\nTest Accuracy :", scores)

        # save model
        model.save(args.model_dir + '/1')
