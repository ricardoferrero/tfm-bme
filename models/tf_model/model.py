import tensorflow as tf


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
