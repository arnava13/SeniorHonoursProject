#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:20:20 2020

@author: Michi
"""
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tf.enable_v2_behavior()
tfd = tfp.distributions




def make_model(model_name, **params):
    
    if model_name=='dummy':
        return make_model_dummy(**params)
    elif model_name=='custom':
        return make_custom_model(**params)
    else:
        raise ValueError('Enter a valid model name.')



# ------------------------------------------------------

def make_custom_model(drop=0.5, 
                      n_labels=6, 
                      input_shape=(100, 4), 
                      padding='valid', 
                      filters=(8, 16, 32),
                      kernel_sizes=(10, 5, 2),
                      strides=(2, 2, 1),
                      pool_sizes=(2, 2, 0),
                      strides_pooling=(2, 1, 0),
                      activation=tf.nn.leaky_relu,
                      bayesian=True, 
                      n_dense=1, 
                      swap_axes=True, 
                      BatchNorm=True):
    n_conv = len(kernel_sizes)
    
    # Ensure compile-time constants for TPU compatibility
    filters = tuple([int(f) for f in filters])
    kernel_sizes = tuple([int(ks) for ks in kernel_sizes])
    strides = tuple([int(s) for s in strides])
    pool_sizes = tuple([int(ps) for ps in pool_sizes])
    strides_pooling = tuple([int(sp) for sp in strides_pooling])

    if swap_axes:
        print('using 1D layers and %s channels' % input_shape[-1])
        is_1D = True
        flayer = tf.keras.layers.GlobalAveragePooling1D()
        maxpoolL = tf.keras.layers.MaxPooling1D
        
        # Choose layer type based on bayesian flag
        clayer = tfp.layers.Convolution1DFlipout if bayesian else tf.keras.layers.Conv1D
        dlayer = tfp.layers.DenseFlipout if bayesian else tf.keras.layers.Dense
    else:
        print('using 2D layers and %s channels' % input_shape[-1])
        is_1D = False
        flayer = tf.keras.layers.GlobalAveragePooling2D()
        maxpoolL = tf.keras.layers.MaxPooling2D
        
        # Choose layer type based on bayesian flag
        clayer = tfp.layers.Convolution2DFlipout if bayesian else tf.keras.layers.Conv2D
        dlayer = tfp.layers.DenseFlipout if bayesian else tf.keras.layers.Dense
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i in range(n_conv):
        # Define compile-time constants for layer arguments
        ks = kernel_sizes[i] if is_1D else (kernel_sizes[i], 1)
        st = strides[i] if is_1D else (strides[i], 1)
        ps = pool_sizes[i] if is_1D else (pool_sizes[i], 1)
        spool = strides_pooling[i] if is_1D else (strides_pooling[i], 1)
        
        # Convolutional Layer
        x = clayer(filters=filters[i], kernel_size=ks, strides=st, padding=padding, activation=activation)(x)
        
        # Pooling Layer
        if ps != 0:
            x = maxpoolL(pool_size=ps, strides=spool, padding=padding)(x)
        
        # Batch Normalization
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

    # Global Average Pooling
    x = flayer(x)
    
    # Dense Layers
    for _ in range(n_dense):
        x = dlayer(filters[-1], activation=activation)(x)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)
        if drop != 0.:
            x = tf.keras.layers.Dropout(drop)(x)

    # Output Layer
    outputs = dlayer(n_labels)(x)
    model = tf.keras.Model(inputs, outputs)

    return model
                        



# ------------------------------------------------------



def make_fine_tuning_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                           bayesian=True, trainable=True, drop=0.5, BatchNorm=True, include_last=False):
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Ensuring layer's trainability as per the 'trainable' flag
    for layer in base_model.layers[:-1]:  # Iterate through all but the last layer
        layer.trainable = trainable
        x = layer(x, training=trainable)
        if not layer.trainable:
            print(f'Layer {layer.name} frozen.')

    # Optionally include the last layer of the base model
    if include_last:
        last_layer = base_model.layers[-1]
        last_layer.trainable = trainable
        x = last_layer(x, training=trainable)
        if not last_layer.trainable:
            print(f'Layer {last_layer.name} frozen.')
        x = tf.nn.softmax(x)  # Ensure output is passed through softmax

    # Add additional Dense layer if 'dense_dim' is specified
    if dense_dim > 0:
        if bayesian:
            x = tfp.layers.DenseFlipout(dense_dim, activation=tf.nn.relu)(x)  # Using relu for example
        else:
            x = tf.keras.layers.Dense(dense_dim, activation=tf.nn.relu)(x)

        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        if not bayesian and drop != 0.:
            x = tf.keras.layers.Dropout(drop)(x)

    # Output Layer
    if bayesian:
        outputs = tfp.layers.DenseFlipout(n_out_labels)(x)
    else:
        outputs = tf.keras.layers.Dense(n_out_labels)(x)

    model = tf.keras.Model(inputs, outputs)

    return model



# ------------------------------------------------------


def make_model_dummy(drop=0., 
                          n_labels=6, 
                          input_shape=( 125, 4, 1), 
                          padding='valid', 
                          k_1=96, k_2 = 256, k_3  =384,
                          activation=tf.nn.leaky_relu,
                          bayesian=False
                          ):
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=input_shape))

    # 1st Convolutional Layer
    if bayesian:
        layer1 = tfp.layers.Convolution2DFlipout(
            filters=k_1,  # Using k_1 for the number of filters as an example
            kernel_size=(11, 1),
            strides=(2, 1),
            padding=padding,
            activation=activation
        )
    else:
        layer1 = tf.keras.layers.Conv2D(
            filters=k_1,  # Using k_1 for the number of filters as an example
            kernel_size=(11, 1),
            strides=(2, 1),
            padding=padding,
            activation=activation
        )
    model.add(layer1)

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Output Layer
    if bayesian:
        outLayer = tfp.layers.DenseFlipout(n_labels)
    else:
        outLayer = tf.keras.layers.Dense(n_labels)
    model.add(outLayer)

    return model



# ------------------------------------------------------



def make_unfreeze_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                        bayesian=True, drop=0.5, BatchNorm=True):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for layer in base_model.layers[:-1]:  # go through until the last layer
        layer.trainable = True
        x = layer(x, training=True)

    # Assuming the last layer is correctly configured for the desired output
    # and making it trainable as well
    last_layer = base_model.layers[-1]
    last_layer.trainable = True
    x = last_layer(x, training=True)

    # If there's a need to add additional layers after unfreezing
    if dense_dim > 0:
        if bayesian:
            # Using compile-time constants for Bayesian layers
            x = tfp.layers.DenseFlipout(dense_dim, activation=tf.nn.relu)(x)
        else:
            # Non-Bayesian path
            x = tf.keras.layers.Dense(dense_dim, activation=tf.nn.relu)(x)

        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        if not bayesian and drop != 0.:
            x = tf.keras.layers.Dropout(drop)(x)

    # Assuming you want to redefine the output layer
    if bayesian:
        outputs = tfp.layers.DenseFlipout(n_out_labels)(x)
    else:
        outputs = tf.keras.layers.Dense(n_out_labels)(x)

    model = tf.keras.Model(inputs, outputs)

    return model



