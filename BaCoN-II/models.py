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
    
    # Keeping original dynamic configurations but ensuring values are compile-time constants
    filters = [int(f) for f in filters]
    kernel_sizes = [int(ks) for ks in kernel_sizes]
    strides = [int(s) for s in strides]
    pool_sizes = [int(ps) for ps in pool_sizes]
    strides_pooling = [int(sp) for sp in strides_pooling]

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i in range(n_conv):
        if swap_axes:
          
            ks = kernel_sizes[i]
            st = strides[i]
            ps = pool_sizes[i]
            spool = strides_pooling[i]
            clayer = tfp.layers.Convolution1DFlipout if bayesian else tf.keras.layers.Conv1D
            maxpoolL = tf.keras.layers.MaxPooling1D
        else:

            ks = (kernel_sizes[i], kernel_sizes[i])
            st = (strides[i], strides[i])
            ps = (pool_sizes[i], pool_sizes[i])
            spool = (strides_pooling[i], strides_pooling[i])
            clayer = tfp.layers.Convolution2DFlipout if bayesian else tf.keras.layers.Conv2D
            maxpoolL = tf.keras.layers.MaxPooling2D

       
        x = clayer(filters=filters[i], kernel_size=ks, strides=st, padding=padding, activation=activation)(x)
        
        if ps > 0:
            x = maxpoolL(pool_size=ps, strides=spool, padding=padding)(x)
        
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

    flayer = tf.keras.layers.GlobalAveragePooling1D() if swap_axes else tf.keras.layers.GlobalAveragePooling2D()
    x = flayer(x)
    
    for _ in range(n_dense):
        dlayer = tfp.layers.DenseFlipout if bayesian else tf.keras.layers.Dense
        x = dlayer(units=filters[-1], activation=activation)(x)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)
        if drop > 0.:
            x = tf.keras.layers.Dropout(drop)(x)

    outputs = dlayer(units=n_labels)(x)
    model = tf.keras.Model(inputs, outputs)

    return model

                        



# ------------------------------------------------------



def make_fine_tuning_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                           bayesian=True, trainable=True, drop=0.5, BatchNorm=True, include_last=False):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    
    for layer in base_model.layers[:-1]:
        layer.trainable = trainable
        x = layer(x, training=trainable)
        if not layer.trainable:
            print(f'Layer {layer.name} frozen.')

    
    if include_last:
        last_layer = base_model.layers[-1]
        last_layer.trainable = trainable
        x = last_layer(x, training=trainable)
        if not last_layer.trainable:
            print(f'Layer {last_layer.name} frozen.')

    
    if dense_dim > 0:
        dlayer = tfp.layers.DenseFlipout if bayesian else tf.keras.layers.Dense
        x = dlayer(dense_dim, activation=tf.nn.relu)(x)  

        if BatchNorm:
            x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        if drop != 0.:
            x = tf.keras.layers.Dropout(drop)(x)

    # Output Layer
    outputs = dlayer(n_out_labels, activation=None)(x)  
    if include_last:
        outputs = tf.nn.softmax(outputs)  

    model = tf.keras.Model(inputs, outputs)

    return model



# ------------------------------------------------------


def make_model_dummy(drop=0., 
                     n_labels=6, 
                     input_shape=(125, 4, 1), 
                     padding='valid', 
                     k_1=96, k_2=256, k_3=384,
                     activation=tf.nn.leaky_relu,
                     bayesian=False):
    # Initialize a Sequential model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=input_shape))  # Define the input layer

    # Choose between Bayesian and non-Bayesian layers based on the 'bayesian' flag
    # for the convolutional layer
    if bayesian:
        model.add(tfp.layers.Convolution2DFlipout(
            filters=k_1,  # Using k_1 for the number of filters
            kernel_size=(11, 1),  # Kernel size
            strides=(2, 1),  # Strides
            padding=padding,  # Padding
            activation=activation,# Activation function
        ))
    else:
        model.add(tf.keras.layers.Conv2D(
            filters=k_1,  # Using k_1 for the number of filters
            kernel_size=(11, 1),  # Kernel size
            strides=(2, 1),  # Strides
            padding=padding,  # Padding
            activation=activation  # Activation function
        ))

    # Add Global Average Pooling layer
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Define the output layer, choosing between Bayesian and non-Bayesian based on 'bayesian' flag
    if bayesian:
        model.add(tfp.layers.DenseFlipout(n_labels))
    else:
        model.add(tf.keras.layers.Dense(n_labels))

    return model



# ------------------------------------------------------



def make_unfreeze_model(base_model, input_shape, n_out_labels, dense_dim=0, 
                        bayesian=True, drop=0.5, BatchNorm=True):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Unfreeze and iterate through all layers except the last, integrating them into the new model
    for layer in base_model.layers[:-1]:
        layer.trainable = True  # Unfreeze layer
        x = layer(x, training=True)  # Apply layer to input

    # Unfreeze and include the last layer from the base model
    last_layer = base_model.layers[-1]
    last_layer.trainable = True  # Unfreeze last layer
    x = last_layer(x, training=True)  # Apply last layer to current model output

    # Add additional Dense layer with specified dimensions if requested
    if dense_dim > 0:
        dense_layer = tfp.layers.DenseFlipout(dense_dim, activation=tf.nn.relu) if bayesian else tf.keras.layers.Dense(dense_dim, activation=tf.nn.relu)
        x = dense_layer(x)

        # Optionally include Batch Normalization
        if BatchNorm:
            bn_layer = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)
            x = bn_layer(x)

        # Optionally include Dropout for regularization
        if drop != 0.:
            dropout_layer = tf.keras.layers.Dropout(drop)
            x = dropout_layer(x)

    # Define the output layer
    output_layer = tfp.layers.DenseFlipout(n_out_labels) if bayesian else tf.keras.layers.Dense(n_out_labels)
    outputs = output_layer(x)

    # Construct and return the new model
    model = tf.keras.Model(inputs, outputs)

    return model



