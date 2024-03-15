#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:21:14 2020

@author: Michi
@editor: Arnav, March 2024, add TPU support
"""
import argparse
import os
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tf.enable_v2_behavior()
tfd = tfp.distributions
from data_generator import create_generators
from models import *
from utils import DummyHist, plot_hist, str2bool, Logger, get_flags
import sys
import time

@tf.function
def train_on_batch(IDs, x, y, train_generator, epoch, model, optimizer, loss, train_acc_metric, train_loss_metric, bayesian=False, n_train_example=60000, TPU=False, strategy=None, batch_size=2500, save_indexes=False):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        for layer in model.layers:  # Supports frozen weights
            x = layer(x, training=layer.trainable)
        logits = x
        if bayesian:
            # Assuming model.losses are appropriately scaled
            kl = sum(model.losses) / n_train_example
            loss_value = loss(y, logits, kl, TPU=TPU, batch_size=batch_size)
        else:
            loss_value = loss(y, logits, TPU=TPU, batch_size=batch_size)
        train_acc_metric.update_state(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Compute accuracy
        proba = tf.nn.softmax(logits)
        prediction = tf.argmax(proba, axis=1)
        train_acc_metric.update_state(tf.argmax(y, axis=1), prediction)        
    if save_indexes:
        train_generator.write_indexes(epoch, IDs)
    train_loss_metric.update_state(loss)

@tf.function
def val_step(IDs, x, y, val_generator, epoch, model, loss, val_loss_metric, val_acc_metric, bayesian=False, n_val_example=10000, TPU=False, strategy=None, batch_size=2500, save_indexes=False):
    dataset = val_generator.dataset
    if TPU:
        with strategy.scope():
            val_logits = model(x, training=False)
            if bayesian:
                val_kl = sum(model.losses)/n_val_example
                val_loss_value = loss(y, val_logits, val_kl, TPU=TPU, batch_size = batch_size)
            else:
                val_loss_value = loss(y, val_logits, TPU=TPU, batch_size = batch_size)
            val_proba = tf.nn.softmax(val_logits)
            val_prediction = tf.argmax(val_proba, axis=1)
            val_acc_metric.update_state(tf.argmax(y, axis=1), val_prediction)
    else:
        val_acc_metric.update_state(tf.argmax(y, axis=1), val_prediction)
        val_logits = model(x, training=False)
        if bayesian:
            val_kl = sum(model.losses)/n_val_example
            val_loss_value = loss(y, val_logits, val_kl, TPU=TPU, strategy=strategy, batch_size = batch_size)
        else:
            val_loss_value = loss(y, val_logits, TPU=TPU, strategy=strategy, batch_size = batch_size)
        val_proba = tf.nn.softmax(val_logits)
        val_prediction = tf.argmax(val_proba, axis=1)
    if save_indexes:
        val_generator.write_indexes(epoch, IDs)
    val_loss_metric.update_state(val_loss_value)
    val_acc_metric.update_state(y, val_logits)


@tf.function
def my_loss(y, logits, TPU=False, batch_size=None):
    if TPU:
        loss_f = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #tf.nn.softmax_cross_entropy_with_logits(y, logits)
    else:
        loss_f = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #tf.nn.softmax_cross_entropy_with_logits(y, logits)
    loss_per_example = loss_f(y, logits)
    
    if TPU:
        # Use the strategy to reduce the loss across replicas
        loss = tf.nn.compute_average_loss(loss_per_example, global_batch_size=batch_size)
    else:
        loss = loss_per_example

    return loss


@tf.function
def ELBO(y, logits, kl, TPU=False, batch_size=None):
    neg_log_likelihood = my_loss(y, logits, TPU=TPU, batch_size=batch_size)
    elbo_loss = neg_log_likelihood + kl
    return elbo_loss

def my_train(model, optimizer, loss,
             epochs, 
             train_generator, 
             val_generator, manager, ckpt, train_loss_metric,            
             train_acc_metric, val_loss_metric, val_acc_metric, TPU=False, strategy=None,
             restore=False, patience=100,
             bayesian=False, save_ckpt=False, decayed_lr_value=None, save_indexes = False
              ):
  fname_hist = manager.directory+'/hist'
  fname_idxs_train = manager.directory+'/idxs_train.txt'
  fname_idxs_val = manager.directory+'/idxs_val.txt'
  if not restore:
      history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy':[] }
      best_loss=np.infty
      print("Initializing checkpoint from scratch.")
  else:
      # this is to fix bug in restoring optimizer. See https://github.com/tensorflow/tensorflow/issues/33150
      #print('Training on one batch to properly restore model....') 
      #x, y = train_generator[0]
      #_ = train_on_batch(x, y, model, optimizer, loss, train_acc_metric, 
      #                  bayesian=bayesian, n_train_example=train_generator.batch_size*train_generator.n_batches)
      #train_acc_metric.reset_states()
      
      #optimizer.iterations # this is to fix bug in restoring optimizer. See https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
      print('Restoring ckpt...')
      ckpt.restore(manager.latest_checkpoint)
      print('ckpt step: %s' %ckpt.step)
      hist_start=int(ckpt.step)
      print('Starting from history at step %s' %hist_start)
      history = {'loss': np.loadtxt(fname_hist+'_loss.txt').tolist()[0:hist_start], 
                 'val_loss': np.loadtxt(fname_hist+'_val_loss.txt').tolist()[0:hist_start], 
                 'accuracy': np.loadtxt(fname_hist+'_accuracy.txt').tolist()[0:hist_start], 
                 'val_accuracy':np.loadtxt(fname_hist+'_val_accuracy.txt').tolist()[0:hist_start] }
      for key in history.keys():
        fname = fname_hist+'_'+key+'.txt'
        fname_new = fname_hist+'_'+key+'_original.txt'
        os.rename(fname, fname_new)
      print('Saved copy of original histories.')
      if manager.latest_checkpoint:
          print("Restoring checkpoint from {}".format(manager.latest_checkpoint))
          best_train_loss = history['loss'][-1]
          best_loss = history['val_loss'][-1]
          print('Starting from  (loss, val_loss) =  %.4f, %.4f' %(best_train_loss, best_loss ))
      else:
          print("Checkpoint not found. Initializing checkpoint from scratch.")
      
      print('Last learning rate was %s' %ckpt.optimizer.learning_rate)
      #if decayed_lr_value is not None:
        #lr_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.lr, len(training_generator), FLAGS.decay)
      #  ckpt.optimizer.learning_rate = decayed_lr_value(hist_start) #FLAGS.lr
      #  print('Learning rate set to %s' %ckpt.optimizer.learning_rate)
      #else:
      #    print('Re-starting from this value for the learing rate')
  
  n_val_example=val_generator.batch_size*val_generator.n_batches
  n_train_example=train_generator.batch_size*train_generator.n_batches
  count = 0
  batchstep = 0
  for epoch in range(epochs):
    print("Epoch %d" % (epoch,))
    start_time = time.time()
    train_loss_metric.reset_states()
    train_acc_metric.reset_states()
    val_loss_metric.reset_states()
    val_acc_metric.reset_states()
    if TPU:
        with strategy.scope():
            for IDs, x_batch_train, y_batch_train in train_generator.dataset:
                strategy.run(train_on_batch, args=(IDs, x_batch_train, y_batch_train, model, optimizer, loss, train_acc_metric, train_loss_metric, bayesian, n_train_example, train_generator.batch_size))
            for IDs, x_batch_val, y_batch_val in val_generator.dataset:
                strategy.run(val_step, args=(IDs, x_batch_val, y_batch_val, model, loss, val_acc_metric, val_loss_metric, bayesian, n_val_example, val_generator.batch_size))
            val_loss_value = strategy.reduce(tf.distribute.ReduceOp.MEAN, val_loss_metric.result(), axis=None)
            loss_value = strategy.reduce(tf.distribute.ReduceOp.MEAN, train_loss_metric.result(), axis=None)
    else:
        for IDs, x_batch_train, y_batch_train in train_generator.dataset:
            train_on_batch(IDs, x_batch_train, y_batch_train, model, optimizer, loss, train_acc_metric, train_loss_metric, bayesian, n_train_example, val_generator.batch_size)
        for IDs, x_batch_val, y_batch_val in val_generator.dataset:
            val_step(x_batch_val, y_batch_val, model, loss, val_acc_metric, val_loss_metric, bayesian, n_val_example, val_generator.batch_size)
        val_loss_value = val_loss_metric.result()
        loss_value = train_loss_metric.result()

    if val_loss_value.numpy()<best_loss: #int(ckpt.step) % 10 == 0:
        if save_ckpt and not TPU:
            save_path = manager.save()
            print("Validation loss decreased. Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        else:
            #print('Creating directory %s' %manager.directory)
            tf.io.gfile.makedirs(manager.directory)

        best_loss = val_loss_value.numpy()      
        #print("New loss {:1.2f}".format(best_loss))
        count = 0
    else:
        count +=1
        print('Loss did not decrease. Count = %s' %count)
        if count==patience:
            print('Max patience reached. ')
            break
    
    
    
    ckpt.step.assign_add(1)

    train_acc = train_acc_metric.result().numpy()
    train_loss = loss_value.numpy()
    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    train_acc_metric.reset_states()

    val_acc = val_acc_metric.result().numpy()
    val_loss = val_loss_value.numpy()
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    val_acc_metric.reset_states()
    

    if epoch==0:
        if restore:          
            for key in history.keys():
                fname = fname_hist+'_'+key+'.txt'
                with open(fname, 'a') as fh:
                    for el in history[key][:-1]:
                        fh.write(str(el) +'\n')
            print('Re-wrote histories until epoch %s' %str(len(history['val_accuracy'][:-1])) )
                    
        with open(fname_idxs_train, 'a') as fit:
            for ID in train_generator.list_IDs:
                fit.write(str(ID) +'\n')
        with open(fname_idxs_val, 'a') as fiv:
            for ID in val_generator.list_IDs:   
                fiv.write( str(ID) +'\n' )
    
    for key in history.keys():
            fname = fname_hist+'_'+key+'.txt'
            with open(fname, 'a') as fh:
                fh.write(str(history[key][-1])+'\n') 
                
    ###
    # Uncomment if training in a jupyter notebook, to print the status on epoch bar
    ###
    #epoch_bar.set_postfix(train_loss=loss_value.numpy(), val_loss=val_loss_value.numpy(), 
    #                      train_accuracy = train_acc.numpy(), val_accuracy=val_acc.numpy())
    #print("Time taken: %.2fs" % (time.time() - start_time))
    print("Time:  %.2fs, ---- Loss: %.4f, Acc.: %.4f, Val. Loss: %.4f, Val. Acc.: %.4f\n" % (time.time() - start_time, train_loss, train_acc, val_loss, val_acc))

  return model, history


def compute_loss(generator, model, bayesian=False, TPU=False, strategy=None):
    x_batch_train, y_batch_train = generator[0]
    logits = model(x_batch_train, training=False)
    if bayesian:
            kl = sum(model.losses)/generator.batch_size/generator.n_batches
            loss_0 = ELBO(y_batch_train, logits, kl, TPU=TPU, batch_size=generator.batch_size)
    else:
            loss_0 = my_loss(y_batch_train, logits, TPU=TPU, batch_size=generator.batch_size)
    return loss_0




def main():
    
    in_time=time.time()
        
    ## Read params from stdin
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bayesian", default=True, type=str2bool, required=False)
    
    parser.add_argument("--test_mode", default=False, type=str2bool, required=False)
    parser.add_argument("--n_test_idx", default=2, type=int, required=False)
    parser.add_argument("--seed", default=1312, type=int, required=False)
    
    parser.add_argument("--fine_tune", default=False, type=str2bool, required=False)
    parser.add_argument("--one_vs_all", default=False, type=str2bool, required=False)
    parser.add_argument("--c_0", nargs='+', default=['lcdm'], required=False)
    parser.add_argument("--c_1", nargs='+', default=['fR', 'dgp', 'wcdm', 'rand', 'ds'], required=False)
    parser.add_argument("--dataset_balanced", default=False, type=str2bool, required=False)
    parser.add_argument("--include_last", default=False, type=str2bool, required=False)
    
    
    parser.add_argument("--log_path", default='', type=str, required=False)
    parser.add_argument("--restore", default=False, type=str2bool, required=False)

    
    # FNAMES ETC
    parser.add_argument("--fname", default='my_model', type=str, required=False)
    parser.add_argument("--model_name", default='custom', type=str, required=False)
    parser.add_argument("--my_path", default=None, type=str, required=False)
    parser.add_argument("--DIR", default='data/train_data/', type=str, required=False)
    parser.add_argument("--TEST_DIR", default='data/test_data/', type=str, required=False)  
    parser.add_argument("--models_dir", default='models/', type=str, required=False)
    parser.add_argument("--save_ckpt", default=True, type=str2bool, required=False)
    parser.add_argument("--save_indexes", default=False, type=str2bool, required=False)
    parser.add_argument("--out_path_overwrite", default=False, type=str2bool, required=False)
    parser.add_argument("--curves_folder", default=None, type=str, required=False)
    parser.add_argument("--save_processed_spectra", default=False, type=str2bool, required=False)    

    
    # INPUT DATA DIMENSION
    parser.add_argument("--im_depth", default=500, type=int, required=False)
    parser.add_argument("--im_width", default=1, type=int, required=False)
    parser.add_argument("--im_channels", default=4, type=int, required=False)
    parser.add_argument("--swap_axes", default=True, type=str2bool, required=False)
    
    
    
    # PARAMETERS TO GENERATE DATA
    parser.add_argument("--sort_labels", default=True, type=str2bool, required=False)
    parser.add_argument("--norm_data_name", default=None, type=str, required=False)
    parser.add_argument("--normalization", default='stdcosmo', type=str, required=False)
    parser.add_argument("--sample_pace", default=1, type=int, required=False)
    
    parser.add_argument("--k_max", default=2.5, type=float, required=False)
    parser.add_argument("--k_min", default=0, type=float, required=False)
    parser.add_argument("--i_max", default=None, type=int, required=False)
    parser.add_argument("--i_min", default=None, type=int, required=False)
    
    parser.add_argument("--add_noise", default=True, type=str2bool, required=False)
    parser.add_argument("--n_noisy_samples", default=10, type=int, required=False)
    parser.add_argument("--add_shot", default=True, type=str2bool, required=False)
    parser.add_argument("--add_sys", default=True, type=str2bool, required=False)
    parser.add_argument("--add_cosvar", default=True, type=str2bool, required=False)
    parser.add_argument("--sigma_sys", default=None, type=float, required=False)
    parser.add_argument("--sys_scaled", default=None, type=str2bool, required=False)
    parser.add_argument("--sys_factor", default=None, type=float, required=False)
    parser.add_argument("--sys_max", default=None, type=str2bool, required=False)  
    parser.add_argument("--sigma_curves", default=None, type=float, required=False) 
    parser.add_argument("--sigma_curves_default", default=None, type=float, required=False)
    parser.add_argument("--rescale_curves", default=None, type=str, required=False)
    
    parser.add_argument('--z_bins', nargs='+', default=[0,1,2,3], required=False)

    
    # NET STRUCTURE
    parser.add_argument("--n_dense", default=1, type=int, required=False)
    parser.add_argument("--filters", nargs='+', default=[8,16,32], required=False)
    parser.add_argument("--kernel_sizes", nargs='+', default=[10,5,2], required=False)
    parser.add_argument("--strides", nargs='+', default=[2,2,1], required=False)
    parser.add_argument("--pool_sizes", nargs='+', default=[2,2,0], required=False)
    parser.add_argument("--strides_pooling", nargs='+', default=[2,1,0], required=False)
    
    # FINE TUNING OPTIONS
    parser.add_argument("--add_FT_dense", default=False, type=str2bool, required=False)
    parser.add_argument("--trainable", default=False, type=str2bool, required=False)
    parser.add_argument("--unfreeze", default=False, type=str2bool, required=False)
    
    
    # PARAMETERS FOR TRAINING
    parser.add_argument("--lr", default=0.01, type=float, required=False)
    parser.add_argument("--drop", default=0.5, type=float, required=False)
    parser.add_argument("--n_epochs", default=70, type=int, required=False)
    parser.add_argument("--val_size", default=0.15, type=float, required=False)
    parser.add_argument("--test_size", default=0., type=float, required=False)
    parser.add_argument("--batch_size", default=2500, type=int, required=False)
    parser.add_argument("--patience", default=100, type=int, required=False)
    parser.add_argument("--GPU", default=True, type=str2bool, required=False)
    parser.add_argument("--TPU", default=False, type=str2bool, required=False)
    parser.add_argument("--decay", default=0.95, type=float, required=False)
    parser.add_argument("--BatchNorm", default=True, type=str2bool, required=False)
    parser.add_argument("--padding", default='valid', type=str, required=False)

    FLAGS = parser.parse_args()
    
    FLAGS.z_bins = [int(z) for z in FLAGS.z_bins]
    FLAGS.filters = [int(z) for z in FLAGS.filters]
    FLAGS.kernel_sizes = [int(z) for z in FLAGS.kernel_sizes]
    FLAGS.strides = [int(z) for z in FLAGS.strides]
    FLAGS.pool_sizes = [int(z) for z in FLAGS.pool_sizes]
    FLAGS.strides_pooling = [int(z) for z in FLAGS.strides_pooling]
    FLAGS.c_1.sort()
    FLAGS.c_0.sort()

    if FLAGS.TPU and FLAGS.GPU:
        print('Cannot use both TPU and GPU. Using GPU only ')
        FLAGS.TPU=False
    
    if FLAGS.TPU and FLAGS.save_ckpt:
        print("Cannot save checkpoints in TPU training mode, proceeding without saving checkpoints.")
        FLAGS.save_ckpt=False
    
    if FLAGS.TPU:
        try:
            tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Automatically detects the TPU
            tf.config.experimental_connect_to_cluster(tpu_resolver)  # Connects to the TPU cluster
            tf.tpu.experimental.initialize_tpu_system(tpu_resolver)  # Initializes the TPU system
            strategy = tf.distribute.TPUStrategy(tpu_resolver)
            tpu_device = tpu_resolver.master()  # Retrieves the TPU device URI
        except:
            raise Exception("TPU not found. Check if TPU is enabled in the notebook settings")
    else:
        strategy = None
                  
    #if not FLAGS.fine_tune:
    #    if not FLAGS.dataset_balanced and FLAGS.one_vs_all:
    #        raise ValueError('dataset_balanced must be true in one vs all mode')
        #if not FLAGS.one_vs_all and not FLAGS.dataset_balanced:
        #    raise ValueError('when not in  one vs all mode, dataset_balanced must be true')
    log_fname_add=''
    if FLAGS.fine_tune:
        log_fname_add+='_'
        FLAGS_ORIGINAL = get_flags(FLAGS.log_path)
        if len(FLAGS.c_1)>1:
            add_ckpt_name = ''
            temp_dict={ label:'non_lcdm' for label in FLAGS.c_1}
            if not FLAGS.one_vs_all:
                #raise ValueError('one vs all must be true when fine tuning against one label')
                print('Fine tuning reauires ne vs all to be true. Correcting original flag')
                FLAGS.one_vs_all=True
        else:
            # fine tuning 1vs 1
            temp_dict={ label:label for label in FLAGS.c_1}
            add_ckpt_name = '_'+('-').join(FLAGS.c_1)+'vs'+('-').join(FLAGS.c_0)
            log_fname_add+='_'+('-').join(FLAGS.c_1)+'vs'+('-').join(FLAGS.c_0)
        if not FLAGS.dataset_balanced:
            add_ckpt_name += '_unbalanced'
            log_fname_add+='_unbalanced'
        else:
            add_ckpt_name += '_balanced'
            log_fname_add += '_balanced'
        ft_ckpt_name_base_unfreezing=add_ckpt_name+'_frozen_weights'
        if not FLAGS.trainable:
            add_ckpt_name+='_frozen_weights'
            log_fname_add+='_frozen_weights'
        else:
            add_ckpt_name+='_all_weights'
            log_fname_add+='_all_weights'
        
        if FLAGS.include_last:
            add_ckpt_name+='_include_last'
            log_fname_add+='_include_last'
        else:
            add_ckpt_name+='_without_last'
            log_fname_add+='_without_last'
        
        if FLAGS.unfreeze:
            add_ckpt_name+='_unfrozen'
            log_fname_add+='_unfrozen'
        #FLAGS.group_lab_dict = temp_dict
        if not FLAGS.out_path_overwrite:
            out_path = FLAGS_ORIGINAL.models_dir+FLAGS_ORIGINAL.fname
        else:
            out_path = FLAGS.models_dir+FLAGS.fname
            
    elif FLAGS.one_vs_all:
        if len(FLAGS.c_1)>1:
            add_ckpt_name = ''
            temp_dict={ label:'non_lcdm' for label in FLAGS.c_1}
        else:
            # training  1vs 1
            temp_dict={ label:label for label in FLAGS.c_1}
            add_ckpt_name = '_'+('-').join(FLAGS.c_1)+'vs'+('-').join(FLAGS.c_0)
        out_path = FLAGS.models_dir+FLAGS.fname
    else:
        out_path = FLAGS.models_dir+FLAGS.fname
    
    
    if FLAGS.one_vs_all or FLAGS.fine_tune: 
        FLAGS.group_lab_dict = temp_dict
        for i in range(len(FLAGS.c_0) ):
            FLAGS.group_lab_dict[FLAGS.c_0[i]]=FLAGS.c_0[i]
        
        
    
    if FLAGS.test_mode and not FLAGS.fine_tune:
        out_path=out_path+'_test'
        
    
    ###
    # Uncomment the parts below to redirect output to file. 
    # Does not work on Google Colab
    ###
    
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        tf.io.gfile.makedirs(out_path)
    else:
       print('Directory %s not created' %out_path)
    
    
    logfile = os.path.join(out_path, FLAGS.fname+log_fname_add+'_log.txt')
    myLog = Logger(logfile)
    sys.stdout = myLog
    
    
    #with open(out_path+'/params.txt', 'w') as fpar:    
    #    print('Opened params file %s. Writing params' %(out_path+'/params.txt'))
    print('\n -------- Parameters:')
    for key,value in vars(FLAGS).items():
            print (key,value)
        #    fpar.write(' : '.join([str(key), str(value)])+'\n')
    
    
    print('\n------------ CREATING DATA GENERATORS ------------')
    if FLAGS.TPU:
        training_generator, validation_generator = create_generators(FLAGS, strategy=strategy)
    else:
        training_generator, validation_generator = create_generators(FLAGS)
    
    if FLAGS.fine_tune:
        print('\n------------ CREATING ORIGINAL DATA GENERATORS FOR CHECK------------')
        if FLAGS.TPU:
            or_training_generator, or_validation_generator = create_generators(FLAGS_ORIGINAL, strategy=strategy)
        else:
            or_training_generator, or_validation_generator = create_generators(FLAGS_ORIGINAL)
        n_classes = or_training_generator.n_classes_out # in order to build correctly original model
        model_name = FLAGS_ORIGINAL.model_name
        bayesian=FLAGS_ORIGINAL.bayesian
    else:
        n_classes = training_generator.n_classes_out
        model_name = FLAGS.model_name
        bayesian = FLAGS.bayesian
    
    print('------------ DONE ------------\n')
    
    
    
    print('------------ BUILDING MODEL ------------')
    if FLAGS.swap_axes:
        input_shape = ( int(training_generator.dim[0]), 
                   int(training_generator.n_channels))
    else:
        input_shape = ( int(training_generator.dim[0]), 
                   int(training_generator.dim[1]), 
                   int(training_generator.n_channels))
    print('Input shape %s' %str(input_shape))
    
    if FLAGS.test_mode:
        drop=0
    else:
        drop=FLAGS.drop
    
    
    if FLAGS.fine_tune:
        try:
                BatchNorm=FLAGS_ORIGINAL.BatchNorm
        except AttributeError:
                print(' ####  FLAGS.BatchNorm not found! #### \n Probably loading an older model. Using BatchNorm=True')
                BatchNorm=True

        filters, kernel_sizes, strides, pool_sizes, strides_pooling, n_dense, padding= FLAGS_ORIGINAL.filters, FLAGS_ORIGINAL.kernel_sizes, FLAGS_ORIGINAL.strides, FLAGS_ORIGINAL.pool_sizes, FLAGS_ORIGINAL.strides_pooling, FLAGS_ORIGINAL.n_dense, FLAGS_ORIGINAL.padding
    else:
        try:
            BatchNorm=FLAGS.BatchNorm
        except AttributeError:
            print(' ####  FLAGS.BatchNorm not found! #### \n Probably loading an older model. Using BatchNorm=True')
            BatchNorm=True
        filters, kernel_sizes, strides, pool_sizes, strides_pooling, n_dense, padding = FLAGS.filters, FLAGS.kernel_sizes, FLAGS.strides, FLAGS.pool_sizes, FLAGS.strides_pooling, FLAGS.n_dense, FLAGS.padding
    
    if FLAGS.TPU:
        with strategy.scope():
            model=make_model(     model_name=model_name,
                        drop=drop, 
                        n_labels=n_classes, 
                        input_shape=input_shape, 
                        padding=padding, 
                        filters=filters,
                        kernel_sizes=kernel_sizes,
                        strides=strides,
                        pool_sizes=pool_sizes,
                        strides_pooling=strides_pooling,
                        activation=tf.nn.leaky_relu,
                        bayesian=bayesian, 
                        n_dense=n_dense, swap_axes=FLAGS.swap_axes, BatchNorm=BatchNorm
                            )

            model.build(input_shape=input_shape)
    else:
        model=make_model(     model_name=model_name,
                         drop=drop, 
                          n_labels=n_classes, 
                          input_shape=input_shape, 
                          padding=padding, 
                          filters=filters,
                          kernel_sizes=kernel_sizes,
                          strides=strides,
                          pool_sizes=pool_sizes,
                          strides_pooling=strides_pooling,
                          activation=tf.nn.leaky_relu,
                          bayesian=bayesian, 
                          n_dense=n_dense, swap_axes=FLAGS.swap_axes, BatchNorm=BatchNorm
                             )
        model.build(input_shape=input_shape)

    print(model.summary())
    
    if FLAGS.fine_tune:
        if FLAGS.TPU:
            with strategy.scope():
                loss_0 = compute_loss(or_training_generator, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        else:
            loss_0 = compute_loss(or_training_generator, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        print('Loss before loading weights/ %s\n' %loss_0.numpy())
    
    if FLAGS.decay is not None:
        if FLAGS.TPU:
            with strategy.scope():
                lr_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.lr, len(training_generator), FLAGS.decay)
                optimizer = tf.keras.optimizers.Adam(lr_fn)
        else:
            lr_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.lr, len(training_generator), FLAGS.decay)
            optimizer = tf.keras.optimizers.Adam(lr_fn)
    else:
        if FLAGS.TPU:
            with strategy.scope():
                optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    
    if FLAGS.restore and FLAGS.decay is not None:
        decayed_lr_value = lambda step: FLAGS.lr * FLAGS.decay**(step / len(training_generator))
        
    #optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    #if FLAGS.decay is not None:
    #    optimizer.decay = tf.Variable(tf.Variable(FLAGS.decay))
    
    
    if not FLAGS.unfreeze:
        ckpts_path = out_path+'/tf_ckpts/'
    else:
        ckpts_path=out_path+'/tf_ckpts_fine_tuning'+ft_ckpt_name_base_unfreezing+'/'
    ckpt_name = 'ckpt'
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    
    if FLAGS.fine_tune:
        print('Loading ckpt from %s' %ckpts_path)
        latest = tf.train.latest_checkpoint(ckpts_path)
        print('Loading ckpt %s' %latest)
        if not FLAGS.test_mode:
            ckpts_path = out_path+'/tf_ckpts_fine_tuning'+add_ckpt_name+'/'
        else:
            ckpts_path = out_path+'/tf_ckpts_fine_tuning_test'+add_ckpt_name+'/'
        ckpt_name = ckpt_name+'_fine_tuning'+add_ckpt_name
        if FLAGS.test_mode:
            ckpt_name+='_test'
        ckpt.restore(latest)
        print('Last learning rate was %s' %ckpt.optimizer.learning_rate)
        ckpt.optimizer.learning_rate = FLAGS.lr
        print('Learning rate set to %s' %ckpt.optimizer.learning_rate)
        
        if FLAGS.TPU:
            with strategy.scope():
                loss_1 = compute_loss(or_training_generator, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        else:
            loss_1 = compute_loss(or_training_generator, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)

        print('Loss after loading weights/ %s\n' %loss_1.numpy())
        if FLAGS.add_FT_dense:
            if not FLAGS.swap_axes:
                dense_dim=filters[-1]
            else:
                dense_dim=filters[-1]
        else:
            dense_dim=0
        
        if not FLAGS.unfreeze:
            if FLAGS.TPU:
                with strategy.scope():
                    model = make_fine_tuning_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_generator.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       trainable=FLAGS.trainable, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm, include_last=FLAGS.include_last)
            else:
                model = make_fine_tuning_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_generator.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       trainable=FLAGS.trainable, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm, include_last=FLAGS.include_last)
        else:
            if FLAGS.TPU:
                with strategy.scope():
                    model = make_unfreeze_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_generator.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm)
            else: 
                model = make_unfreeze_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_generator.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm)
        model.build(input_shape=input_shape)
        print(model.summary())

        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    elif FLAGS.one_vs_all:
        if not FLAGS.test_mode:
            ckpts_path = out_path+'/tf_ckpts'+add_ckpt_name+'/'
        else:
            ckpts_path = out_path+'/tf_ckpts_test'+add_ckpt_name+'/'
        ckpt_name = ckpt_name+add_ckpt_name
        if FLAGS.test_mode:
            ckpt_name+='_test'
        
    with tf.device('/CPU:0'):
        manager = tf.train.CheckpointManager(ckpt, ckpts_path, 
                                            max_to_keep=2, 
                                            checkpoint_name=ckpt_name)
    
    if FLAGS.TPU:
        with strategy.scope():
            train_acc_metric = tf.keras.metrics.Accuracy()
            val_acc_metric = tf.keras.metrics.Accuracy()
            train_loss_metric = tf.keras.metrics.Mean()
            val_loss_metric = tf.keras.metrics.Mean()
    else:
        train_acc_metric = tf.keras.metrics.Accuracy()
        val_acc_metric = tf.keras.metrics.Accuracy()
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()
    
    
    if FLAGS.GPU:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            #raise SystemError('GPU device not found')
            print('GPU device not found ! Device: %s' %device_name)
        else: print('Found GPU at: {}'.format(device_name))
    
    
    print('------------ TRAINING ------------\n')
    if FLAGS.bayesian:
        loss=ELBO
    else:
        loss=my_loss
    
    
    #print('Model n_classes : %s ' %n_classes)
    print('Features shape:', training_generator.xshape)
    print('Labels shape:', training_generator.yshape)
   
    model, history = my_train(model, optimizer, loss,
                FLAGS.n_epochs, 
                training_generator, 
                validation_generator, manager, ckpt, train_loss_metric,
                train_acc_metric, val_loss_metric, val_acc_metric, TPU=FLAGS.TPU,
                strategy=strategy, patience=FLAGS.patience, restore=FLAGS.restore, 
                bayesian=bayesian, save_ckpt=FLAGS.save_ckpt, decayed_lr_value=None, save_indexes = FLAGS.save_indexes #not(FLAGS.test_mode)
            )
    hist_path =  out_path+'/hist.png'
    if FLAGS.fine_tune:
        if FLAGS.test_mode:
            hist_path = out_path +'/hist_fine_tuning'+add_ckpt_name+'_test.png'
        else:
            hist_path = out_path +'/hist_fine_tuning'+add_ckpt_name+'.png'  
    
    plot_hist(DummyHist(history), epochs=len(history['loss']), save=True, path=hist_path, show=False)
    
    
    print('Done in %.2fs' %(time.time() - in_time))
    ###
    # Uncoment if saving output on file, to properly close
    ###    
    sys.stdout = sys.__stdout__
    myLog.close()


     
        
if __name__=='__main__':
    
    main()
    
