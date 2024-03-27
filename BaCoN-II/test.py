#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:23:28 2020

@author: Michi
@editor: Linus, Jan 2023
"""

import argparse
import os
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tf.enable_v2_behavior()
tfd = tfp.distributions
from data_generator import create_test_dataset, create_datasets
from models import *
from utils import DummyHist, plot_hist, str2bool, Logger, get_flags
import sys
import time


def load_model_for_test(FLAGS, input_shape, n_classes=5,
                        dataset=None, FLAGS_ORIGINAL=None, new_fname=None, TPU=False, model_weights_path=None):
         

    
    print('------------ BUILDING MODEL ------------\n')
    
    print('Model n_classes : %s ' %n_classes)
    if dataset is not None:
        for batch in dataset.dataset.take(1):
            x, y = batch
            print('Features shape: %s' %str(x[0].shape))
            print('Labels shape: %s' %str(y[0].shape))
    
    try:
        BatchNorm=FLAGS.BatchNorm
    except AttributeError:
        print(' ####  FLAGS.BatchNorm not found! #### \n Probably loading an older model. Using BatchNorm=True')
        BatchNorm=True
    
    model=make_model(  model_name=FLAGS.model_name,
                         drop=0, 
                          n_labels=n_classes, 
                          input_shape=input_shape, 
                          padding='valid', 
                          filters=FLAGS.filters,
                          kernel_sizes=FLAGS.kernel_sizes,
                          strides=FLAGS.strides,
                          pool_sizes=FLAGS.pool_sizes,
                          strides_pooling=FLAGS.strides_pooling,
                          activation=tf.nn.leaky_relu,
                          bayesian=FLAGS.bayesian, 
                          n_dense=FLAGS.n_dense, swap_axes=FLAGS.swap_axes, BatchNorm=BatchNorm
                          )            
            

    
    model.build(input_shape=input_shape)
    #print(model.summary())
    
    if FLAGS.fine_tune:
        
        if FLAGS.add_FT_dense:
            dense_dim=FLAGS.filters[-1]
        else:
            dense_dim=0

        if len(FLAGS.c_1)==1:
            ft_ckpt_name = '_'+('-').join(FLAGS.c_1)+'vs'+('-').join(FLAGS.c_0)
        else:
            ft_ckpt_name=''
        if not FLAGS.dataset_balanced:
            ft_ckpt_name += '_unbalanced'
        else:
            ft_ckpt_name += '_balanced'
        if not FLAGS.trainable and FLAGS.fine_tune:
            ft_ckpt_name+='_frozen_weights'
        else:
            ft_ckpt_name+='_all_weights'
        if FLAGS.include_last:
            ft_ckpt_name+='_include_last'
        else:
            ft_ckpt_name+='_without_last'
        if FLAGS.unfreeze:
            ft_ckpt_name+='_unfrozen'
        
            
        model = make_fine_tuning_model(base_model=model, input_shape=input_shape, n_out_labels=dataset.n_classes_out,
                                       dense_dim= dense_dim, bayesian=FLAGS.bayesian, trainable=False, drop=0, BatchNorm=FLAGS.BatchNorm,
                                       include_last=FLAGS.include_last)
        model.build(input_shape=input_shape)
    print(model.summary())
    
    if dataset is not None:
        print('Computing loss for randomly initialized model...')
        loss_0 = compute_loss(dataset, model, bayesian=FLAGS.bayesian)
        print('Loss before loading weights/ %s\n' %loss_0.numpy())
            
    if TPU:
        print('------------ LOADING MODEL WEIGHTS ------------\n')
        model.load_weights(model_weights_path)
    else:    
        print('------------ RESTORING CHECKPOINT ------------\n')
        if new_fname is None:
            out_path = FLAGS.models_dir+FLAGS.fname
        else:
            out_path = FLAGS.models_dir+new_fname
        optimizer = tf.keras.optimizers.legacy.Adam(FLAGS.lr)
        ckpts_path = out_path
        ckpts_path+='/tf_ckpts'
        if FLAGS.fine_tune:
            ckpts_path += '_fine_tuning'+ft_ckpt_name
        ckpts_path+='/'
        print('Looking for ckpt in ' + ckpts_path)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
        
        
        latest = tf.train.latest_checkpoint(ckpts_path)
        if latest:
            print("Restoring checkpoint from {}".format(latest))
        else:
            raise ValueError('Checkpoint not found')
            #print('Checkpoint not found')
        ckpt.restore(latest)
        
    if dataset is not None:
        loss_1 = compute_loss(dataset, model, bayesian=FLAGS.bayesian)
        print('Loss after loading weights/ %s\n' %loss_1.numpy())   
        
    return model


def compute_loss(dataset, model, bayesian=False):
    for x_batch_train, y_batch_train in dataset.dataset.take(1):
        logits = model(x_batch_train, training=False)
    if bayesian:
        kl = sum(model.losses)/dataset.n_batches/dataset.batch_size
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        base_loss = loss_fn(y_batch_train, logits)
        loss_0 = base_loss + kl
    else:
        loss_0 = tf.keras.losses.categorical_crossentropy(y_batch_train, logits, from_logits=True)
    return loss_0


def print_cm(cm, names, out_path, cm_name_custom,tot_acc,tot_acc_no_uncl, FLAGS):
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc('text', usetex=True)
    import seaborn as sns
    
    for i, name in enumerate(names):
        if name == 'dgp':
            names[i] = 'DGP'
        if name == 'ds':
            names[i] = 'DS'
        if name == 'fr':
            names[i] = '$f(R)$'
        if name == 'lcdm':
            names[i] = '$\Lambda$CDM'
        if name == 'rand':
            names[i] = 'Random'
        if name == 'wcdm':
            names[i] = '$w$CDM'

    matrix_proportions = np.zeros((len(names),len(names)))
    for i in range(0,len(names)):
        np.seterr(invalid='ignore') # Suppress/hide the warning when divide by zero
        matrix_proportions[i,:] = cm[i,:]/float(cm[i,:].sum())
    #print(matrix_proportions)    
    
    confusion_df = pd.DataFrame(matrix_proportions, 
                            index=names,columns=names)
    plt.figure(figsize=(8,8))
    sns.heatmap(confusion_df, annot=True,
            annot_kws={"size": 18}, cmap='gist_gray_r',
            cbar=False, square=True, fmt='.2f');
                
    #accuracy = np.trace(cm) / np.sum(cm).astype('float')
    #misclass = 1 - accuracy
    #score_str='Accuracy: %s, Misclassified: %s'%(accuracy, misclass)
    plt.ylabel(r'True categories',fontsize=14);
    plt.xlabel(r'Predicted categories',fontsize=14);
    plt.tick_params(labelsize=16);
    plt.title("Test accuracy w/ N.C.: {:.3f}".format(tot_acc.numpy()),fontsize=20)
    
    #plt.show()
    cm_path = out_path+'/cm_'+cm_name_custom
    if FLAGS.fine_tune:
        cm_path+='_FT'
    if not FLAGS.trainable:
        cm_path+='_frozen_weights'
    
    cm_path_vals = cm_path+'_values.txt'
    cm_path+='.pdf'
    plt.savefig(cm_path)
    np.savetxt(cm_path_vals, cm)
    print('Saved confusion matrix at %s' %cm_path)
    print('Saved confusion matrix values at %s' %cm_path_vals)
    
    return matrix_proportions    


def evaluate_accuracy(model, test_dataset, out_path, cm_name_custom, names=None, FLAGS=None):
    acc_total=0
    y_true_tot=[]
    y_pred_tot=[]
    for batch_idx, batch in enumerate(test_dataset.dataset):
        X, y = batch
        pred = model.predict(X, verbose=0)
        y_pred = tf.argmax(tf.nn.softmax(pred, axis=1), axis=1)
        y_true = tf.argmax(y, axis=1)
        equality_batch = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(equality_batch, tf.float32))
        print('Accuracy on %s batch: %s %%' %(batch_idx, accuracy.numpy()))
        acc_total += accuracy
        y_true_tot.append(y_true)
        y_pred_tot.append(y_pred)

    tot_acc = acc_total/test_dataset.n_batches
    print('-- Total accuracy: %s %%' %( tot_acc.numpy()))  
    tot_acc_no_uncl = 0
    
    #### CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tf.concat(y_true_tot,axis=0),tf.concat(y_pred_tot,axis=0))
    _ = print_cm(cm, names, out_path, cm_name_custom, tot_acc,tot_acc_no_uncl,FLAGS)
    
    
    return tot_acc


def predict_bayes_label(mean_prob, th_prob=0.5):
  if type(mean_prob) is not np.ndarray:
      mean_prob=mean_prob.numpy()
  if mean_prob[mean_prob>th_prob].sum()==0.:
      pred_label = 99
  else:
      pred_label = tf.argmax(mean_prob).numpy()
  return pred_label


def predict_mean_proba(X, model, num_monte_carlo=100, softmax=True, verbose=False):
        sampled_logits = tf.stack([model.predict(X, verbose=0)
                          for _ in range(num_monte_carlo)], axis=0)
        if softmax:
            sampled_probas = tf.nn.softmax(sampled_logits, axis=-1)
        else:
            sampled_probas= sampled_logits
        if verbose:
            print("sampled_probas shape: %s" %str(sampled_probas.shape))
        mean_proba=tf.reduce_mean(sampled_probas, axis=0)
        if verbose:
            print("mean_proba  shape: %s" %str(mean_proba.shape))
        return mean_proba, sampled_probas


def my_predict(X, model, num_monte_carlo=100, th_prob=0.5, verbose=False):
  if verbose:
      print('using th_prob=%s'%th_prob)
  mean_proba, sampled_probas = predict_mean_proba(X, model, num_monte_carlo=num_monte_carlo, verbose=verbose)
  mean_pred = tf.map_fn(fn=lambda x: predict_bayes_label(x, th_prob=th_prob), elems=mean_proba)
  if verbose:
      print("mean_pred  shape: %s" %str(mean_pred.shape))
  return sampled_probas, mean_proba, mean_pred 


def evaluate_accuracy_bayes(model, test_dataset, out_path, cm_name_custom, num_monte_carlo=50, th_prob=0.5, names=None, FLAGS=None):
    acc_total=0
    acc_total_no_uncl=0
    y_true_tot=[]
    y_pred_tot=[]
    all_sampled_probas = []

    print('Threshold probability for classification: %s ' %th_prob)
    for batch_idx, batch in enumerate(test_dataset.dataset):
        X, y = batch
        y_true = tf.argmax(y, axis=1)
        
        # Predict mean probability in each class by averaging on MC samples, then  label with prob threshold
        sampled_probas, mean_proba, mean_pred = my_predict(X, model, num_monte_carlo, th_prob)
        
        # Compute accuracy
        equality_batch = tf.equal(tf.cast(mean_pred, dtype=tf.int64), y_true)
        accuracy = tf.reduce_mean(tf.cast(equality_batch, tf.float32))  
        
        equality_batch_no_uncl = tf.equal(tf.cast(mean_pred[mean_pred!=99], dtype=tf.int64), y_true[mean_pred!=99])
        accuracy_no_uncl = tf.reduce_mean(tf.cast(equality_batch_no_uncl, tf.float32))  
        
        print('Accuracy on %s batch using median of sampled probabilities: %s %%' %(batch_idx, accuracy.numpy()))
        print('Accuracy on %s batch using median of sampled probabilities, not considering unclassified examples: %s %%' %(batch_idx, accuracy_no_uncl.numpy()))
        acc_total += accuracy
        acc_total_no_uncl+=accuracy_no_uncl
        y_true_tot.append(y_true)
        y_pred_tot.append(mean_pred)
        all_sampled_probas.append(sampled_probas)
    tot_acc = acc_total/test_dataset.n_batches
    tot_acc_no_uncl = acc_total_no_uncl/test_dataset.n_batches
    print('-- Accuracy on test set using median of sampled probabilities: %s %% \n' %( tot_acc.numpy()))
    print('-- Accuracy on test set using median of sampled probabilities, not considering unclassified examples: %s %% \n' %( tot_acc_no_uncl.numpy()))  
    
    #### CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tf.concat(y_true_tot,axis=0),tf.concat(y_pred_tot,axis=0))
    
    if tf.unique(tf.concat(y_pred_tot,axis=0)).y.shape[0]>len(names):
        print('Adding Not classified label')
        names = names+['N. C.']
        
    _ = print_cm(cm, names, out_path, cm_name_custom, tot_acc, tot_acc_no_uncl,FLAGS)
    
    
    return tot_acc





   
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bayesian", default=None, type=str2bool, required=False)
    parser.add_argument("--log_path", default='', type=str, required=True)
    parser.add_argument("--TEST_DIR", default=None, type=str, required=False)
    parser.add_argument("--models_dir", default=None, type=str, required=False)
    parser.add_argument("--curves_folder", default=None, type=str, required=False)
    
    parser.add_argument("--n_monte_carlo_samples", default=100, type=int, required=False)
    parser.add_argument("--th_prob", default=0.5, type=float, required=False)
    
    parser.add_argument("--batch_size", default=None, type=int, required=False)
    
    parser.add_argument("--add_noise", default=None, type=str2bool, required=False)
    parser.add_argument("--n_noisy_samples", default=None, type=int, required=False)
    parser.add_argument("--add_shot", default=None, type=str2bool, required=False)
    parser.add_argument("--add_sys", default=None, type=str2bool, required=False)
    parser.add_argument("--add_cosvar", default=None, type=str2bool, required=False)
    parser.add_argument("--sigma_sys", default=None, type=float, required=False)
    parser.add_argument("--sys_scaled", default=None, type=str2bool, required=False)
    parser.add_argument("--sys_factor", default=None, type=float, required=False) 
    parser.add_argument("--sigma_curves", default=None, type=float, required=False)
    parser.add_argument("--sigma_curves_default", default=None, type=float, required=False)
    parser.add_argument("--save_processed_spectra", default=False, type=str2bool, required=False)   
    parser.add_argument("--rescale_curves", default=None, type=str, required=False)
    
    #parser.add_argument('--z_bins', nargs='+', default=[0,1,2,3], required=False)
    parser.add_argument("--save_indexes", default=False, type=str2bool, required=False)
    parser.add_argument("--cm_name_custom", default='', type=str, required=False)
    parser.add_argument("--norm_data_name", default=None, type=str, required=False)
    #parser.add_argument("--sample_pace", default=4, type=int, required=False)


    
    args = parser.parse_args()
    
    print('Reading log from %s ' %args.log_path)
    n_flag_ow=False
    FLAGS = get_flags(args.log_path)

    print('TEST_NORM_FILE: FLAGS.norm_data_name from log_path ', FLAGS.norm_data_name)

    # new to get user name for confuion matrix
    cm_name_custom = args.cm_name_custom
    print('Saving cm at %s' %cm_name_custom)
     
    if args.save_indexes is not None:
        print('Setting save_indexes to %s' %args.save_indexes)
        FLAGS.save_indexes = args.save_indexes
    if args.TEST_DIR is not None:
        print('Using data in the directory %s' %args.TEST_DIR)
        FLAGS.TEST_DIR = args.TEST_DIR
    if args.models_dir is not None:
        print('Reading model from the directory %s' %args.models_dir)
        FLAGS.models_dir = args.models_dir
    if args.curves_folder is not None:
        print('Reading curves from the directory %s' %args.curves_folder)
    if args.batch_size is not None:
        print('Using batch_size %s' %args.batch_size)
        FLAGS.batch_size = args.batch_size
    if args.add_noise is not None:
        n_flag_ow=True
        FLAGS.add_noise = args.add_noise
    if args.n_noisy_samples is not None:
        n_flag_ow=True
        FLAGS.n_noisy_samples=args.n_noisy_samples
    if args.add_shot is not None:
            n_flag_ow=True
            FLAGS.add_shot=args.add_shot
    if args.add_sys is not None:
            n_flag_ow=True
            FLAGS.add_sys=args.add_sys
    if args.add_cosvar is not None:
            n_flag_ow=True
            FLAGS.add_cosvar=args.add_cosvar
    if args.sigma_sys is not None:
            n_flag_ow=True
            FLAGS.sigma_sys=args.sigma_sys
    #if args.sample_pace is not None:
    #        n_flag_ow=True
    #        FLAGS.sample_pace=args.sample_pace
    if args.norm_data_name is not None:
            n_flag_ow=True
            FLAGS.norm_data_name=args.norm_data_name
    if args.sigma_curves is not None:
        n_flag_ow=True
        FLAGS.sigma_curves = args.sigma_curves
    if args.sigma_curves_default is not None:
        n_flag_ow=True
        FLAGS.sigma_curves_default = args.sigma_curves_default
    if args.rescale_curves is not None:
        FLAGS.rescale_curves = args.rescale_curves
        
    
    print('TEST_NORM_FILE: FLAGS.norm_data_name after args written to FLAGS', FLAGS.norm_data_name)

    
    out_path = os.path.join(FLAGS.models_dir,FLAGS.fname)
    print('-------------------- out_path with FLAGS_fname', out_path)

    # LT: added to save prints as log-file (use same method as in train.py)
    logfile = os.path.join(out_path, cm_name_custom+'_log.txt')
    myLog = Logger(logfile)
    sys.stdout = myLog

    if n_flag_ow:
        print('Overwriting noise flags. Using n_noisy_samples=%s, add_shot=%s, add_sys=%s,add_cosvar=%s, sigma_curves=%s' %(FLAGS.n_noisy_samples, str(FLAGS.add_shot),str(FLAGS.add_sys),str(FLAGS.add_cosvar), FLAGS.sigma_curves))
        
        
    print('\n -------- Parameters:')
    for key,value in vars(FLAGS).items():
            print (key,value)
    
    
    print('------------ CREATING DATASETS ------------\n')
    test_dataset = create_test_dataset(FLAGS)
    for _ in test_dataset.dataset:
        pass
        
    print('------------ DONE ------------\n')
    
    TPU = FLAGS.TPU
    model_weights_path = FLAGS.model_weights_path
    
    if FLAGS.swap_axes:
        input_shape = ( int(test_dataset.dim[0]), 
                   int(test_dataset.n_channels))
    else:
        input_shape = ( int(test_dataset.dim[0]), 
                   int(test_dataset.dim[1]), 
                   int(test_dataset.n_channels))
    print('Input shape %s' %str(input_shape))
    
             
    model_loaded =  load_model_for_test(FLAGS, input_shape, n_classes=test_dataset.n_classes_out,
                                        dataset=test_dataset, TPU=TPU, model_weights_path=model_weights_path)
    
    
    names=[ test_dataset.inv_labels_dict[i] for i in range(len(test_dataset.inv_labels_dict.keys()))]
    
    if FLAGS.bayesian:
        _ = evaluate_accuracy_bayes(model_loaded, test_dataset, out_path,cm_name_custom,  
                                    num_monte_carlo = args.n_monte_carlo_samples, th_prob=args.th_prob, 
                                    names=names, FLAGS=FLAGS)
    else:
         _ = evaluate_accuracy(model_loaded, test_dataset, out_path,cm_name_custom,  names=names, FLAGS=FLAGS)
    
    # LT: close log file
    #sys.stdout = sys.__stdout__
    myLog.flush() 
    myLog.close()  


if __name__=='__main__':
    
    main()
