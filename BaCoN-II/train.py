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
from data_generator import create_datasets
from models import *
from utils import DummyHist, plot_hist, str2bool, Logger, get_flags
import sys
import time
import shutil

class BayesianLoss(tf.keras.losses.Loss):
    def __init__(self, n_train_examples, n_val_examples):
        super().__init__()
        self.n_examples = n_train_examples  # Default to training examples
        self.n_train_examples = n_train_examples
        self.n_val_examples = n_val_examples
        self.base_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, y_true, y_pred):
        kl_loss = sum(self.model.losses) / self.n_examples
        neg_log_likelihood = self.base_loss(y_true, y_pred)
        return neg_log_likelihood + kl_loss

    def set_mode_training(self):
        self.n_examples = self.n_train_examples

    def set_mode_validation(self):
        self.n_examples = self.n_val_examples

    def set_model(self, model):
        self.model = model

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss, ckpt, checkpoint_manager, fname_hist, patience=10, restore=False, history=None, save_ckpt=False):
        self.ckpt = ckpt
        self.checkpoint_manager = checkpoint_manager
        self.fname_hist = fname_hist
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.save_ckpt = save_ckpt
        self.restore = restore
        self.history = history or {}
        self.loss_function = loss
        # Ensure the checkpoint directory exists
        tf.io.gfile.makedirs(checkpoint_manager.directory)
        self.start_time = time.time()  # Overall training start time
    
    def on_train_batch_begin(self, batch, logs=None):
        self.loss_function.set_mode_training()
    
    def on_test_batch_begin(self, batch, logs=None):
        self.loss_function.set_mode_validation()
    
    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        if epoch == 0 and self.restore:
            # Rewriting historical data if restoring
            for key in self.history.keys():
                fname = os.path.join(self.fname_hist, f'{key}.txt')
                with open(fname, 'a') as fh:
                    for el in self.history[key][:-1]:
                        fh.write(str(el) + '\n')
            print(f'Re-wrote histories until epoch {len(self.history["val_accuracy"][:-1])}')

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        current_val_loss = logs.get('val_loss')
        improvement_flag = False

        if current_val_loss < self.best_loss:
            self.best_loss = current_val_loss
            save_path = self.checkpoint_manager.save()
            improvement_flag = True
            self.count = 0  # Reset counter after improvement
            if self.save_ckpt:
                print(f"\nEpoch {epoch+1}: Validation loss decreased to {current_val_loss:.4f}. Checkpoint saved: {save_path}")
        else:
            self.count += 1
            print(f'\nEpoch {epoch+1}: Loss did not decrease. Count = {self.count}')

        # Early stopping check
        if self.count >= self.patience:
            print('Max patience reached. Stopping training.')
            self.model.stop_training = True

        # Increment the checkpoint step
        self.ckpt.step.assign_add(1)

        # Log metrics and epoch duration
        for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:  # Adjust according to your metrics
            if key in logs:
                fname = os.path.join(self.fname_hist, f'{key}.txt')
                with open(fname, 'a') as fh:
                    fh.write(f'{logs[key]}\n')

        # Print epoch summary
        total_time = time.time() - self.start_time
        print(f"Time:  {total_time:.2f}, ---- Loss: {logs.get('loss', 0):.4f}, Acc.: {logs.get('accuracy', 0):.4f}, Val. Loss: {logs.get('val_loss', 0):.4f}, Val. Acc.: {logs.get('val_accuracy', 0):.4f}\n")

def my_train(model, loss, epochs,
             train_dataset, 
             val_dataset, ckpt_path, ckpt, ckpt_manager, TPU=False, strategy=None,
             restore=False, patience=100, shuffle=True, save_ckpt=False):
    fname_hist = os.path.join(ckpt_path, 'hist')
    if not os.path.exists(fname_hist):
        os.makedirs(fname_hist)
    if restore:
        print('Restoring ckpt...')
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored checkpoint from {}".format(ckpt_manager.latest_checkpoint))
            print('ckpt step: %s' % ckpt.step.numpy())

            hist_start = int(ckpt.step)
            print('Starting from history at step %s' % hist_start)

            # Load history from files
            history = {
                key: np.loadtxt(fname_hist+'_'+key+'.txt').tolist()[0:hist_start] 
                for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']
            }

            # Rename original history files to keep a backup
            for key in history.keys():
                fname = fname_hist+'_'+key+'.txt'
                fname_new = fname_hist+'_'+key+'_original.txt'
                os.rename(fname, fname_new)
            print('Saved copy of original histories.')
            
            best_train_loss = history['loss'][-1]
            best_loss = history['val_loss'][-1]
            print(f"Starting from (loss, val_loss) = {best_train_loss:.4f}, {best_loss:.4f}")

            # Print last learning rate
            print('Last learning rate was %s' % ckpt.optimizer.learning_rate.numpy())
        else:
            print("Checkpoint not found. Initializing checkpoint from scratch.")
    else:
        print("Initializing checkpoint from scratch.")
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy':[] }
        best_loss = np.infty
    
    callback = TrainingCallback(loss, ckpt, ckpt_manager, fname_hist=fname_hist, patience=10, save_ckpt=save_ckpt)
    if TPU:
        print('val_dataset.n_examples: %s' %val_dataset.n_examples)
        print('train_dataset.n_examples: %s' %train_dataset.n_examples)
        print('val_dataset.batch_size: %s' %val_dataset.batch_size)
        print('train_dataset.batch_size: %s' %train_dataset.batch_size)
        print('num_replicas_in_sync: %s' %strategy.num_replicas_in_sync)
        with strategy.scope():
            val_steps_per_epoch = val_dataset.n_examples // (val_dataset.batch_size * strategy.num_replicas_in_sync)
            train_steps_per_epoch = train_dataset.n_examples // (train_dataset.batch_size * strategy.num_replicas_in_sync)
            train_dataset.dataset = strategy.experimental_distribute_dataset(train_dataset.dataset)
            val_dataset.dataset = strategy.experimental_distribute_dataset(val_dataset.dataset)
            history = model.fit(train_dataset.dataset, epochs=epochs,
                                validation_data=val_dataset.dataset,
                                callbacks=[callback], steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps_per_epoch)
    else:
        history = model.fit(train_dataset.dataset, epochs=epochs,
                            validation_data=val_dataset.dataset,
                            callbacks=[callback])

    return model, history


def compute_loss(dataset, model, bayesian=False, TPU=False, strategy=None):
    x_batch_train, y_batch_train = dataset.take(0)
    logits = model(x_batch_train, training=False)
    if bayesian:
            kl = sum(model.losses)/dataset.batch_size/dataset.n_batches
            base_loss = tf.keras.losses.CategoricalCrossentropy(y_batch_train, logits, from_logits=True)
            loss_0 = base_loss + kl
    else:
            loss_0 = tf.keras.losses.categorical_crossentropy(y_batch_train, logits, from_logits=True)
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
    parser.add_argument("--shuffle", default='False', type=str2bool, required=False)

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
    
    
    print('\n------------ CREATING DATASETS ------------')
    if FLAGS.TPU:
        training_dataset, validation_dataset = create_datasets(FLAGS, strategy=strategy)
    else:
        training_dataset, validation_dataset = create_datasets(FLAGS)

    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if FLAGS.fine_tune:
        print('\n------------ CREATING ORIGINAL DATASETS FOR CHECK------------')
        if FLAGS.TPU:
            or_training_dataset, or_validation_dataset = create_datasets(FLAGS_ORIGINAL, strategy=strategy)
        else:
            or_training_dataset, or_validation_dataset = create_datasets(FLAGS_ORIGINAL)
        n_classes = or_training_dataset.n_classes_out # in order to build correctly original model
        model_name = FLAGS_ORIGINAL.model_name
        bayesian=FLAGS_ORIGINAL.bayesian
    else:
        n_classes = training_dataset.n_classes_out
        model_name = FLAGS.model_name
        bayesian = FLAGS.bayesian
    
    print('------------ DONE ------------\n')
    
    
    
    print('------------ BUILDING MODEL ------------')
    if FLAGS.swap_axes:
        input_shape = ( int(training_dataset.dim[0]), 
                   int(training_dataset.n_channels))
    else:
        input_shape = ( int(training_dataset.dim[0]), 
                   int(training_dataset.dim[1]), 
                   int(training_dataset.n_channels))
    print('Input shape %s' %str(input_shape))
    
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

    if FLAGS.decay is not None:
        if FLAGS.TPU:
            with strategy.scope():
                lr_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.lr, len(training_dataset), FLAGS.decay)
                optimizer = tf.keras.optimizers.Adam(lr_fn)
        else:
            lr_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.lr, len(training_dataset), FLAGS.decay)
            optimizer = tf.keras.optimizers.Adam(lr_fn)
    else:
        if FLAGS.TPU:
            with strategy.scope():
                optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    
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
            if FLAGS.bayesian:
                loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
                loss.set_model(model)
            else:
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
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
        if FLAGS.bayesian:
            loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
            loss.set_model(model)
        else:
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)

    print(model.summary())
    
    if FLAGS.fine_tune:
        if FLAGS.TPU:
            with strategy.scope():
                loss_0 = compute_loss(or_training_dataset, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        else:
            loss_0 = compute_loss(or_training_dataset, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        print('Loss before loading weights/ %s\n' %loss_0.numpy())

    
    if FLAGS.restore and FLAGS.decay is not None:
        decayed_lr_value = lambda step: FLAGS.lr * FLAGS.decay**(step / len(training_dataset))
        
    #optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    #if FLAGS.decay is not None:
    #    optimizer.decay = tf.Variable(tf.Variable(FLAGS.decay))
    
    
    if not FLAGS.unfreeze:
        ckpts_path = out_path+'/tf_ckpts/'
    else:
        ckpts_path=out_path+'/tf_ckpts_fine_tuning'+ft_ckpt_name_base_unfreezing+'/'
    ckpt_name = 'ckpt'
    
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
                loss_1 = compute_loss(or_training_dataset, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)
        else:
            loss_1 = compute_loss(or_training_dataset, model, bayesian=FLAGS.bayesian, TPU=FLAGS.TPU, strategy=strategy)

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
                                       n_out_labels=training_dataset.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       trainable=FLAGS.trainable, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm, include_last=FLAGS.include_last)
                    if FLAGS.bayesian:
                        loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
                        loss.set_model(model)
                    else:
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            else:
                model = make_fine_tuning_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_dataset.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       trainable=FLAGS.trainable, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm, include_last=FLAGS.include_last)
                if FLAGS.bayesian:
                    loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
                    loss.set_model(model)
                else:
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)
        else:
            if FLAGS.TPU:
                with strategy.scope():
                    model = make_unfreeze_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_dataset.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm)
                    if FLAGS.bayesian:
                        loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
                        loss.set_model(model)
                    else:
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            else: 
                model = make_unfreeze_model(base_model=model, input_shape=input_shape, 
                                       n_out_labels=training_dataset.n_classes_out,
                                       dense_dim= dense_dim, bayesian=bayesian, 
                                       drop=drop,  BatchNorm=FLAGS.BatchNorm)
                if FLAGS.bayesian:
                    loss=BayesianLoss(n_train_examples=training_dataset.n_batches*training_dataset.batch_size, n_val_examples=validation_dataset.n_batches*validation_dataset.batch_size)
                    loss.set_model(model)
                else:
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)
        print(model.summary())
    elif FLAGS.one_vs_all:
        if not FLAGS.test_mode:
            ckpts_path = out_path+'/tf_ckpts'+add_ckpt_name+'/'
        else:
            ckpts_path = out_path+'/tf_ckpts_test'+add_ckpt_name+'/'
        ckpt_name = ckpt_name+add_ckpt_name
        if FLAGS.test_mode:
            ckpt_name+='_test' 
        
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpts_path, 
                                         max_to_keep=2, 
                                         checkpoint_name=ckpt_name)
    
    if FLAGS.GPU:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            #raise SystemError('GPU device not found')
            print('GPU device not found ! Device: %s' %device_name)
        else: print('Found GPU at: {}'.format(device_name))
    
    
    print('------------ TRAINING ------------\n')

    #print('Model n_classes : %s ' %n_classes)
    print('Features shape:', training_dataset.xshape)
    print('Labels shape:', training_dataset.yshape)
   
    model, history = my_train(model, loss,
                FLAGS.n_epochs, 
                training_dataset, 
                validation_dataset, ckpts_path, ckpt, manager, TPU=FLAGS.TPU,
                strategy=strategy, patience=FLAGS.patience, restore=FLAGS.restore, shuffle=FLAGS.shuffle, 
                save_ckpt=FLAGS.save_ckpt #not(FLAGS.test_mode)
            )
    
    #Delete cached datasets
    if not FLAGS.TPU:
        try:
            shutil.rmtree('cache')
            print(f"Cache folder deleted.")
        except FileNotFoundError:
            print(f"Cache folder does not exist.")
        except Exception as e:
            print(f"Error deleting cache folder.")
    
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
    
