#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:07:43 2020

@author: Michi
@edit: Linus 15.02.2023, noise - 29.03.2023
"""
import os
import numpy as np
import random
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from utils import cut_sample, get_all_indexes, get_fname_list, find_nearest


@tf.function
def generate_noise(k, P, pi, 
                   add_shot=True,
                   add_sys=True,
                   add_cosvar=True,
                   sys_scaled=False,
                   sys_factor=0.03,
                   sys_max=False,
                   V=tf.constant([10.43, 6.27, 3.34, 0.283]), 
                   nbar=tf.constant([0.000358, 0.000828, 0.00103, 0.00128]),
                   delta_k=0.055, sigma_sys=15, quadrature=True):

    sigma_noise = tf.zeros(P.shape)
    sigma_hat_noise = (2 * pi / ((k[:, None]) * tf.sqrt(V * (1e3)**3 * delta_k)))
    if add_cosvar:
        sigma_noise = tf.abs(P * sigma_hat_noise)
      
    if add_shot:
        sigma_noise_shot = (sigma_hat_noise / tf.convert_to_tensor(nbar, dtype=tf.float32))
        sigma_noise = sigma_noise + sigma_noise_shot
    if add_sys:
        if sys_scaled:
            if quadrature:
                sigma_noise = tf.sqrt(sigma_noise**2 + (P * sys_factor)**2)
            else:
                sigma_noise = sigma_noise + tf.abs(P * sys_factor)
        elif sys_max:
            sigma_noise = tf.maximum(sigma_noise, tf.abs(P * sys_factor))
        else:
            if quadrature:
                sigma_noise = tf.sqrt(sigma_noise**2 + sigma_sys**2)
            else:
                sigma_noise = sigma_noise + sigma_sys
     
    return tf.cast(sigma_noise, dtype=tf.float32)

class DataGenerator(tf.compat.v2.keras.utils.Sequence): 
    @tf.function
    def read_file(self, file_path, *, column_indices=None, dtype=tf.float32):
        """with tf.device('/cpu:0'):"""
        file_content = tf.io.read_file(file_path)

        file_content = tf.strings.regex_replace(file_content, "\r\n", "\n")
        file_content = tf.strings.regex_replace(file_content, "\r", "\n")
        lines = tf.strings.split([file_content], '\n').values
        lines = tf.cond(tf.equal(lines[-1], ""), lambda: lines[:-1], lambda: lines)

        def extract_columns(line):
            # Normalize spaces (convert tabs or multiple spaces to single space)
            line = tf.strings.regex_replace(line, "\s+", " ")
            line = tf.strings.strip(line)  # Strip leading/trailing whitespace
            columns = tf.strings.split([line], ' ').values
            if column_indices is not None:
                selected_columns = tf.gather(columns, column_indices)
            else:
                selected_columns = columns
    
            return tf.strings.to_number(selected_columns, out_type=dtype)

        columns_values = tf.map_fn(extract_columns, lines, fn_output_signature=dtype)
        if isinstance(columns_values, tf.RaggedTensor):
            # Convert the ragged tensor to a regular tensor with padding
            columns_values = columns_values.to_tensor()
        return columns_values

    def __init__(self, list_IDs, labels, labels_dict, batch_size=32, 
                data_root = 'data/', dim=(500, 4), n_channels=1,
                shuffle=True, normalization='stdcosmo',
                save_indexes=False, models_dir = 'models/', idx_file_name = '_',
                norm_data_name='/planck_hmcode2020.txt',
                fname = 'my_model',
                #fname_user='my_model',
                curves_folder = 'curve_files_sys/curve_files_train1k_sysFactor0o04_start0o03_dirChange0',
                sample_pace = 4, pad=False, 
                Verbose=False, Verbose_2=False,
                k_max=2.5, i_max = None,
                k_min=0.0, i_min = None,
                add_noise=True, n_noisy_samples = 10, 
                add_shot=True, add_sys=True, add_cosvar=True, sigma_sys=5,
                rescale_curves = None, 
                sys_scaled=False, sys_factor=0.03, sys_max=False,
                sigma_curves = 0.04,
                sigma_curves_default = 0.10,
                save_processed_spectra = False,
                fine_tune = False, 
                c_0=None, c_1=None, group_lab_dict=None,
                z_bins=[0, 1, 2, 3], swap_axes=False,
                dataset_balanced=False, test_mode=False, one_vs_all=False,
                seed=1234, TPU=False, strategy = None
                ):
      
        tf.print('Data Generator Initialization')
        tf.print('CWD: ', os.getcwd())
        self.one_vs_all=one_vs_all
        self.dataset_balanced=dataset_balanced
        self.sigma_sys=sigma_sys
        self.add_shot=add_shot
        self.add_sys=add_sys
        self.add_cosvar=add_cosvar
        self.sys_scaled=sys_scaled
        self.sys_factor=tf.constant(sys_factor, tf.float32)
        self.pi = tf.constant(np.pi, dtype=tf.float32)
        self.sys_max=sys_max
        if group_lab_dict:
            self.group_lab_dict = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(list(group_lab_dict.keys()), list(group_lab_dict.values())), default_value="", name="group_lab_mapping")
        else:
            self.group_lab_dict = None
        self.fine_tune=fine_tune
        self.c_0=c_0
        self.c_1=c_1
        self.fname = fname # name model
        #self.fname_user=fname_user
        self.curves_folder=curves_folder 
        self.sigma_curves = tf.constant(sigma_curves, tf.float32)
        self.sigma_curves_default = tf.constant(sigma_curves_default, tf.float32)
        self.save_processed_spectra = save_processed_spectra
        self.rescale_curves = rescale_curves
        self.models_dir = models_dir
        self.name_spectra_folder = os.fsdecode(os.path.join(self.models_dir, self.fname, 'processed_spectra'))
        self.seed = seed
        self.rng = tf.random.Generator.from_seed(self.seed)
        self.swap_axes = swap_axes 

        self.TPU=TPU
        self.strategy = strategy
        
        self.k_max=k_max
        self.k_min=k_min
        self.i_max=i_max
        self.i_min=i_min
        self.sample_pace=sample_pace
        if sample_pace ==1:
            self.dim = dim
        else:
            self.dim = (tf.cast((dim[0]/sample_pace), tf.int32), tf.cast(dim[1], tf.int32)) 
            self.n_channels = tf.convert_to_tensor(n_channels, tf.int32)
            self.z_bins=tf.convert_to_tensor(z_bins, dtype=tf.int32)
            
        tf.print('Using z bins %s' %z_bins)
        if not self.swap_axes:
            if self.z_bins.shape[0]!=self.dim[1]:
                raise ValueError('Number of z bins does not match dimension 1 of the data.')
        else:
            if self.z_bins.shape[0]!=self.n_channels:
                raise ValueError('Number of z bins does not match n_channels.')
        
        self.data_root=data_root
        self.norm_data_path = tf.io.gfile.join(self.data_root, norm_data_name)
        tf.print('Data root dir is %s' %self.data_root)
        tf.print('Normalisation file is %s' %self.norm_data_path)
        with tf.device('/cpu:0'):
            self.all_ks = self.read_file(self.norm_data_path, column_indices=[0], dtype=tf.float32)
            self.all_ks = tf.cast(self.all_ks, dtype=tf.float32)
            self.original_k_len = tf.cast(tf.size(self.all_ks).numpy(), tf.int32)
            if self.sample_pace !=1:
                    self.all_ks = self.read_file(self.norm_data_path, column_indices=[0], dtype=tf.float32)[::self.sample_pace]
        

        # Select points from k_max or i_max

        if self.k_max is not None:
            tf.print('Specified k_max is %s' %self.k_max)
            self.i_max, k_max_res = find_nearest(self.all_ks, self.k_max) 
            tf.print('Corresponding i_max is %s' %self.i_max.numpy())
            tf.print('Closest k to k_max is %s' %k_max_res.numpy())

        elif self.i_max is not None:
            self.k_max = tf.gather(self.all_ks, self.i_max)
            tf.print('Specified i_max is %s' %self.i_max)
            tf.print('Corresponding k_max is %s' %self.k_max.numpy())
            
        elif self.i_max is not None and self.k_max is not None:
            tf.print('Specified i_max is %s' %self.i_max)
            tf.print('Specified k_max is %s' %self.k_max)
            
            i_max, k_max = find_nearest(self.all_ks, self.k_max)
            assert(i_max==self.i_max)

        else:
            self.i_max = -1
            tf.print('No max in k. k_max=%s' %tf.gather(self.all_ks, self.i_max))

        # Select points from k_min or i_min

        if self.k_min is not None:
            tf.print('Specified k_min is %s' %self.k_min)
            self.i_min, k_min_res = find_nearest(self.all_ks, self.k_min) 
            tf.print('Corresponding i_min is %s' %self.i_min.numpy())
            tf.print('Closest k to k_min is %s' %k_min_res.numpy())

        elif self.i_min is not None:
            self.k_min = tf.gather(self.all_ks, self.i_min)
            tf.print('Specified i_min is %s' %self.i_min)
            tf.print('Corresponding k_min is %s' %self.k_min.numpy())
            
        elif self.i_min is not None and self.k_min is not None:
            tf.print('Specified i_min is %s' %self.i_min)
            tf.print('Specified k_min is %s' %self.k_min)
            
            i_min, k_min = find_nearest(self.all_ks, self.k_min)
            assert(i_min==self.i_min)

        else:
            self.i_min = 0
            tf.print('No min in k. k_min=%s' %tf.gather(self.all_ks, self.i_min))

        self.all_ks = self.all_ks[self.i_min:self.i_max]
        self.dim = (tf.cast(self.all_ks.shape[0], tf.int32), self.dim[1])
        tf.print('New data dim: %s' %(str(self.dim[0].numpy()) + "," + str(self.dim[1].numpy())))
        tf.print('Final i_max used is %s' %(self.i_max.numpy()))
        tf.print('Final i_min used is %s' %(self.i_min.numpy()))

        self.batch_size = tf.convert_to_tensor(batch_size, tf.int32)
        
        self.labels = labels
        #tf.print(self.labels)
        self.labels_dict = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(list(labels_dict.keys()), list(labels_dict.values())), default_value=-1, name="labels_dict_mapping")
        self.inv_labels_dict={value:key for key,value in zip(labels_dict.keys(), labels_dict.values())}

        #tf.print(self.inv_labels_dict)

        self.list_IDs = list_IDs
        if len(self.list_IDs)==1:
            self.list_IDs_dict = {label:list_IDs+i for i,label in enumerate(labels)}
            tf.print('Ids dict to use in data gen: %s' %str(self.list_IDs_dict))
        else:
            self.list_IDs_dict = {label:list_IDs for label in labels}
    
        
        self.base_case_dataset = not((self.fine_tune and self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and self.dataset_balanced))
        tf.print('one_vs_all: %s' %str(self.one_vs_all))
        tf.print('dataset_balanced: %s' %str(self.dataset_balanced))
        tf.print('base_case_dataset: %s' %str(self.base_case_dataset))
        
        
        self.n_classes_out = tf.constant(len(self.labels), dtype=tf.int32)
        if not self.base_case_dataset:
            self.n_classes = 2*(tf.constant(len(self.c_1), dtype=tf.int32))
        elif (self.fine_tune and not self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and not self.dataset_balanced):
            self.n_classes = tf.constant(len(self.c_1), dtype=tf.int32)+tf.constant(len(self.c_0), dtype=tf.int32)
        else:
            # regular 5 labels case
            self.n_classes =tf.constant(len(self.labels), dtype=tf.int32)
        
        
        tf.print('N. classes: %s' %self.n_classes.numpy()) 
        tf.print('N. n_classes in output: %s' %self.n_classes_out.numpy()) #number of labels to predict
        tf.print('LABELS:', self.labels)
            
        self.shuffle = shuffle
        #tf.print('Batch size: %s' %self.batch_size)
        #tf.print('N. samples used for each different label: %s' %self.n_indexes)
        self.save_indexes = save_indexes
        self.normalization=normalization
        
        if self.normalization=='stdcosmo':
            self.norm_data = self.read_file(self.norm_data_path, dtype=tf.float32)[:, 1:]
            self.norm_data = tf.cast(self.norm_data, dtype=tf.float32)

        if self.sample_pace !=1:
            self.norm_data = self.norm_data[0::self.sample_pace, :]
        self.norm_data = self.norm_data[self.i_min:self.i_max]
        
        self.pad=pad
        self.add_noise=add_noise
        if not self.add_noise:
            self.n_noisy_samples = 1
        else:
            self.n_noisy_samples = n_noisy_samples
            self.n_noisy_samples = tf.convert_to_tensor(self.n_noisy_samples, dtype=tf.int32)
        
        ######
        # Consistency checks
        ######
        
        if not self.base_case_dataset:
            if self.batch_size%(self.n_classes*self.n_noisy_samples):
                tf.print('batch_size,n_classes, len(c_1), n_noisy_samples= %s, %s, %s, %s '%(self.batch_size.numpy(), self.n_classes.numpy(), len(self.c_1), self.n_noisy_samples.numpy()))
                raise ValueError('batch size must be multiple of n_classes x len(c_1) x n_noisy_samples')
        elif not(self.fine_tune and self.dataset_balanced) or not(not self.fine_tune and self.one_vs_all and self.dataset_balanced):
            if self.batch_size%(self.n_classes*self.n_noisy_samples):
                raise ValueError('batch size must be multiple of n_classes x n_noisy_samples')
        else:
            raise ValueError('check dataset_balanced and one_vs_all compatibility')
            
        if not self.base_case_dataset:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                tf.print('Batch size = %s' %self.batch_size.numpy())
                #tf.print('( n_labels x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
                raise ValueError('Batch size must be multiple of n_classes x len(c_1)  x (n_noisy_samples) ')
            self.n_indexes = len(self.c_1)*self.batch_size//(self.n_classes*self.n_noisy_samples) #len(self.c_1)*
            self.n_indexes = tf.convert_to_tensor(self.n_indexes, dtype=tf.int32)
            tf.print('batch_size, n_classes, len(self.c_1), n_noisy_samples= %s, %s, %s, %s' %(self.batch_size.numpy(), self.n_classes.numpy(), len(self.c_1), self.n_noisy_samples.numpy()))
            tf.print('n_indexes=len(self.c_1)*batch_size//(n_classes*n_noisy_samples)=%s' %self.n_indexes.numpy())
            
        else:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                tf.print('Batch size = %s' %self.batch_size.numpy())
                tf.print('( n_classes x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples).numpy())
                raise ValueError('Batch size must be multiple of (number of classes) x (n_noisy_samples) ')
            self.n_indexes = self.batch_size//(self.n_classes*self.n_noisy_samples) # now many index files to read per each batch
            self.n_indexes = tf.convert_to_tensor(self.n_indexes, dtype=tf.int32)
        
        self.n_batches = tf.convert_to_tensor(len(list_IDs)//(self.n_indexes))
        tf.print('list_IDs length: %s' %len(list_IDs))
        tf.print('n_indexes (n of file IDs read for each batch): %s' %self.n_indexes.numpy())
        tf.print('batch size: %s' %self.batch_size.numpy())
        tf.print('n_batches : %s' %self.n_batches.numpy())
        if self.n_batches==tf.constant(0):
            raise ValueError('Not enough examples to support this batch size ')

        tf.print('For each batch we read %s file IDs' %self.n_indexes.numpy())
        if not self.fine_tune or not self.dataset_balanced:
            tf.print('For each file ID we have %s labels' %(self.n_classes.numpy()))
        else:
            tf.print('We read %s IDs for label %s and 1 ID for each of the labels %s' %(str(len(self.c_1)), c_0[0],str( c_1)) )
        if self.add_noise:
            tf.print('For each ID, label we have %s realizations of noise' %self.n_noisy_samples.numpy())
        
        if self.base_case_dataset:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples
            n_check = tf.cast(self.n_classes*self.n_noisy_samples, tf.int32)
        else:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples/len(self.c_1)
            n_check = tf.cast(self.n_classes*self.n_noisy_samples, tf.int32)#*len(self.c_1) # n_indexes must be a multiple of this x batch_size
        
        tf.print('In total, for each batch we have %s training examples' %(n_ex.numpy()))
        tf.print('Input batch size: %s' %self.batch_size.numpy())
        tf.print('N of batches to cover all file IDs: %s' %self.n_batches.numpy())
        if n_ex!=self.batch_size:
            raise ValueError('Effective batch size does not match input batch size')
        
        if self.n_indexes%tf.cast((self.batch_size/(n_check)), dtype=tf.int32)!=0:
            tf.print('Batch size = %s' %self.batch_size.numpy())
            tf.print('( n_labels x n_noisy_samples) = %s' %(n_check*self.n_noisy_samples).numpy())
            tf.print('n_indexes = %s' %self.n_indexes.numpy())
            raise ValueError('Batch size should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer ')
        
        
        if self.n_indexes!=tf.cast(tf.convert_to_tensor(len(list_IDs))/self.n_batches, dtype=tf.int32): 
            tf.print('length of IDs = %s' %str(len(list_IDs)))
            tf.print('n_batches = %s' %self.n_batches.numpy())
            tf.print('n_indexes = %s' %self.n_indexes.numpy())
            tf.print('len(list_IDs)/self.n_batches = %s' %(len(list_IDs)/self.n_batches).numpy())
            raise ValueError('n_batches does not match length of IDs')
        self.Verbose=Verbose
        self.Verbose_2=Verbose_2

        self.dataset=self.__data_generation(self.list_IDs, self.list_IDs_dict)
         

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches#int(np.floor(len(self.list_IDs)*self.n_classes*self.n_noisy_samples / self.batch_size))
    
    def __shape__(self):
        'I dont know what exactly I should put here - where is n_channels ??? '
        return((len(self.list_IDs), self.dim[0]/self.sample_pace, self.dim[1] ))
    
    @tf.function
    def noise_realisation(self, P_original, k, i_noise, P_noise, fname):
        if self.add_noise:
            P_noisy = P_original
            if self.Verbose:
                tf.print('Noise realization %s' %i_noise)
            # add noise if selected
            if self.add_cosvar:
                if self.TPU:
                    with self.strategy.scope():
                        noise_scale = generate_noise(k, P_noise, self.pi, sys_scaled=self.sys_scaled,
                                                sys_factor=self.sys_factor,sys_max=self.sys_max,
                                                add_cosvar=True, add_sys=False, add_shot=False, sigma_sys=self.sigma_sys)
                        noise_cosVar = self.rng.normal(shape=noise_scale.shape, mean=0, stddev=noise_scale)
                        P_noisy = P_noisy + noise_cosVar
                else:
                    noise_scale = generate_noise(k, P_noise, self.pi, sys_scaled=self.sys_scaled,
                                                    sys_factor=self.sys_factor,sys_max=self.sys_max,
                                                    add_cosvar=True, add_sys=False, add_shot=False, sigma_sys=self.sigma_sys)
                    noise_cosVar = self.rng.normal(shape=noise_scale.shape, mean=0, stddev=noise_scale)
                    P_noisy = P_noisy + noise_cosVar
            if self.add_sys:
                curves_loaded = self.curves_loaded[i_noise]
                noise_sys, k_sys = curves_loaded[:, 1:], curves_loaded[:, 0]
                if self.sample_pace!=1:
                    noise_sys = noise_sys[0::self.sample_pace, :]
                    k_sys = k_sys[0::self.sample_pace]
                noise_sys, k_sys = noise_sys[self.i_min:self.i_max], k_sys[self.i_min:self.i_max]
                """if tf.reduce_any(tf.math.not_equal(k, k_sys)): WONT WORK ANYMORE
                    tf.print('ERROR: k-values in spectrum and theory-error curve file not identical')
                    """
                # rescale noise_sys curves according to error (10% default from production curves), 
                # rescale by Gaussian with sigma = 1
                # multiply with normalisation spectrum
                if self.TPU:
                    with self.strategy.scope():
                        noise_sys = (noise_sys-1) * self.sigma_curves/self.sigma_curves_default  * P_noise
                        noise_sys = tf.cast(noise_sys, dtype=tf.float32)
                        if self.rescale_curves == 'uniform':
                            noise_sys = noise_sys * self.rng.uniform(shape=noise_sys.shape, minval=0, maxval=1, dtype=tf.float32)
                        if self.rescale_curves == 'gaussian':
                            noise_sys = noise_sys * self.rng.normal(shape=noise_sys.shape, mean=0, stddev=1, dtype=tf.float32)
                        P_noisy = tf.cast(P_noisy, dtype=tf.float32)
                        P_noisy = P_noisy + noise_sys
                else:
                    noise_sys = (noise_sys-1) * self.sigma_curves/self.sigma_curves_default  * P_noise
                    noise_sys = tf.cast(noise_sys, dtype=tf.float32)

                    if self.rescale_curves == 'uniform':
                        noise_sys = noise_sys * self.rng.uniform(shape=noise_sys.shape, minval=0, maxval=1, dtype=tf.float32)
                    if self.rescale_curves == 'gaussian':
                        noise_sys = noise_sys * self.rng.normal(shape=noise_sys.shape, mean=0, stddev=1, dtype=tf.float32)

                    P_noisy = tf.cast(P_noisy, dtype=tf.float32)
                    P_noisy = P_noisy + noise_sys
            if self.add_shot:
                if self.TPU:
                    with self.strategy.scope():
                        noise_scale = generate_noise(k, P_noise, self.pi, sys_scaled=self.sys_scaled, sys_factor=self.sys_factor, sys_max=self.sys_max, add_cosvar=False, add_sys=False, add_shot=True, sigma_sys=self.sigma_sys)
                        noise_shot = self.rng.normal(shape=noise_scale.shape, mean=0, stddev=noise_scale)
                        P_noisy = P_noisy + noise_shot
                else:
                    noise_scale = generate_noise(k,P_noise,self.pi,sys_scaled=self.sys_scaled,sys_factor=self.sys_factor,sys_max=self.sys_max, add_cosvar=False, add_sys=False, add_shot=True,sigma_sys=self.sigma_sys)
                    noise_shot = self.rng.normal(shape=noise_scale.shape, mean=0, stddev=noise_scale)
                    P_noisy = P_noisy + noise_shot

            expanded = tf.expand_dims(P_noisy, axis=2)
        else:
            if self.Verbose:
                tf.print('No noise')
            expanded = tf.expand_dims(P_original, axis=2)
        # Store sample
        if self.Verbose:
            tf.print('Dimension of data: %s' %str(expanded.shape))
        # swap axis if using one dim array in multiple channels 
        if self.swap_axes:
            if self.Verbose:
                tf.print('Reshaping')
            expanded = tf.transpose(expanded, perm=[0, 2, 1])
            if self.Verbose:
                tf.print('New dimension of data: %s' %str(expanded.shape))
            expanded = tf.gather(expanded, self.z_bins, axis=2)
            if self.Verbose:
                tf.print('expanded first 5:') 
                tf.print(expanded[5])
            # now shape of expanded is (1, n_data_points (k values), 1, n_channels)
        if self.Verbose:
            tf.print('Dimension of data before normalising: %s' %str(expanded.shape))
        X = expanded
        X = tf.cast(X, dtype=tf.float32)              
        if self.Verbose:
            tf.print('dimension of X: %s' %str(X.shape))
            tf.print('X first 5:') 
            tf.print(X[5])

        # Store class   
        label = tf.strings.split(fname, '/')[-2]
        if not self.base_case_dataset:
            label = self.group_lab_dict.lookup(label)
            encoding = self.labels_dict.lookup(label)
        elif (self.fine_tune and not self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and not self.dataset_balanced):
            label = self.group_lab_dict.lookup(label)
            encoding = self.labels_dict.lookup(label)
        else:
            # regular 5 labels case
            encoding = self.labels_dict.lookup(label)
        
        y = tf.cast(encoding, tf.int32)
        return X, y

    @tf.function
    def process_file(self, ID, fname):
        loaded_all = self.read_file(fname, dtype=tf.float32)

        P_original = loaded_all[:, 1:]
        k = loaded_all[:, 0] 

        if self.sample_pace != 1:
            P_original = P_original[::self.sample_pace]
            k = k[::self.sample_pace]
        P_original, k = P_original[self.i_min:self.i_max], k[self.i_min:self.i_max]

        """ WONT WORK ANYMORE
        if self.Verbose:
            tf.print('dimension P_original: %s' %str(P_original.shape))    
            tf.print('P_original first 10:') 
            tf.print(P_original[10])
        """
        
        if self.add_noise:
            P_noise = tf.gather(self.norm_data, self.z_bins, axis=1)

        def loop_over_noise(i_noise, X, y, k, P_noise, fname):
            def condition(i, X, y):
                return i < self.n_noisy_samples
            def body(i, X, y):
                X, y = self.noise_realisation(P_original, k, i, P_noise, fname)
                return [tf.add(i, 1), X, y]
            shape_invariants = [i_noise.get_shape(), tf.TensorShape([self.n_ks, None, self.n_channels]), y.get_shape()]
            i_noise, X, y = tf.while_loop(condition, body, [i_noise, X, y], shape_invariants=shape_invariants)
            return X, y

        i_noise = tf.constant(0, dtype=tf.int32)
        y = tf.constant(0, dtype=tf.int32)
        X, y = loop_over_noise(i_noise, P_original, y, k, P_noise, fname)

        return ID, X, y

    @tf.function
    def normalize_and_onehot(self, ID, X, y):
        if self.normalization == 'batch':
            mu_batch = tf.reduce_mean(X, axis=0)
            std_batch = tf.math.reduce_std(X, axis=0)
            X = (X - mu_batch) / std_batch
        elif self.normalization == 'stdcosmo':
            if self.swap_axes:
                if self.TPU:
                    with self.strategy.scope():
                        divisor = tf.gather(self.norm_data, self.z_bins, axis=1)
                        X = X / divisor - 1
                else:
                    divisor = tf.expand_dims(tf.expand_dims(self.norm_data, axis=0), axis=-1)
                    X = X / divisor - 1
                if self.Verbose:
                    tf.print('axes swapped')
                    tf.print('NORM first 10:', divisor[:10])
            else:
                divisor = tf.expand_dims(tf.expand_dims(self.norm_data, axis=0), axis=-1)
                X = X / divisor - 1
                if self.Verbose:
                    tf.print('axes not swapped')
                    tf.print('Dimension of NORM data:', tf.shape(divisor))
        if self.save_processed_spectra and not self.TPU:
            X_save_init = tf.zeros((self.batch_size * self.n_batches + 1, len(self.all_ks) + 1), dtype=tf.float32)
            y_expanded = tf.expand_dims(tf.concat([[0], y], axis=0), axis=-1)
            all_ks_expanded = tf.expand_dims(tf.concat([[0], self.all_ks], axis=0), axis=0)
            X_save = tf.tensor_scatter_nd_update(X_save_init, tf.range(tf.shape(y_expanded)[0])[:, tf.newaxis], y_expanded)
            X_save = tf.tensor_scatter_nd_update(X_save, tf.range(tf.shape(all_ks_expanded)[0])[tf.newaxis, :], all_ks_expanded)
            def write_spectra(z_bins, X, X_save):
                loop_len = tf.size(z_bins)
                def condition(i, X, X_save):
                    return i < loop_len
                def body(i, X, X_save):
                    z = z_bins[i]
                    X_slice = X[:, :, 0, z]
                    X_save_updated = tf.tensor_scatter_nd_update(X_save, tf.range(1, tf.shape(X_slice)[1] + 1)[:, tf.newaxis], X_slice)
                    spectra_file = tf.strings.join([self.name_spectra_folder, f'processed_spectra_zbin{i}.txt'], separator='/')
                    tf.print(f'Saving processed (noisy and normalised) spectra in {spectra_file}')
                    X_save_string = tf.strings.reduce_join(tf.strings.as_string(X_save_updated), separator=' ', axis=-1)
                    tf.io.write_file(spectra_file, X_save_string)
                    return [tf.add(i, 1), X, X_save]
                i = tf.constant(0, dtype=tf.int32)
                tf.while_loop(condition, body, [i, X, X_save])
            write_spectra(self.z_bins, X, X_save)

                
        if self.swap_axes:
            X = X[:,:,0,:]
            X = X[0,:,:]
        y = tf.one_hot(y, depth=self.n_classes_out)

        self.xshape_file = X.shape
        self.yshape_file = y.shape
        if self.Verbose:
            tf.print('Dimension of data after normalising: %s' %str(X.shape))
            tf.print('Dimension of labels after one-hot encoding: %s' %str(y.shape))
        return ID, X, y
    
    @tf.function
    def load_sys(self, i):
        curve_random_nr = self.rng.uniform(shape=[], minval=1, maxval=1001, dtype=tf.int32)
        curve_nr_string = tf.strings.as_string(curve_random_nr)
        curve_nr_string = tf.strings.join([curve_nr_string, '.txt'])
        curve_file = tf.strings.join([self.curves_folder, curve_nr_string], separator='/')
        curve_dat = self.read_file(curve_file, dtype=tf.float32)
        return curve_dat

    def __data_generation(self, list_IDs, list_IDs_dict):
        'Generates a batched DataSet'
        if not self.fine_tune and not self.one_vs_all:
            fname_list=[]
            ID_list=[]
            for l in self.labels:
                for ID in list_IDs_dict[l]:
                    t_st =  self.data_root + '/'+l+ '/'+ str(ID) + '.txt' 
                    fname_list.append(t_st)
                    ID_list.append(ID)  
        else:
            fname_list = get_fname_list(self.c_0, self.c_1, list_IDs, self.data_root,  list_IDs_dict, dataset_balanced=self.dataset_balanced,)
            ID_list = [int(fname.split('/')[-1].split('.')[0]) for fname in fname_list]
        ID_list = tf.convert_to_tensor(ID_list, dtype=tf.int32)
        if self.fine_tune and self.Verbose :
            tf.print(fname_list)
        
        self.n_noisy_samples_numpy = self.n_noisy_samples.numpy()
        self.n_ks = tf.size(self.all_ks)

        tf.print('len(fname_list), batch_size, n_noisy_samples: %s, %s, %s' %(len(fname_list), self.batch_size.numpy(), self.n_noisy_samples_numpy))
        tf.debugging.assert_equal(tf.constant(len(fname_list), dtype=tf.int32), self.batch_size * self.n_batches // self.n_noisy_samples_numpy, message="Fname list != batch_size * n_batches // n_noisy_samples")

        fname_list = tf.convert_to_tensor(fname_list, dtype=tf.string)

        if self.Verbose_2:
            tf.print("list_IDs_dict")
            tf.print(list_IDs_dict)

        # Load n_noisy_samples random sys noise curves
        if self.add_noise and self.add_sys:
            self.curves_loaded = tf.map_fn(self.load_sys, tf.range(self.n_noisy_samples), dtype=tf.float32)
        
        if self.save_processed_spectra and not self.TPU:
            if not tf.io.gfile.exists(self.name_spectra_folder):
                tf.print('Creating directory %s' %  self.name_spectra_folder)
                tf.io.gfile.makedirs(self.name_spectra_folder)
        elif self.TPU:
            tf.print("WARNING: Cannot save processed spectra in TPU mode.")

        #Process spectrum files
        dataset = tf.data.Dataset.from_tensor_slices((ID_list, fname_list)) 
        if self.TPU:
            with tf.device('/cpu:0'):
                dataset = dataset.map(self.process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            with self.strategy.scope():
                dataset = dataset.map(self.normalize_and_onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                if self.shuffle:
                    dataset = dataset.shuffle(buffer_size=len(list_IDs))
                global_batchsize = self.batch_size * self.strategy.num_replicas_in_sync
                global_batchsize = tf.cast(global_batchsize, dtype=tf.int64)
                dataset = dataset.batch(global_batchsize)
                dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
                dataset = self.strategy.experimental_distribute_dataset(dataset)

        else:
            dataset = dataset.map(self.process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.normalize_and_onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)  
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=len(list_IDs))  
            global_batchsize = self.batch_size   
            global_batchsize = tf.cast(global_batchsize, dtype=tf.int64)
            dataset = dataset.batch(global_batchsize)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.xshape = ((self.batch_size * self.n_batches).numpy(),) + tuple(self.xshape_file)
        self.yshape = ((self.batch_size * self.n_batches).numpy(),) + tuple(self.yshape_file)

        
        return dataset
    
    def write_indexes(self, batch_ID, indices):
        batch_ID = batch_ID.numpy()
        indices = indices.numpy()
        idx_files_dir = tf.io.gfile.join(self.models_dir, 'idx_files')
        if not tf.io.gfile.exists(idx_files_dir):
            tf.print('Creating directory %s' % idx_files_dir)
            tf.io.gfile.makedirs(idx_files_dir)
        idx_file = self.models_dir+'/idx_files/idx_file_batch'+ str(batch_ID)+'.txt'                  
        tf.print('Saving indexes in  %s' %idx_file)
        idx_list = indices
        with open(idx_file, 'w+') as file:
            tf.print('Opened %s' %idx_file)
            for idx in idx_list: #i in range(len(idx_list)):
                file.write(idx+'\n')
    



def read_partition(FLAGS):
    out_path = FLAGS.models_dir+FLAGS.fname
    base_path = out_path+'/tf_ckpts/'
    fname_idxs_train=base_path+'idxs_train.txt'
    fname_idxs_val=base_path+'idxs_val.txt'
    
    tf.print('Reading train indexes from %s ...' %fname_idxs_train)
    train_idxs_dataset = tf.data.TextLineDataset(fname_idxs_train)
    train_idxs = tf.convert_to_tensor([int(idx.numpy()) for idx in train_idxs_dataset])
    tf.print('Train indexes length: %s' % tf.shape(train_idxs)[0])
    tf.print('Reading val indexes from %s ...' % fname_idxs_val)
    val_idxs_dataset = tf.data.TextLineDataset(fname_idxs_val)
    val_idxs = tf.convert_to_tensor([int(idx.numpy()) for idx in val_idxs_dataset])
    tf.print('Val indexes length: %s' %str(len(val_idxs)))
    
    partition = {'train': train_idxs , 'validation': val_idxs }
    return partition
    
    

def create_generators(FLAGS, strategy = None):
    if FLAGS.my_path is not None:
        os.chdir(FLAGS.my_path)
    
    
    # --------------------  CREATE DATA GENERATORS   --------------------
    
    all_index, n_samples, val_size, n_labels, labels, labels_dict, all_labels = get_all_indexes(FLAGS)
    tf.print('create_generators n_labels: %s' %n_labels) 
    if (FLAGS.fine_tune or FLAGS.one_vs_all) and FLAGS.dataset_balanced:
        # balanced dataset , 1/2 lcdm , 1/2 rest in FT or one vs all mode
        case=1
        n_labels_eff = n_labels*len(FLAGS.c_1)
        len_c1=len(FLAGS.c_1)
    elif not (FLAGS.fine_tune or FLAGS.one_vs_all):
        # regular case
        case=2
        n_labels_eff=n_labels
        len_c1=1
    elif (FLAGS.fine_tune or FLAGS.one_vs_all) and not FLAGS.dataset_balanced:
        #  Unbalanced dataset , 1/5 lcdm , 1/5 rest in FT or one vs all mode
        case=3
        n_labels_eff = len(all_labels)
        if FLAGS.one_vs_all and len(FLAGS.c_1)<len(all_labels)-1:
            n_labels_eff = len(FLAGS.c_1)+len(FLAGS.c_0)
        len_c1=1
    tf.print('create_generators n_labels_eff: %s' %n_labels_eff)  
    tf.print('create_generators len_c1: %s' %len_c1)
        
        

    
    # SPLIT TRAIN/VALIDATION /(TEST)
    val_index = np.random.choice(all_index, size=int(np.floor(val_size*n_samples)), replace=False)
    train_index_temp =  np.setdiff1d(all_index, val_index) #np.delete(all_index, val_index-1)
    test_size_eff = FLAGS.test_size/(train_index_temp.shape[0]/n_samples)
    test_index = np.random.choice(train_index_temp, size=int(np.floor(test_size_eff*train_index_temp.shape[0])), replace=False)
    train_index =  np.setdiff1d(train_index_temp, test_index)

    tf.print('Check for no duplicates in test: (0=ok):')
    tf.print(np.array([np.isin(el, train_index) for el in test_index]).sum())
    tf.print('Check for no duplicates in val: (0=ok):')
    tf.print(np.array([np.isin(el, train_index) for el in val_index]).sum())

    tf.print('N of files in training set: %s' %train_index.shape[0])
    tf.print('N of files in validation set: %s' %val_index.shape[0])
    tf.print('N of files in test set: %s' %test_index.shape[0])

    tf.print('Check - total: %s' %(val_index.shape[0]+test_index.shape[0]+train_index.shape[0]))
    
    if FLAGS.add_noise:
        n_noisy_samples = FLAGS.n_noisy_samples
    else:
        n_noisy_samples = 1
    tf.print('--create_generators, train indexes')
    if FLAGS.test_mode:
        if case==3:
            batch_size=train_index.shape[0]*n_labels_eff*n_noisy_samples
        elif case==2:
            batch_size=train_index.shape[0]*n_labels*n_noisy_samples
        elif case==1:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    tf.print('batch_size: %s' %batch_size)

    if not FLAGS.test_mode:
        train_index_1  = cut_sample(train_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False, len_c1=len_c1)
        tf.print('Train index length: %s' %train_index_1.shape[0])
    else:
        train_index_1 = train_index
        tf.print('Train index: %s' %train_index_1)
    tf.print('--create_generators, validation indexes')
    if not FLAGS.test_mode:
        val_index_1  = cut_sample(val_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False,len_c1=len_c1)
        tf.print('Val index length: %s'  %val_index_1.shape[0])
    else:
        val_index_1 = val_index
        tf.print('Validation index: %s' %val_index_1)
    
    tf.print('len(train_index_1), batch_size, n_labels_eff, n_noisy_samples = %s, %s, %s, %s' %(train_index_1.shape[0], batch_size, n_labels_eff,n_noisy_samples ))
    assert train_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0
    assert val_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0
    
    partition={'train': train_index_1, 'validation': val_index_1}
    
    
    if FLAGS.restore:
        partition = read_partition(FLAGS)
        batch_size=FLAGS.batch_size
        
    ###################
    # USE THE BLOCH BELOW TO BE COMPATIBLE WITH OLDER VERSIONS OF DARTA GENERATORS. EVENTUALLY REMOVE
    ###################
    try:
        sigma_sys=FLAGS.sigma_sys
    except AttributeError:
        tf.print(' ####  FLAGS.sigma_sys not found! #### \n Probably loading an older model. Using sigma_sys=0')
        sigma_sys=0.
        
    try:
        z_bins=FLAGS.z_bins
    except AttributeError:
        tf.print(' ####  FLAGS.z_bins not found! #### \n Probably loading an older model. Using 4 z bins')
        z_bins=[0, 1, 2, 3]
    try:
        swap_axes=FLAGS.swap_axes
    except AttributeError:
        if FLAGS.im_channels>1:
            swap_axes=True
        else:
            swap_axes=False
        tf.print(' ####  FLAGS.swap_axes not found! #### \n Probably loading an older model. Set swap_axes=%s' %str(swap_axes))
    ###################
        
    ##Generate a random seed for a tensorflow random number generator used in the class to prevent issues in parallel processing when in TPU mode
    seed = np.random.randint(0, 2**32 - 1)
    
    
    params = {'dim': (FLAGS.im_depth, FLAGS.im_width),
        'batch_size':batch_size, # should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer
        'n_channels': FLAGS.im_channels,
        'shuffle': True,
        'normalization': FLAGS.normalization,
        'sample_pace': FLAGS.sample_pace,
        'add_noise':FLAGS.add_noise,
        'n_noisy_samples':n_noisy_samples,
        'fine_tune':FLAGS.fine_tune,
        'add_shot':FLAGS.add_shot, 'add_sys':FLAGS.add_sys,'add_cosvar':FLAGS.add_cosvar,
        'k_max':FLAGS.k_max, 'i_max':FLAGS.i_max, 'k_min':FLAGS.k_min, 'i_min':FLAGS.i_min, 'sigma_sys':sigma_sys,
        'swap_axes':swap_axes,
        'z_bins':z_bins,
        'test_mode':FLAGS.test_mode,
        'normalization':FLAGS.normalization,
        'models_dir':FLAGS.models_dir,
        'norm_data_name':FLAGS.norm_data_name,
        'fine_tune':FLAGS.fine_tune ,
        'fname':FLAGS.fname,
        #'c_0':FLAGS.c_0 ,
        #'c_1':FLAGS.c_1 ,    
        'curves_folder':FLAGS.curves_folder,  
        'sigma_curves':FLAGS.sigma_curves,
        'sigma_curves_default':FLAGS.sigma_curves_default,
        'save_processed_spectra':FLAGS.save_processed_spectra,
        'rescale_curves':FLAGS.rescale_curves,
        'TPU':FLAGS.TPU
        }
    
    if FLAGS.fine_tune  or FLAGS.one_vs_all:
        params['c_0'] = FLAGS.c_0
        params['c_1'] = FLAGS.c_1
        params['group_lab_dict'] = FLAGS.group_lab_dict
        params['dataset_balanced']=FLAGS.dataset_balanced
        params['one_vs_all']=FLAGS.one_vs_all
        
    
    if not params['add_noise']:
        params['n_noisy_samples']=1
    
    tf.print('\n--DataGenerator Train')
    training_generator = DataGenerator(partition['train'], labels, labels_dict, data_root = FLAGS.DIR, save_indexes=False, seed = seed, strategy=strategy, **params)
    tf.print('\n--DataGenerator Validation')
    validation_generator = DataGenerator(partition['validation'], labels, labels_dict, data_root = FLAGS.DIR,  save_indexes=False, seed = seed, strategy = strategy, **params)

    
    return training_generator, validation_generator #, params




def create_test_generator(FLAGS):
    
    if FLAGS.my_path is not None:
        tf.print('Changing directory to %s' %FLAGS.my_path)
        os.chdir(FLAGS.my_path)
    
    all_index, n_samples, val_size, n_labels, labels, labels_dict, all_labels = get_all_indexes(FLAGS, Test=True)
    

    if (FLAGS.fine_tune or FLAGS.one_vs_all) and FLAGS.dataset_balanced:
        # balanced dataset , 1/2 lcdm , 1/2 rest in FT or one vs all mode
        case=1
        n_labels_eff = n_labels*len(FLAGS.c_1)
        len_c1=len(FLAGS.c_1)
    elif not (FLAGS.fine_tune or FLAGS.one_vs_all):
        # regular case
        case=2
        n_labels_eff=n_labels
        len_c1=1
    elif (FLAGS.fine_tune or FLAGS.one_vs_all) and not FLAGS.dataset_balanced:
        #  Unbalanced dataset , 1/5 lcdm , 1/5 rest in FT or one vs all mode
        case=3
        n_labels_eff = len(all_labels)
        len_c1=1
    tf.print('create_generators n_labels_eff: %s' %n_labels_eff)  
    tf.print('create_generators len_c1: %s' %len_c1)

    #if FLAGS.fine_tune:
    #    n_labels_eff = n_labels*len(FLAGS.c_1)
    #else:
    #    n_labels_eff = n_labels
    
    #if FLAGS.fine_tune and FLAGS.dataset_balanced:
    #    n_labels_eff = n_labels*len(FLAGS.c_1)
    #    len_c1=len(FLAGS.c_1)
    #elif not FLAGS.fine_tune:
    #    n_labels_eff=n_labels
    #    len_c1=1
    #elif FLAGS.fine_tune and not FLAGS.dataset_balanced:
    #    n_labels_eff = len(all_labels)
    #    len_c1=1
        
    
    if FLAGS.add_noise:
        n_noisy_samples = FLAGS.n_noisy_samples
    else:
        n_noisy_samples = 1
    tf.print('--Train')
    if FLAGS.test_mode:
        if not FLAGS.fine_tune:
            batch_size=all_index.shape[0]*n_labels_eff*n_noisy_samples
        else:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    tf.print('batch_size: %s' %batch_size)
        
    test_index_1  = cut_sample(all_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=True, len_c1=len_c1)
    n_test = test_index_1.shape[0]

    assert test_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0

    tf.print('N. of test files used: %s' %n_test)

    partition_test = {'test': test_index_1}
    
    
 
    
    params_test = {'dim': (FLAGS.im_depth, FLAGS.im_width),
          'batch_size':batch_size, # should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer
          'n_channels': FLAGS.im_channels,
          'shuffle': True,
          'normalization': FLAGS.normalization,
          'sample_pace': FLAGS.sample_pace,
          'add_noise':FLAGS.add_noise,
          'n_noisy_samples':n_noisy_samples, 
          'fine_tune':FLAGS.fine_tune,
          'add_shot':FLAGS.add_shot, 'add_sys':FLAGS.add_sys,'add_cosvar':FLAGS.add_cosvar,
          'k_max':FLAGS.k_max, 'i_max':FLAGS.i_max, 'k_min':FLAGS.k_min, 'i_min':FLAGS.i_min, 'sigma_sys':FLAGS.sigma_sys,
          'swap_axes':FLAGS.swap_axes,
          'z_bins':FLAGS.z_bins,
          'test_mode':FLAGS.test_mode,
          'normalization':FLAGS.normalization,
          'norm_data_name':FLAGS.norm_data_name,
          'fine_tune':FLAGS.fine_tune ,
          'fname':FLAGS.fname,
          #'c_0':FLAGS.c_0 ,
          #'c_1':FLAGS.c_1 ,  
          'curves_folder':FLAGS.curves_folder,
          'sigma_curves':FLAGS.sigma_curves,
          'sigma_curves_default':FLAGS.sigma_curves_default,
          'save_processed_spectra':FLAGS.save_processed_spectra,
          'rescale_curves':FLAGS.rescale_curves,
          }
    
    if FLAGS.fine_tune or FLAGS.one_vs_all:
        params_test['c_0'] = FLAGS.c_0
        params_test['c_1'] = FLAGS.c_1
        params_test['group_lab_dict'] = FLAGS.group_lab_dict
        params_test['dataset_balanced']=FLAGS.dataset_balanced
        params_test['one_vs_all']=FLAGS.one_vs_all
        

    if not params_test['add_noise']:
        params_test['n_noisy_samples']=1
    
    seed = np.random.randint(0, 2**32 - 1) 
    
    test_generator = DataGenerator(partition_test['test'], 
                                   labels, labels_dict, 
                                   data_root=FLAGS.TEST_DIR , 
                               save_indexes = FLAGS.save_indexes,
                               models_dir=FLAGS.models_dir+FLAGS.fname,
                               idx_file_name = FLAGS.fname, seed = seed, strategy = None, TPU = False,
                               **params_test)


    
    return test_generator

