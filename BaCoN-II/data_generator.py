#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:07:43 2020

@author: Michi
@edit: Arnav 24/03 Dataset processing optimisations and TPU compatibility
"""
import os
import numpy as np
import random
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from utils import cut_sample, get_all_indexes, get_fname_list, find_nearest

import tensorflow as tf

import tensorflow as tf

def generate_noise(k, P, 
                    add_shot=True,
                    add_sys=True,
                    add_cosvar=True,
                    sys_scaled=False,
                    sys_factor=0.03,
                    sys_max=False,
                    V=tf.constant([10.43, 6.27, 3.34, 0.283]), 
                    nbar=tf.constant([0.000358, 0.000828, 0.00103, 0.00128]),
                    delta_k = 0.055,  sigma_sys=15, quadrature=True):
    
    sigma_noise = tf.zeros_like(P)
    sigma_hat_noise = (2*tf.constant(np.pi)/((k[:, None])*tf.sqrt(V*(1e3)**3*delta_k)))
    if add_cosvar:
            sigma_noise = tf.abs(P*sigma_hat_noise)
            
    if add_shot:
            sigma_noise_shot=(sigma_hat_noise/tf.constant(nbar))
            sigma_noise = sigma_noise+sigma_noise_shot
    if add_sys:
            if sys_scaled:
                if quadrature:
                    sigma_noise = tf.sqrt(sigma_noise**2+(P*sys_factor)**2)
                else:
                    sigma_noise = sigma_noise+tf.abs(P*sys_factor)
            elif sys_max:
                sigma_noise = tf.maximum(sigma_noise, tf.abs(P*sys_factor))
            else:
                if quadrature:
                    sigma_noise = tf.sqrt(sigma_noise**2+sigma_sys**2 )
                else:
                    sigma_noise = sigma_noise+sigma_sys
         
    return sigma_noise


class DataSet(): 
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
      
        print('DataSet Initialization')
        
        self.one_vs_all=one_vs_all
        self.strategy = strategy
        self.dataset_balanced=dataset_balanced
        self.sigma_sys=sigma_sys
        self.add_shot=add_shot
        self.add_sys=add_sys
        self.add_cosvar=add_cosvar
        self.sys_scaled=sys_scaled
        self.sys_factor=sys_factor
        self.sys_max=sys_max
        self.group_lab_dict=group_lab_dict
        if self.group_lab_dict is not None:
            self.group_lab_lookup = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list(self.group_lab_dict.keys()), list(self.group_lab_dict.values())), default_value=-1)
        self.fine_tune=fine_tune
        self.c_0=c_0
        self.c_1=c_1
        self.fname = fname # name model
        #self.fname_user=fname_user
        self.curves_folder=curves_folder 
        self.sigma_curves = sigma_curves
        self.sigma_curves_default = sigma_curves_default
        self.save_processed_spectra = save_processed_spectra
        self.rescale_curves = rescale_curves
        self.seed = seed
        self.rng = tf.random.Generator.from_seed(self.seed)
        self.swap_axes = swap_axes 
        self.TPU = TPU
        self.k_max=k_max
        self.k_min=k_min
        self.i_max=i_max
        self.i_min=i_min
        self.sample_pace=sample_pace
        if sample_pace ==1:
          self.dim = dim
        else:
          self.dim = (int(dim[0]/sample_pace), dim[1]) 
        self.n_channels = n_channels
        self.z_bins=np.asarray(z_bins, dtype=int)
        
        print('Using z bins %s' %z_bins)
        if not self.swap_axes:
            if self.z_bins.shape[0]!=self.dim[1]:
                raise ValueError('Number of z bins does not match dimension 1 of the data.')
        else:
            if self.z_bins.shape[0]!=self.n_channels:
                raise ValueError('Number of z bins does not match n_channels.')
        
        
        self.data_root=data_root
        self.norm_data_path = self.data_root+norm_data_name
        print('Normalisation file is %s' %norm_data_name)

        self.all_ks = np.loadtxt(self.norm_data_path)[:, 0]
        self.original_k_length = self.all_ks.shape[0]
        if self.sample_pace !=1:
                self.all_ks = np.loadtxt(self.norm_data_path)[0::sample_pace, 0]
        self.k_range = self.all_ks

        # Select points from k_max or i_max

        if self.k_max is not None:
            print('Specified k_max is %s' %self.k_max)
            self.i_max, k_max_res = find_nearest(self.all_ks, self.k_max) 
            print('Corresponding i_max is %s' %self.i_max)
            print('Closest k to k_max is %s' %k_max_res)

        elif self.i_max is not None:
            self.k_max = self.all_ks[i_max]
            print('Specified i_max is %s' %self.i_max)
            print('Corresponding k_max is %s' %self.k_max)
            
        elif self.i_max is not None and self.k_max is not None:
            print('Specified i_max is %s' %self.i_max)
            print('Specified k_max is %s' %self.k_max)
            
            i_max, k_max = find_nearest(self.all_ks, self.k_max)
            assert(i_max==self.i_max)

        else:
            self.i_max = -1
            print('No max in k. Using all ks . k_max=%s' %self.all_ks[i_max])

        # Select points from k_min or i_min

        if self.k_min is not None:
            print('Specified k_min is %s' %self.k_min)
            self.i_min, k_min_res = find_nearest(self.all_ks, self.k_min) 
            print('Corresponding i_min is %s' %self.i_min)
            print('Closest k to k_min is %s' %k_min_res)

        elif self.i_min is not None:
            self.k_min = self.all_ks[i_min]
            print('Specified i_min is %s' %self.i_min)
            print('Corresponding k_min is %s' %self.k_min)
            
        elif self.i_min is not None and self.k_min is not None:
            print('Specified i_min is %s' %self.i_min)
            print('Specified k_min is %s' %self.k_min)
            
            i_min, k_min = find_nearest(self.all_ks, self.k_min)
            assert(i_min==self.i_min)

        else:
            self.i_min = 0
            print('No min in k. Using all ks . k_min=%s' %self.all_ks[i_min])

        self.all_ks = self.all_ks[self.i_min:self.i_max]
        self.dim = (self.all_ks.shape[0], self.dim[1])
        print('New data dim: %s' %str(self.dim) )
        print('Final i_max used is %s' %self.i_max)
        print('Final i_min used is %s' %self.i_min)

            
        
            
        self.batch_size = batch_size
        
        self.labels = labels
        #print(self.labels)
        self.labels_dict = labels_dict
        self.labels_lookup = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list(self.labels_dict.keys()), list(self.labels_dict.values())), default_value=-1)
        self.inv_labels_dict={value:key for key,value in zip(self.labels_dict.keys(), self.labels_dict.values())}
        #print(self.inv_labels_dict)
        self.list_IDs = list_IDs
        if len(self.list_IDs)==1:
            self.list_IDs_dict = {label:list_IDs+i for i,label in enumerate(labels)}
            print('Ids dict to use in data gen: %s' %str(self.list_IDs_dict))
        else:
            self.list_IDs_dict = {label:list_IDs for label in labels}
        
        
        self.base_case_dataset = not((self.fine_tune and self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and self.dataset_balanced))
        print('one_vs_all: %s' %str(self.one_vs_all))
        print('dataset_balanced: %s' %str(self.dataset_balanced))
        print('base_case_dataset: %s' %str(self.base_case_dataset))
        
        
        self.n_classes_out = len(self.labels)
        if not self.base_case_dataset:
            self.n_classes = 2*(len(self.c_1))
        elif (self.fine_tune and not self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and not self.dataset_balanced):
            self.n_classes = len(self.c_1)+len(self.c_0)
        else:
            # regular 5 labels case
            self.n_classes =len(self.labels)
        
        
        print('N. classes: %s' %self.n_classes) 
        print('N. n_classes in output: %s' %self.n_classes_out) #number of labels to predict
        print('LABELS:', self.labels)
            
        self.shuffle = shuffle
        #print('Batch size: %s' %self.batch_size)
        #print('N. samples used for each different label: %s' %self.n_indexes)
        self.save_indexes = save_indexes
        if self.save_indexes:
            self.save_indexes_dict={}
        self.normalization=normalization
        
        self.norm_data = np.loadtxt(self.norm_data_path)[:, 1:]
        if self.sample_pace !=1:
            self.norm_data = self.norm_data[0::self.sample_pace, :]
        self.norm_data = self.norm_data[self.i_min:self.i_max]
              
        
        self.idx_file_name = idx_file_name
        self.models_dir = models_dir
        self.pad=pad
        self.add_noise=add_noise
        if not self.add_noise:
          self.n_noisy_samples = 1
        else:
          self.n_noisy_samples = n_noisy_samples
        
        
        ######
        # Consistency checks
        ######
        
        if not self.base_case_dataset:
            if self.batch_size%(self.n_classes*self.n_noisy_samples):
                print('batch_size,n_classes, len(c_1), n_noisy_samples= %s, %s, %s, %s '%(self.batch_size, self.n_classes, len(self.c_1), self.n_noisy_samples))
                raise ValueError('batch size must be multiple of n_classes x len(c_1) x n_noisy_samples')
        elif not(self.fine_tune and self.dataset_balanced) or not(not self.fine_tune and self.one_vs_all and self.dataset_balanced):
            if self.batch_size%(self.n_classes*self.n_noisy_samples):
                raise ValueError('batch size must be multiple of n_classes x n_noisy_samples')
        else:
            raise ValueError('check dataset_balanced and one_vs_all compatibility')
            
        if not self.base_case_dataset:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                print('Batch size = %s' %self.batch_size)
                #print('( n_labels x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
                raise ValueError('Batch size must be multiple of n_classes x len(c_1)  x (n_noisy_samples) ')
            self.n_indexes = len(self.c_1)*self.batch_size//(self.n_classes*self.n_noisy_samples) #len(self.c_1)*
            print('batch_size, n_classes, len(self.c_1), n_noisy_samples= %s, %s, %s, %s' %(self.batch_size, self.n_classes, len(self.c_1), self.n_noisy_samples))
            print('n_indexes=len(self.c_1)*batch_size//(n_classes*n_noisy_samples)=%s' %self.n_indexes)
             
        else:
            if self.batch_size%(self.n_classes*self.n_noisy_samples)!=0:
                print('Batch size = %s' %self.batch_size)
                print('( n_classes x n_noisy_samples) = %s' %(self.n_classes*self.n_noisy_samples))
                raise ValueError('Batch size must be multiple of (number of classes) x (n_noisy_samples) ')
            self.n_indexes = self.batch_size//(self.n_classes*self.n_noisy_samples) # now many index files to read per each batch
        
        self.n_batches = len(list_IDs)//(self.n_indexes)
        print('list_IDs length: %s' %len(list_IDs))
        print('n_indexes (n of file IDs read for each batch): %s' %self.n_indexes)
        print('batch size: %s' %self.batch_size)
        print('n_batches : %s' %self.n_batches)
        if self.n_batches==0:
            raise ValueError('Not enough examples to support this batch size ')
   
        print('For each batch we read %s file IDs' %self.n_indexes)
        if not self.fine_tune or not self.dataset_balanced:
            print('For each file ID we have %s labels' %(self.n_classes ))
        else:
            print('We read %s IDs for label %s and 1 ID for each of the labels %s' %(str(len(self.c_1)), c_0[0],str( c_1)) )
        if self.add_noise:
          print('For each ID, label we have %s realizations of noise' %self.n_noisy_samples)
         
        if self.base_case_dataset:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples
            n_check = self.n_classes*self.n_noisy_samples 
        else:
            n_ex = self.n_indexes*self.n_classes*self.n_noisy_samples/len(self.c_1)
            n_check = self.n_classes*self.n_noisy_samples#*len(self.c_1) # n_indexes must be a multiple of this x batch_size
        
        print('In total, for each batch we have %s training examples' %(n_ex))
        print('Input batch size: %s' %self.batch_size)
        print('N of batches to cover all file IDs: %s' %self.n_batches)
        if n_ex!=self.batch_size:
            raise ValueError('Effective batch size does not match input batch size')
        
        if self.n_indexes%(self.batch_size/(n_check))!=0:
          print('Batch size = %s' %self.batch_size)
          print('( n_labels x n_noisy_samples) = %s' %(n_check*self.n_noisy_samples))
          print('n_indexes = %s' %self.n_indexes)
          raise ValueError('Batch size should satisfy  m x Batch size /( n_labels x n_noisy_samples) =  n_indexes  with m a positive integer ')
        
        
        if self.n_indexes!=len(list_IDs)/self.n_batches: 
          print('length of IDs = %s' %str(len(list_IDs)))
          print('n_batches = %s' %self.n_batches)
          print('n_indexes = %s' %self.n_indexes)
          print('len(list_IDs)/self.n_batches = %s' %(len(list_IDs)/self.n_batches))
          raise ValueError('n_batches does not match length of IDs')
        self.Verbose=Verbose
        self.Verbose_2=Verbose_2

        self.dataset=self.create_dataset(self.list_IDs, self.list_IDs_dict)

        if self.save_processed_spectra:
            for X, y in self.dataset.take(1):
                self.save_spectra(X, y)

    @tf.function
    def noise_realisations(self, fname, P_original, k, i_file):
        X_realisations = tf.TensorArray(dtype=tf.float32, size=self.n_noisy_samples)
        y_realisations = tf.TensorArray(dtype=tf.int32, size=self.n_noisy_samples)
        def cond(i_noise):
            return i_noise < self.n_noisy_samples
        def body(i_noise):
            if self.add_noise:
                P_noisy = P_original
                if self.Verbose:
                    tf.print('Noise realization %s' %i_noise)
                # add noise if selected
                if self.add_cosvar:
                    noise_scale = generate_noise(k, self.P_noise, sys_scaled=self.sys_scaled,
                                                    sys_factor=self.sys_factor,sys_max=self.sys_max,
                                                    add_cosvar=True, add_sys=False, add_shot=False, sigma_sys=self.sigma_sys)
                    noise_cosvar = self.rng.normal(shape=noise_scale.shape, mean=0, stddev=noise_scale)
                    P_noisy = P_noisy + noise_cosvar
                if self.add_sys:
                    if self.add_sys:
                        curve_loaded = self.all_curves[i_file*self.n_noisy_samples+i_noise]
                        noise_sys, k_sys = curve_loaded[:, 1:], curve_loaded[:, 0]
                    if self.sample_pace!=1:
                        noise_sys = noise_sys[0::self.sample_pace, :]
                        k_sys = k_sys[0::self.sample_pace]
                    noise_sys, k_sys = noise_sys[self.i_min:self.i_max], k_sys[self.i_min:self.i_max]
                    # rescale noise_sys curves according to error (10% default from production curves), 
                    # rescale by Gaussian with sigma = 1
                    # multiply with normalisation spectrum
                    noise_sys = (noise_sys-1) * self.sigma_curves/self.sigma_curves_default  * self.norm_data[:,self.z_bins]
                    if self.rescale_curves == 'uniform':
                        noise_sys = noise_sys * self.rng.uniform(shape = (), minval=0, maxval=1)
                    if self.rescale_curves == 'gaussian':
                        noise_sys = noise_sys * self.rng.normal(shape = (), mean=0, stddev=1)
                    P_noisy = P_noisy + noise_sys
                if self.add_shot:
                    noise_scale = generate_noise(k,self.P_noise, sys_scaled=self.sys_scaled,sys_factor=self.sys_factor,sys_max=self.sys_max, add_cosvar=False, add_sys=False, add_shot=True,sigma_sys=self.sigma_sys)
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
                expanded = tf.gather(expanded, self.z_bins, axis=1)
                if self.Verbose:
                    tf.print('Dimension of data before normalising: %s' %str(expanded.shape))
                    tf.print('expanded first 10:') 
                    tf.print(expanded[10])
                # now shape of expanded is (n_data_points, 1, n_channels=3)
            X = expanded  
            if self.Verbose:
                tf.print('dimension of X: %s' %str(X.shape))
                tf.print('X first 10:') 
                tf.print(X[:10])
            label = fname.split('/')[-2]
                
            if not self.base_case_dataset:
                label = self.group_lab_lookup.lookup(label)
                encoding = self.labels_lookup.lookup(label)
            elif (self.fine_tune and not self.dataset_balanced) or (not self.fine_tune and self.one_vs_all and not self.dataset_balanced):
                label = self.group_lab_lookup.lookup(label)
                encoding = self.labels_lookup.lookup(label)
            else:
                # regular 5 labels case
                encoding = self.labels_lookup.lookup(label)
            if self.Verbose:
                tf.print('Label for this example: %s' %label)
                tf.print('Encoding: %s' % encoding)
            y = encoding
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            X_realisations = X_realisations.write(i_noise, X)
            y_realisations = y_realisations.write(i_noise, y)
            return i_noise + 1
        i_noise = tf.constant(0)
        i_noise = tf.while_loop(cond, body, [i_noise])
        X_realisations = X_realisations.stack()
        y_realisations = y_realisations.stack()
        return X_realisations, y_realisations
    
    @tf.function
    def normalize(self, X, y):
        if self.Verbose:
            tf.print("X shape into normalize: ", tf.shape(X))
        if self.normalization == 'batch':
            mu_batch = tf.reduce_mean(X, axis=0)
            std_batch = tf.math.reduce_std(X, axis=0)
            X = (X - mu_batch) / std_batch
        elif self.normalization == 'stdcosmo':
            if self.swap_axes:
                divisor = tf.expand_dims(tf.expand_dims(self.norm_data, axis=0), axis=2)
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
        if self.Verbose:
            tf.print("X shape before normalize slicing: ", tf.shape(X))
        if self.swap_axes:
            X = X[:,:,0,:]
        if self.Verbose:
            tf.print("X shape after normalize slicing: ", tf.shape(X))
        return X, y
    
    @tf.function
    def transformations(self, dataset):
        dataset = dataset.shuffle(buffer_size=self.batch_size*self.n_batches)
        dataset = dataset.map(self.noise_realisations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda X, y: tf.data.Dataset.from_tensor_slices((X, y)))
        dataset = dataset.map(lambda X, y: (X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes_out)))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def create_dataset(self, list_IDs, list_IDs_dict):
        'Generates a batched DataSet'
        if not self.fine_tune and not self.one_vs_all:
            fname_list=[]
            for l in self.labels:
                for ID in list_IDs_dict[l]:
                    t_st =  self.data_root + '/'+l+ '/'+ str(ID) + '.txt' 
                    fname_list.append(str(t_st))
        else:
             fname_list = get_fname_list(self.c_0, self.c_1, list_IDs, self.data_root,  list_IDs_dict, dataset_balanced=self.dataset_balanced,)
        if self.fine_tune and self.Verbose :
            print(fname_list)
            
        print('len(fname_list), batch_size, n_noisy_samples, n_batches: %s, %s, %s, %s' %(len(fname_list), self.batch_size, self.n_noisy_samples, self.n_batches))
        assert len(fname_list)==self.batch_size*self.n_batches // self.n_noisy_samples

        fname_list = np.array(fname_list, dtype=str)

        if self.Verbose_2:
            print("list_IDs_dict")
            print(list_IDs_dict)

        #Load spectrum files
        P_originals = []
        k_originals = []
        i_file_list = []
        for i, fname in enumerate(fname_list):
            if self.Verbose:
                print('Loading file %s' %fname)
            loaded = np.loadtxt(fname)
            P_original, k = loaded[:, 1:], loaded[:, 0]
            del loaded
            if self.sample_pace != 1:
                P_original = P_original[::self.sample_pace]
                k = k[::self.sample_pace]
            P_original, k = P_original[self.i_min:self.i_max], k[self.i_min:self.i_max]
            if i == 0:
                self.k_range = tf.convert_to_tensor(k, dtype=tf.float32)
            if self.Verbose:
                print('Dimension of P_original: %s' %str(P_original.shape))
                print('Dimension of k: %s' %str(k.shape))
            P_originals.append(P_original)
            k_originals.append(k)
            i_file_list.append(i)
    
        #Load all theory error curves
        self.all_curves = np.zeros((self.n_noisy_samples*self.n_batches*self.batch_size, self.original_k_length, self.n_channels + 1), dtype=np.float32)
        for i in range(self.n_noisy_samples*self.n_batches*self.batch_size):
            curve_random_nr = self.rng.uniform(shape=[], minval=1, maxval=1000, dtype=tf.int32)
            curve_random_nr = tf.strings.as_string(curve_random_nr).numpy().decode('utf-8')
            curve_path = os.path.join(self.curves_folder, '{}.txt'.format(curve_random_nr))
            self.all_curves[i] = np.loadtxt(curve_path)

        k_test = k_originals[0]
        k_sys_test = self.all_curves[0][:, 0]
        k_sys_test = k_sys_test[0::self.sample_pace]
        k_sys_test = k_sys_test[self.i_min:self.i_max]
        if (k_test.all() != k_sys_test.all()):
            print('ERROR: k-values in spectrum and theory-error curve file not identical')
            

        #Convert these to tensors for map operations
        self.P_noise = tf.convert_to_tensor(self.norm_data[:, self.z_bins], dtype=tf.float32)
        self.norm_data = tf.convert_to_tensor(self.norm_data, dtype=tf.float32)
        self.all_curves = tf.convert_to_tensor(self.all_curves, dtype=tf.float32)
        P_originals = tf.convert_to_tensor(P_originals, dtype=tf.float32)
        k_originals = tf.convert_to_tensor(k_originals, dtype=tf.float32)
        i_file_list = tf.convert_to_tensor(i_file_list, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((fname_list, P_originals, k_originals, i_file_list))
        del fname_list, P_originals, k_originals, i_file_list

        if self.TPU:
            with self.strategy.scope():
                self.transformations(dataset)
                
        else:
            dataset = self.transformations(dataset)
        for x, y in dataset.take(1):
            self.xshape = x.shape
            self.yshape = y.shape
            
        return dataset
    
    def save_spectra(self, X, y):
        with tf.device('/CPU:0'):
            for batch in self.dataset.take(1):
                X, y = batch
                y = tf.argmax(y, axis=-1)
                name_spectra_folder = os.path.join(self.models_dir,self.fname,'processed_spectra') 
                if not os.path.exists(name_spectra_folder):
                    print('Creating directory %s' %  name_spectra_folder)
                    os.makedirs(name_spectra_folder)
                # new matrix for spectra, first column is class_idx, first row is k-values
                X_save = np.empty((self.batch_size+1, len(self.all_ks)+1))
                X_save[1:,0] = y  
                X_save[0,1:] = self.all_ks
                for i_z in self.z_bins:
                    X_save[1:,1:] = X[:,:,i_z]
                    spectra_file = os.path.join(name_spectra_folder, 'processed_spectra_zbin{}.txt'.format(i_z))
                    if not os.path.exists(spectra_file):
                        print('Saving processed (noisy and normalised) spectra in %s' % spectra_file)
                        with open(spectra_file, "a+") as myCurvefile:
                            np.savetxt(myCurvefile, X_save, delimiter=' ', newline='\r\n')




def read_partition(FLAGS):
    out_path = FLAGS.models_dir+FLAGS.fname
    base_path = out_path+'/tf_ckpts/'
    fname_idxs_train=base_path+'idxs_train.txt'
    fname_idxs_val=base_path+'idxs_val.txt'
    
    print('Reading train indexes from %s ...' %fname_idxs_train)
    train_idxs=np.array(np.loadtxt(fname_idxs_train).tolist()).astype(int)
    print('Train indexes length: %s' %str(len(train_idxs)))
    print('Reading val indexes from %s ...' %fname_idxs_val)
    val_idxs=np.array(np.loadtxt(fname_idxs_val).tolist()).astype(int)
    print('Val indexes length: %s' %str(len(val_idxs)))
    
    partition = {'train': train_idxs , 'validation': val_idxs }
    return partition
    
    

def create_datasets(FLAGS, strategy=None):
    
    if FLAGS.my_path is not None:
        os.chdir(FLAGS.my_path)
       
    
    # --------------------  CREATE DATA GENERATORS   --------------------
    
        
    if FLAGS.my_path is not None:
        os.chdir(FLAGS.my_path)
       
    
    # --------------------  CREATE DATA GENERATORS   --------------------
    
    all_index, n_samples, val_size, n_labels, labels, labels_dict, all_labels = get_all_indexes(FLAGS)
    print('create_generators n_labels: %s' %n_labels) 
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
    print('create_generators n_labels_eff: %s' %n_labels_eff)  
    print('create_generators len_c1: %s' %len_c1)
        
        

    
    # SPLIT TRAIN/VALIDATION /(TEST)
    val_index = np.random.choice(all_index, size=int(np.floor(val_size*n_samples)), replace=False)
    train_index_temp =  np.setdiff1d(all_index, val_index) #np.delete(all_index, val_index-1)
    test_size_eff = FLAGS.test_size/(train_index_temp.shape[0]/n_samples)
    test_index = np.random.choice(train_index_temp, size=int(np.floor(test_size_eff*train_index_temp.shape[0])), replace=False)
    train_index =  np.setdiff1d(train_index_temp, test_index)

    print('Check for no duplicates in test: (0=ok):')
    print(np.array([np.isin(el, train_index) for el in test_index]).sum())
    print('Check for no duplicates in val: (0=ok):')
    print(np.array([np.isin(el, train_index) for el in val_index]).sum())

    print('N of files in training set: %s' %n_labels_eff*train_index.shape[0])
    print('N of files in validation set: %s' %n_labels_eff*val_index.shape[0])
    print('N of files in test set: %s' %n_labels_eff*test_index.shape[0])

    print('Check - total per class: %s' %(val_index.shape[0]+test_index.shape[0]+train_index.shape[0]))
    
    if FLAGS.add_noise:
        n_noisy_samples = FLAGS.n_noisy_samples
    else:
        n_noisy_samples = 1
    print('--create_generators, train indexes')
    if FLAGS.test_mode:
        if case==3:
            batch_size=train_index.shape[0]*n_labels_eff*n_noisy_samples
        elif case==2:
            batch_size=train_index.shape[0]*n_labels*n_noisy_samples
        elif case==1:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    print('batch_size: %s' %batch_size)

    if not FLAGS.test_mode:
        train_index_1  = cut_sample(train_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False, len_c1=len_c1)
        print('Train index length: %s' %train_index_1.shape[0])
    else:
        train_index_1 = train_index
        print('Train index: %s' %train_index_1)
    print('--create_generators, validation indexes')
    if not FLAGS.test_mode:
        val_index_1  = cut_sample(val_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=False,len_c1=len_c1)
        print('Val index length: %s'  %val_index_1.shape[0])
    else:
        val_index_1 = val_index
        print('Validation index: %s' %val_index_1)
    
    print('len(train_index_1), batch_size, n_labels_eff, n_noisy_samples = %s, %s, %s, %s' %(train_index_1.shape[0], batch_size, n_labels_eff,n_noisy_samples ))
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
        print(' ####  FLAGS.sigma_sys not found! #### \n Probably loading an older model. Using sigma_sys=0')
        sigma_sys=0.
        
    try:
        z_bins=FLAGS.z_bins
    except AttributeError:
        print(' ####  FLAGS.z_bins not found! #### \n Probably loading an older model. Using 4 z bins')
        z_bins=[0, 1, 2, 3]
    try:
        swap_axes=FLAGS.swap_axes
    except AttributeError:
        if FLAGS.im_channels>1:
            swap_axes=True
        else:
            swap_axes=False
        print(' ####  FLAGS.swap_axes not found! #### \n Probably loading an older model. Set swap_axes=%s' %str(swap_axes))
    ###################
        
    ##Generate a random seed now to prevent issues in parallel processing when in TPU mode
    seed = np.random.randint(0, 2**32 - 1)
    
    params = {'dim': (FLAGS.im_depth, FLAGS.im_width),
          'batch_size':batch_size, # should satisfy  m x Batch size /n_noisy_samples =  n_indexes  with m a positive integer
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
    
    print('\n--DataSet Train')
    training_dataset = DataSet(partition['train'], labels, labels_dict, data_root = FLAGS.DIR, save_indexes=False, seed = seed, strategy=strategy, **params)
    print('\n--DataSet Validation')
    validation_dataset = DataSet(partition['validation'], labels, labels_dict, data_root = FLAGS.DIR,  save_indexes=False,strategy=strategy, seed = seed, **params)

    
    return training_dataset, validation_dataset #, params




def create_test_dataset(FLAGS):
    
     
    if FLAGS.my_path is not None:
        print('Changing directory to %s' %FLAGS.my_path)
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
    print('create_generators n_labels_eff: %s' %n_labels_eff)  
    print('create_generators len_c1: %s' %len_c1)

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
    print('--Train')
    if FLAGS.test_mode:
        if not FLAGS.fine_tune:
            batch_size=all_index.shape[0]*n_labels_eff*n_noisy_samples
        else:
            batch_size=n_labels_eff*n_noisy_samples
    else:
        batch_size=FLAGS.batch_size
    print('batch_size: %s' %batch_size)
       
    test_index_1  = cut_sample(all_index, batch_size, n_labels=n_labels_eff, n_noise=n_noisy_samples, Verbose=True, len_c1=len_c1)
    n_test = test_index_1.shape[0]

    #assert test_index_1.shape[0]%(batch_size//(n_labels_eff*n_noisy_samples))==0

    print('N. of test files used: %s' %n_test)

    partition_test = {'test': test_index_1}
    
    
 
    
    params_test = {'dim': (FLAGS.im_depth, FLAGS.im_width),
          'batch_size':batch_size, # should satisfy  m x Batch size /(n_noisy_samples) =  n_indexes  with m a positive integer
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
    
    test_dataset = DataSet(partition_test['test'], 
                                   labels, labels_dict, 
                                   data_root=FLAGS.TEST_DIR , 
                               save_indexes = FLAGS.save_indexes,
                               models_dir=FLAGS.models_dir+FLAGS.fname,
                               idx_file_name = FLAGS.fname, seed = seed,
                               **params_test)


    
    return test_dataset

