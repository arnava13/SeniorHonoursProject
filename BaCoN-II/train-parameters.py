#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My training file

@author: Ben Bose - 4.11.2022
@editor: Linus Thummel - 31.01.2023
"""

# Training options:
import subprocess
import argparse

def main():

   # -------------------- always check these parameters --------------------

   # path to training data 
   DIR='data/ee2/train-4classes'; train_fname = 'EE2_4class_train20k'

   # path to normalisation file
   norm_data_name = '/planck_ee2.txt'; planck_fname = 'ee2' # norm name for model name

   # path to folder with theory error curves
   curves_folder = 'data/theory_error/filters_earliest_onset'; sigma_curves_default = '0.05'
   # change rescale distribution of theory error curves, uniform recommended 
   rescale_curves = 'uniform' # or gaussian or None

   # scale cut and resolution
   k_max='2.5' # which k-modes to include?
   sample_pace='4'

   # should the processed curves be saved in an extra file (False recommended, only efficient for n_epochs = 1)
   save_processed_spectra='False' 

   # noise model - Cosmic variance and shot noise are Euclid-like
   n_noisy_samples='10' # How many noise examples to train on?
   add_noise='True'# Add noise?
   add_shot='False'# Add shot noise term? -- default should be False
   add_sys='True'# Add systematic error term?
   add_cosvar='True'# Add cosmic variance term?

   # batch size is a multiple of number of classes * noise realisations, e.g. 4 classes, 10 noise --> must be multiple of 40
   batch_size='8000' # 4 models, 10 noise samples for 20k-data

   # final part of the model name, to quickly find model in folder
   fname_extra = 'EARLIEST-uniform'


   # -------------------- following parameters can be changed --------------------
   # ----------------- but we recommend to keep the values as set ----------------

   # sys error curves - relative scale of curve-amplitude 
   sigma_curves='0.01'

   # read sigma_curves as a parameter from the command line
   parser = argparse.ArgumentParser()
   parser.add_argument("--sigma_curves", default=None, type=str, required=False)
   args = parser.parse_args()
   if args.sigma_curves is not None:
      sigma_curves = args.sigma_curves

   # Type of model
   bayesian='True' # Bayesian NN or traditional NN (weights single valued or distributions?)
   model_name='custom' # Custom or dummy - dummy has presets as given in models.py

   # fine-tune = only 'LCDM' and 'non-LCDM' 
   fine_tune='False' # Model trained to distinguish between 'LCDM' and 'non-LCDM' classes or between all clases. i.e. binary or multi-classifier

   #Test mode?
   test_mode='False' # Network trained on 1 batch of minimal size or not?
   seed='1312' # Initial seed for test mode batch
   # Saving or restoring training
   restore='False' # Restore training from a checkpoint?
   save_ckpt='True' # Save checkpoints?

   # Directories
   mypath=None # Parent directory
   # Edit this with base model directory
   mdir='models/'

   GPU='False' # Use GPUs?
   val_size='0.15' # Validation set % of data set

   add_FT_dense='False' #if True, adds an additional dense layer before the final 2D one
   n_epochs='50' # How many epochs to train for?
   patience='50' # terminate training after 'patience' epochs if no decrease in loss function
   lr='0.01' # learning rate
   decay='0.95' #decay rate: If None : Adam(lr), 



   # ---------------- model name configured automatically -----------------

   fname = 'filters_' + train_fname + '_samplePace' + sample_pace + '_kmax' + k_max + '_planck-' + planck_fname + '_epoch' + n_epochs + '_batchsize' + batch_size

   if add_noise=='True':
      fname = fname + '_noiseSamples' + n_noisy_samples
      if add_cosvar=='True':
         fname = fname + '_wCV'
      else:
         fname = fname + '_noCV'
      if add_shot=='True':
         fname = fname + '_wShot'
      else:
         fname = fname + '_noShot'
      if add_sys=='True':
         fname = fname + '_wSys' 
         fname = fname + '_rescale' + rescale_curves
      else:
         fname = fname + '_noSys'
   else:
      fname = fname + '_noNoise'

   if GPU=='True':
      fname = fname + '_GPU'

   if add_noise=='True':
      if add_sys=='True':
         fname = fname + '_scaledNorm' + sigma_curves
      
   fname = fname + '_' + fname_extra


   fname = fname.replace('.', 'o')

   # overwrite automatic name and set model name manually
   # fname = 'my_model_name' 


   # -------------------- BNN parameters -----------------
   # ------------------- no change needed ----------------

   # Example image details
   im_depth='500' # Number in z direction (e.g. 500 wave modes for P(k) examples)
   im_width='1' # Number in y direction (1 is a 2D image : P(k,mu))
   im_channels='4'  # Number in x direction (e.g. 4 redshifts for P(k,z))
   swap_axes='True' # Do we swap depth and width axes? True if we have 1D image in 4 channels
   sort_labels='True' # Sorts labels in ascending/alphabetical order

   z1='0' # which z-bins to include? Assumes each channel is a new z bin.
   z2='1' # which z-bins to include? Assumes each channel is a new z bin.
   z3='2' # which z-bins to include? Assumes each channel is a new z bin.
   z4='3' # which z-bins to include? Assumes each channel is a new z bin.

   # Number of layers and kernel sizes
   k1='10'
   k2='5'
   k3='2'
    # The dimensionality of the output space (i.e. the number of filters in the convolution)
   f1='8'
   f2='16'
   f3='32'
    # Stride of each layer's kernel
   s1='2'
   s2='2'
   s3='1'
   # Pooling layer sizes
   p1='2'
   p2='2'
   p3='0'
   # Strides in Pooling layer
   sp1='2'
   sp2='1'
   sp3='0'

   n_dense='1' # Number of dense layers

   # labels of different cosmologies
   #c0_label = 'lcdm'
   #c1_label = 'fR dgp wcdm'

   log_path = mdir+fname+'_log'
   if fine_tune:
     log_path_original= log_path
     log_path += '_FT'
     log_path_original += '.txt'
   else:
      log_path_original=''

   log_path += '.txt'

   # -------------- adapt c_0 and c_1 =  'fR', 'dgp', 'ds', 'wcdm', 'rand' <<<<<<<<<<<<<<<

   proc = subprocess.Popen(["python3", "train.py", "--test_mode" , test_mode, "--seed", seed, \
                     "--bayesian", bayesian, "--model_name", model_name, \
                     "--fine_tune", fine_tune, "--log_path", log_path_original,\
                     "--restore", restore, \
                     "--models_dir", mdir, \
                     "--fname", fname, \
                     "--DIR", DIR, \
                     '--norm_data_name', norm_data_name, \
                     '--curves_folder', curves_folder,\
                     "--c_0", 'lcdm', \
                     "--c_1", 'fR', 'dgp', 'wcdm', \
                     "--save_ckpt", save_ckpt, \
                     "--im_depth", im_depth, "--im_width", im_width, "--im_channels", im_channels, \
                     "--swap_axes", swap_axes, \
                     "--sort_labels", sort_labels, \
                     "--add_noise", add_noise, "--add_shot", add_shot, "--add_sys", add_sys,"--add_cosvar", add_cosvar, \
                     "--sigma_curves", sigma_curves, \
                     "--sigma_curves_default", sigma_curves_default, \
                     "--save_processed_spectra", save_processed_spectra, \
                     "--sample_pace", sample_pace,\
                     "--n_noisy_samples", n_noisy_samples, \
                     "--rescale_curves", rescale_curves, \
                     "--val_size", val_size, \
                     "--z_bins", z1,z2,z3,z4, \
                     "--filters", f1,f2,f3, "--kernel_sizes", k1,k2,k3, "--strides", s1,s2,s3, "--pool_sizes", p1,p2,p3, "--strides_pooling", sp1,sp2,sp3, \
                     "--k_max", k_max,\
                     "--n_dense", n_dense,\
                     "--add_FT_dense", add_FT_dense, \
                     "--n_epochs", n_epochs, "--patience", patience, "--batch_size", batch_size, "--lr", lr, \
                     "--decay", decay, \
                     "--GPU", GPU],\
                     stdout=subprocess.PIPE, \
                     stderr=subprocess.PIPE)

   with open(log_path, "w") as log_file:
     while proc.poll() is None:
        line = proc.stderr.readline()
        if line:
           my_bytes=line.strip()
           print ("err: " + my_bytes.decode('utf-8'))
           log_file.write(line.decode('utf-8'))
        line = proc.stdout.readline()
        if line:
           my_bytes=line.strip()
           print ("out: " + my_bytes.decode('utf-8'))
           log_file.write(line.decode('utf-8'))


if __name__=='__main__':
    
    main()
