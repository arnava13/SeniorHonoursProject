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
import threading

def handle_output(stream, log_file):
    for line in iter(stream.readline, b''):  # This iterates over the output line by line
        my_bytes = line.strip()
        if my_bytes:
            output = f"{my_bytes.decode('utf-8')}"
            print(output)
            log_file.write(output + "\n")

def main():

   # Training options:

# ------------------ change the training directory and model name here ------------------

   # training data directory path
   DIR='data/fivelabel_train'
   ## (batch size, is a multiple of #classes * noise realisations, e.g. 5 classes, 10 noise --> must be multiple of 50)
   batch_size='10000'

   # training the network
   GPU='False' # Use GPUs in training?
   TPU='True' # Train with TPUs?
   # which kind of model
   bayesian='True' # Bayesian NN or traditional NN (weights single valued or distributions?)
   n_epochs='50' # How many epochs to train for?


   # ------------------ other parameters ------------------

   # Directories
   mypath=None # Parent directory

   # Edit this with base model directory
   mdir='models/'

   # normalisation file
   norm_data_name = '/planck_ee2.txt'

   # scale cut and resolution
   k_max='2.5' # max of k-modes to include?
   k_min='0.0' # min of k-modes to include?
   sample_pace='2'

   # should the processed curves be saved in an extra file (False recommended, only efficient for n_epochs = 1)
   save_processed_spectra='False'

   # fine-tune = only 'LCDM' and 'non-LCDM'
   fine_tune='False' # Model trained to distinguish between 'LCDM' and 'non-LCDM' classes or between all clases. i.e. binary or multi-classifier

   # -------------------- noise model --------------------

   # noise model - Cosmic variance and shot noise are Euclid-like
   n_noisy_samples='10' # How many noise examples to train on?
   add_noise='True'# Add noise?
   add_shot='False'# Add shot noise term? -- default should be False
   add_sys='True'# Add systematic error term?
   add_cosvar='True'# Add cosmic variance term?

   # path to folder with theory error curves
   curves_folder = 'data/curve_files_sys/theory_error'; sigma_curves_default = '0.05'
   # change rescale distribution of theory error curves, uniform recommended
   rescale_curves = 'uniform' # or gaussian or None

   # sys error curves - relative scale of curve-amplitude
   sigma_curves='0.05'

   # ----------------- Model Name Generation -------------

   #Two or Six label?
   """
   if fine_tune != 'True':
   two_or_six = 'sixlabel'
   else:
   two_or_six = 'twolabel'
   """

   two_or_six = 'fivelabel'

   #Examples Per Class?
   examples_per_class = 10000

   #Generation code?
   gen_code = "EE2"

   #LCDM * Filter or Equal Examples from each class * Filter for randoms?
   #randstype = "equalexamples-randoms"
   randstype = "LCDM-randoms"

   #Additional Notes?
   fname_notes = ""

   #Assemble fname
   fname = f"{two_or_six}_{examples_per_class}_{gen_code}_{randstype}_kmin-{k_min}_kmax-{k_max}_{fname_notes}"

   # -------------------- additional training settings --------------------


   # Type of model
   model_name='custom' # Custom or dummy - dummy has presets as given in models.py

   #Test mode?
   test_mode='False' # Network trained on 1 batch of minimal size or not?
   seed='1312' # Initial seed for test mode batch
   # Saving or restoring training
   restore='False' # Restore training from a checkpoint?
   save_ckpt='True' # Save checkpoints?


   val_size='0.15' # Validation set % of data set

   add_FT_dense='False' #if True, adds an additional dense layer before the final 2D one

   patience='20' # terminate training after 'patience' epochs if no decrease in loss function
   lr='0.01' # learning rate
   decay='0.90' #decay rate: If None : Adam(lr),

   padding='valid'

   # -------------------- BNN parameters -----------------

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

   log_path = mdir + fname + '_log'

   if fine_tune == "True":
      log_path += '_FT'

   log_path += '.txt'

   # -------------- adapt c_0 and c_1 =  'fR', 'dgp', 'ds', 'wcdm', 'rand' <<<<<<<<<<<<<<<

   # -------------- training loads parameters entered above  --------------
   proc = subprocess.Popen(["python3", "train.py", "--test_mode" , test_mode, "--seed", seed, \
                        "--bayesian", bayesian, "--model_name", model_name, \
                        "--fine_tune", fine_tune, "--log_path", log_path,\
                        "--restore", restore, \
                        "--models_dir", mdir, \
                        "--fname", fname, \
                        "--DIR", DIR, \
                        '--norm_data_name', norm_data_name, \
                        '--curves_folder', curves_folder,\
                        "--c_0", 'lcdm', \
                        "--c_1", 'fR', 'dgp', 'wcdm', 'ds', 'rand'\
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
                        "--k_min", k_min,\
                        "--n_dense", n_dense,\
                        "--add_FT_dense", add_FT_dense, \
                        "--n_epochs", n_epochs, "--patience", patience, "--batch_size", batch_size, "--lr", lr, \
                        "--decay", decay, \
                        "--GPU", GPU, \
                        "--TPU", TPU, \
                        "--padding", padding],\
                        stdout=subprocess.PIPE, \
                        stderr=subprocess.PIPE)

   with open(log_path, "w") as log_file:

      stdout_thread = threading.Thread(target=handle_output, args=(proc.stdout, log_file))
      stderr_thread = threading.Thread(target=handle_output, args=(proc.stderr, log_file))

      stdout_thread.start()
      stderr_thread.start()

      stdout_thread.join()
      stderr_thread.join()

      proc.wait()


if __name__=='__main__':
    
    main()
