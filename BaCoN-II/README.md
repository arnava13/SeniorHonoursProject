# BaCoN-II
This is the new version of [BaCoN](https://github.com/Mik3M4n/BaCoN) with an improved noise model for the theoretical error. We're now using a variety of smooth curves that approximate the error in the theoretical modelling on smaller scales. 

**The training of a model can now take up to 4 days.** (for 20,000 training spectra with 10 noise realisations each)

## Run on Goolge Colab
For a quick test of this code, we recommend to clone this github repo to a personal google drive and then run the jupyter notebook ```notebooks/training_colab.ipynb``` in google colab. (Use GPU runtime. For that go to the arrow at the upper right corner and then select 'Change runtime type' -> GPU). The training and testing of a model can be tested with the small data sets for training (100 spectra for class) and testing (10 or 1k spectra per class) that are included in the data folder in this repository. This can be used to check that the code is running. For the training of full models we recommend to use the full training data available here (with 20,000 spectra per class).

The architecture of the CNN model is set in the ```models.py``` file. It has the following parameters

```
def make_custom_model(    drop=0.5, 
                          n_labels=5, 
                          input_shape=(100, 4), 
                          padding='valid', 
                          filters=(8, 16, 32),
                          kernel_sizes=(10,5,2),
                          strides=(2,2,1),
                          pool_sizes=(2, 2, 0),
                          strides_pooling=(2, 1, 0),
                          activation=tf.nn.leaky_relu,
                          bayesian=True, 
                          n_dense=1, swap_axes=True, BatchNorm=True
                          ):
```
The architecture of the CNN is shown here:

<img src="https://github.com/cosmicLinux/BaCoN-II/assets/142009018/945bc75f-3b99-40a5-9f8c-01c6de025485"
     alt = "BaCoN architecture"
     width="700" />
     
## BaCoN (BAyesian COsmological Network)

BaCoN allows to train and test Bayesian Convolutional Neural Networks in order to **classify matter power spectra as being representative of different cosmologies**, as well as to compute the classification confidence. 
The code now supports the following theories:  **LCDM, wCDM, f(R), DGP, and a randomly generated class** (see the reference for details).

**We also provide a jupyter notebook that allows to train a model and test the classification on a test data set or on a single spectrum.**

The first release of BaCoN was accompagnied by the paper [Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity](https://arxiv.org/abs/2012.03992). 

Bibtex:

```
@misc{mancarella2020seeking,
      title={Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity}, 
      author={Michele Mancarella and Joe Kennedy and Benjamin Bose and Lucas Lombriser},
      year={2020},
      eprint={2012.03992},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO}
}
```

The training and testing data was generated  by [Ben Bose](https://github.com/nebblu) with [ReACT](https://github.com/nebblu/ReACT).
We use synthetic data of matter power spectra inluding the physical effects of massive neutrinos and baryonic effects. The preocess of data generation is published in [On the road to per cent accuracy – V. The non-linear power spectrum beyond ΛCDM with massive neutrinos and baryonic feedback](https://academic.oup.com/mnras/article/508/2/2479/6374568)

Bibtex:

```
@article{bose2021road,
  title={On the road to per cent accuracy--V. The non-linear power spectrum beyond $\Lambda$CDM with massive neutrinos and baryonic feedback},
  author={Bose, Benjamin and Wright, Bill S and Cataneo, Matteo and Pourtsidou, Alkistis and Giocoli, Carlo and Lombriser, Lucas and McCarthy, Ian G and Baldi, Marco and Pfeifer, Simon and Xia, Qianli},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={508},
  number={2},
  pages={2479--2491},
  year={2021},
  publisher={Oxford University Press}
}
```

## Overview and code organisation


The package provides the following modules:

* ```data generator.py```: data generator that generates batches of data. Data are dark matter power spectra normalised to the Planck LCDM cosmology, in the redshift bins (0.1,0.478,0.783,1.5) and k in the range 0.01-2.5 h Mpc^-1.
* ```models.py``` : contains models' architecture
* ```train.py```: module to train and fine-tune models. Checkpoints are automatically saved each time the validation loss decreases. Both bayesian and "traditional" NNs are available.
* ```test.py```: evaluates the accuracy and the confusion matrix.

A jupyter notebook to classify power spectra with pre-trained weights and computing the confidence in classification is available in ```notebooks/```. 

The first base model is a five-label classifier with LCDM, wCDM, f(R), DGP, and "random" as classes, while the second is a two-label classifier with classes LCDM and non-LCDM.

Details on training, data preparation, variations ot the base model, and extensions are available in the dedicated sections. The Bayesian implementation uses [Tensorflow probability](https://www.tensorflow.org/probability) with [Convolutional](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DFlipout) and [DenseFlipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) methods.




## New noise model

We have produced 1000 curves with random fluctations that are saved in the folder ```/data/theory_error/filters_earliest_onset/```. (With names ```1.txt``` to ```1000.txt```) The theory error curves become relevant from about $k = 0.03 \ \mathrm{h/Mpc}$ and have different shapes. They are scaled to a peak amplitude that can be set with the ```sigma_curves```-parameter in the ```train-parameters.py``` file. We recommend $\sigma_\mathrm{curves} = 0.05$ for EE2 data with $k_\mathrm{max} = 2.5 \ \mathrm{h/Mpc}$. 
Some example curves with $\sigma_\mathrm{curves} = 0.10$ are shown below.

<img src="https://github.com/cosmicLinux/BaCoN-II/assets/142009018/7ecece0e-876d-4e15-baf7-2128f5e7db65"
     width="300" />

A plot of 10 different realisations of theory error curves (scaled to 5%) superimposed onto some LCDM example spectra (including cosmic variance) is shown below.
     
<img src="https://github.com/cosmicLinux/BaCoN-II/assets/142009018/1c945ac0-67dc-45e7-8a4a-d28e417e3719"
     width="350" />

The amplitude of the curves can be rescaled with a uniform distribution to obtain some smaller errors as well. This option can be selected with the training parameter ```rescale_curves = 'uniform'```.

<img src="https://github.com/cosmicLinux/BaCoN-II/assets/142009018/a89b264c-d848-4ac6-805d-e5ad050c6c8c"
     width="300" />

The Cosmic Variance is still modelled as Gaussian noise on large scales. Shot noise is also a Gaussian noise, though we recommend to leave it out.



## Data structure
Get the spectra from here and put them into the data folder. The random spectra included in the data folders here are the updated random class from the second paper. Copy the normalisation file ```planck_ee2.txt``` from the ```data/normalisation/``` folder into the train and test data folders. The resulting data structure should look like this:

```bash
data/train_data/
		├── dgp/
			├──1.txt
			├──2.txt
			├──...
		├── fR/
			├──1.txt
			├──2.txt
			├──...
		├── lcdm/
			├──1.txt
			├──2.txt
			├──...
		├── wcdm/
			├──1.txt
			├──2.txt
			├──...	
		└── planck_ee2.txt		
```

Adapt the name of the training data folder in the ```train-parameters.py``` file accordingly (using the ```DIR``` parameter). The name of the test data folder can be set when the testing is started (see below).

## Training

Change the ```train-parameters.py``` file to set the training parameters that are then passed on to ```train.py```. 
To train the model with a theory error of 5 % execute
```bash
python3 train-curves-parameter.py --sigma_curves='0.05'
```

All the parameters are described in the file but we will explain the most important ones here:

* ```DIR```: path to the training data,e.g. ```'data/train_data'```, use ```train_fname``` to set the name of the training data in the model name
* ```norm_data_name```: path to the file of the normalisation spectrum, e.g. ```'/planck_ee2.txt'```, use ```planck_fname``` to set the name of the normalisation in the model name
* ```k_max```: maximal wavenumber of spectrum used
* ```sample_pace```: only sample every nth k-bin, we recommend n = 4 for the supplied spectra
* ```save_processed_spectra```: Set to ```True``` if the first batch of normalised and noisy spectra should be written to a file to plot them later. Only recommended for a special training run with ```n_epochs``` = 1.
* ```batch_size```: how many spectra per batch. **Has to be adapted to the number of classes.** Must be a multiple of (number of classes) * (noise realisations), e.g. 4 classes, 10 noisy samples -> must be multiple of 40, for example 8000 for 20,000 spectra per class
* ```fname_extra```: appended to the automatically generated model name, to quickly find model in folder

The noise parameters are:

* ```add_noise```: if ```True```, then noise gets added according to the following parameters:
     * ```n_noisy_samples```: number of noise samples added to every training (and testing) spectrum, we suggest 10.
     * ```add_cosvar```: if ```True```, adds Gaussian noise on large scales, depends on survey volume, here adapted for the Euclid telescope.
     * ```add_shot```: if ```True```, adds Gaussian noise to produce the shot noise of a Euclid-like Galaxy survey. We recommend ```False``` for the matter power spectrum.
     * ```add_sys```: if ```True```, theory error curves are added to account for the theoretical error in the modelling
          * ```curves_folder``` path to the folder with files containing the theory error curves
          * ```sigma_curves``` maximal amplitude of the plateau of the theory error curves on small scales (as added to the normalised power spectrum). It can be set in the ```train-parameters.py``` file but it can also be passed on as a parameter in the command line. (See above)
          * ```rescale_curves``` rescale the amplitude of theory error curves. Distribution can be ```uniform```(recommended), ```gaussian``` or ```None```.

When changing the total number of classes (equivalent to the number of folders in the training data) then the batch size has to be adapted accordingly. The new class names have to be added to the c1 parameter that is set in the subprocess call at the end of the train-parameters.py file.

The name of the model is automatically generated from the training parameters in ```train-parameters.py``` as 
<pre> model_<i>&lt;train_name&gt;</i>_samplePace<i>&lt;sample_pace&gt;</i>_kmax<i>&lt;k_max&gt;</i>_planck-<i>&lt;planck_fname&gt;</i>_epoch<i>&lt;n_epochs&gt;</i>
  _batchsize<i>&lt;batch_size&gt;</i>_noiseSamples<i>&lt;n_noisy_samples&gt;</i>_wCV_noShot_wSys_rescale<i>&lt;rescale_curves&gt;</i>
  _GPU_sigmaCurves<i>&lt;sigma_curves&gt;</i>_<i>&lt;fname_extra&gt;</i>
</pre>




## Testing

To test a trained model with a test data set and produce a confusion matrix for the classification run the following

<pre>
python3 test.py --log_path='models/<i>&lt;model_name&gt;</i>/<i>&lt;model_name&gt;</i>_log.txt' 
  --TEST_DIR='<i>&lt;path/to/test-data&gt;</i>' 
  --cm_name_custom='<i>&lt;cm_name&gt;</i>-noSys'
  --add_sys='False'
</pre>

Please adapt the following options

- ```--log_path```: Name and Path of the log-file of the model to be tested.
- ```--TEST_DIR```: Path to the test data, e.g. ```'data/test_data'```.
- ```--cm_name_custom```: Add an (optional) addition to the name of the cm-file (confusion matrix).
- ```--add_sys```: Choose ```'False'``` to leave out the systematic error in testing. This overwrites the setting from the log-file of the training of the model. 
