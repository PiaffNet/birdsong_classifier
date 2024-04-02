# Birdsong Classifier

## General information

The library it's part of a larger *research* project called **PiaffNet**. Our goal is to make it easier for everyone to study and identify birds by their vocalizations using modern **Deep Learning** algorithms. PiaffNet uses modern standard of implementations in Python language. We priveledge simplicity and good enough accuracy over unnescessory complexity.

## Introduction

The birdsong classifier library provides a simple predictive model based on supervised learning over data samples of annotated bird sounds. We use a **Convolutional Neural Network** (CNN) models. Note that this version implemented in TensorFlow library.

The classifier uses a MEL sounds spectograms with $64$ bands and a break frequency at $1750$ Hz. The spectrogam constructed using a Fast Fourier Transform (FFT) with $512$ samples at $48$ kHz sampling rate and an overlap of $25$%. The frequency range of spectogram is limited between $150$ Hz and $15$ kHz to assure the frequency range covering of the majority of bird vocalizations.


## install

### pip install

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip list
```

## how to use

### Train your model

#### Preparing audio for preprocessing

When in project directory

```bash
mkdir raw_data
mkdir raw_data/train_audio
mkdir raw_data/split_data
mkdir raw_data/images_png
```

Copy your files in the train_audio directory, the program uses audio_dataset_from_directory so it needs to be in this format :

train_audio/<br>
&emsp;species1/<br>
&emsp;&emsp;song1.audio<br>
&emsp;&emsp;song2.audio<br>
&emsp;&emsp;...<br>
&emsp;species2/<br>
&emsp;&emsp;song1.audio<br>
&emsp;&emsp;song2.audio<br>
&emsp;&emsp;...<br>
&emsp;...<br>



Slice your audio and detect silent segments :

```bash
make run_slicing
```


#### Transform audio to mel spectrogram

```bash
make run_preprocess
```

#### Train the model on your data

```bash
make run_train
```
