# Birdsong Classifier

## General information

The library it's part of a larger *research* project called **PiaffNet**. Our goal is to make easier for everyone the studies of bird vocalizations by modern **Deep Learning** algorithms. PiaffNet uses modern standard of implementations in Python language. We priveledge simplicity and good enough accuracy over unnescessory complexity.

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
