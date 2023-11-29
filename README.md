# birdsong_classifier

we selected a Fast Fourier Transform (FFT) window size of $10.7$ ms ($512$ samples at $48$ kHz sampling rate) and an overlap of 25%, each frame representing a time step of 8 ms.

The frequency range of most bird vocalizations is limited between 250 Hz and 8.3 kHz (Hu and Cardoso, 2009, supplemental material). Therefore, we restricted the frequency range of the spectrogram to values between 150 Hz and 15 kHz covering the frequency range of the vast majority of bird vocalizations but also leaving room for pitch shifts during data augmentation.

We performed frequency compression using a mel scale with 64 bands and a break frequency at 1750 Hz—considerably above the original proposal (Stevens et al., 1937) to achieve approximate linear scaling up to 500 Hz. According to the work of Schlüter (2018), using a nonlinear magnitude scale seems to be the most appropriate choice for bird call recognition in noisy environments


## install

### Wagon general install

```bash
pip install --upgrade pip
pip install -r https://gist.githubusercontent.com/krokrob/53ab953bbec16c96b9938fcaebf2b199/raw/9035bbf12922840905ef1fbbabc459dc565b79a3/minimal_requirements.txt
pip list
```
