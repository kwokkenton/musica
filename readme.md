# Automatic music transcription

We define automatic music transcription as
- given an audiofile
- produce a symbolic music represention from it

## Data processing

Transformer paper
- We used an audio sample rate of 16,000 kHz, an FFT length of 2048 samples, and a hop width of 128 samples. We scaled the output to 512 mel bins (to match the model’s embedding size) and used the log-scaled magnitude

## Data
```
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

## Or use libmedia
```
## References

Hawthorne, Curtis, Ian Simon, Rigel Swavely, Ethan Manilow, and Jesse Engel. ‘Sequence-to-Sequence Piano Transcription with Transformers’. arXiv, 19 July 2021. https://doi.org/10.48550/arXiv.2107.09142.

https://magenta.tensorflow.org/transcription-with-transformers

https://magenta.tensorflow.org/onsets-frames

https://magenta.tensorflow.org/datasets/maestro