# Automatic music transcription

We define automatic music transcription as

- given an audiofile
- produce a symbolic music represention from it

## Data processing

Transformer paper

- We used an audio sample rate of 16,000 kHz, an FFT length of 2048 samples, and a hop width of 128 samples. We scaled the output to 512 mel bins (to match the model’s embedding size) and used the log-scaled magnitude

## Data

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

## Or use libmedia
```

Midi files contain events, of event_types, which contains pitch, velocity and timing.

## Approach

1. Input: Wav
2. Create mel-spectrogram, token pairs
3. Feed it in

### Tokenisation

Dataset
Say each batch is one music file for simplicity.

- We chunk one music file into N segments of some duration D.
- For each of the N segments, obtain the decoder output for it
  - Precompute the start times
  - Obtain the MIDI sequence for each chunk
  - Tokenise the MIDI sequence

Spectrogram generation

- `Hop length` is the number of audio samples between each fft window
- N_fft is the number of audio samples used i

  Use Miditok tokenisation ~ 7000 tokens.

- Note [128 values]
- Velocity [ 128 values]
- Time [6000 values]
- EOS [1 value]
- Pad

Temporal resolution of 10 ms.

Idea is to tokenise during runtime, we find

### Modelling

1. Custom Encoder-Decoder using Transformers
2. Pretrained WHISPERS encoder + custom decoder from scratch
3. CNNs are a bit old

### Inference/ Demo

1. Record a clip
2. Split the clip
3. Decode
4. Decokenise

```
brew install fluidsynth
https://github.com/bzamecnik/midi2audio/tree/master?tab=readme-ov-file
```

## Training

Loss: Cross entropy

1. Model learns to predict the token of 0 duration, as it means 'key-off'
2.

## References

https://engineering.atspotify.com/2022/06/meet-basic-pitch/

Hawthorne, Curtis, Ian Simon, Rigel Swavely, Ethan Manilow, and Jesse Engel. ‘Sequence-to-Sequence Piano Transcription with Transformers’. arXiv, 19 July 2021. https://doi.org/10.48550/arXiv.2107.09142.

https://magenta.tensorflow.org/transcription-with-transformers

https://magenta.tensorflow.org/onsets-frames

https://magenta.tensorflow.org/datasets/maestro
