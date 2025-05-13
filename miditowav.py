from midi2audio import FluidSynth

# using the default sound font in 44100 Hz sample rate
fs = FluidSynth('/Users/kenton/projects/mlx-institute/musica/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2', gain=0.75)
# fs.midi_to_audio('input.mid', 'output.wav', gain=0.75)
fs.midi_to_audio('/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi', 'output_midi.wav')

# optional third argument to control gain (defaults to 0.2)


# FLAC, a lossless codec, is supported as well (and recommended to be used)
# fs.midi_to_audio('input.mid', 'output.flac')