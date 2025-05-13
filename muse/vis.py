from typing import List

import librosa
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
from mido import Message, MetaMessage, MidiFile, MidiTrack


def messages_to_wav(messages: List[Message], tempo: int, sample_rate:int, ticks_per_beat, out_file):
    """ Converts list of 
    """
    temp_file = 'custom_output.mid'
    soundfont = '/Users/kenton/projects/mlx-institute/musica/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2'

    # Converts MIDI into a track
    mid = MidiFile(ticks_per_beat = ticks_per_beat)
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4,
                        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    track = MidiTrack()
    mid.tracks.append(track)
    for i in messages:
        track.append(i)

    track.append(MetaMessage('end_of_track', time=1))
    mid.save(temp_file)

    # using the default sound font in 44100 Hz sample rate
    fs = FluidSynth(soundfont, sample_rate = sample_rate, gain=0.75)
    fs.midi_to_audio(temp_file, out_file)

def save_out_spectrogram_and_midi(spectrogram, messages, tempo=500000, sample_rate = 16e3, ticks_per_beat = 384):
    """ For verification, saves out generated MIDI

    # 500,000 µs = 120 BPM
    """


    messages_to_wav(tempo, sample_rate, ticks_per_beat, out_file)
 



    # Now do spectrogram into a wave
    S_linear = librosa.feature.inverse.mel_to_stft(librosa.db_to_amplitude(spectrogram), sr=sr, n_fft=n_fft)
    y = librosa.griffinlim(S_linear, n_fft=n_fft,  hop_length=hop_length)
    sf.write("output_spectrogram.wav", y, samplerate=int(sr))
    

def visualise_spectrogram(S_dB, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    return