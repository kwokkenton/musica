import logging
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from midi2audio import FluidSynth
from mido import Message, MetaMessage, MidiFile, MidiTrack

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def plot_spectrogram_from_wav(wav_path:str, n_fft = 2048, hop_length = 128, n_mels = 512, fmax = 8000, show=False):

    y, sr = librosa.load(wav_path, sr = None, mono=True)
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft = n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    # Convert to decibels (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    if show:
        plt.show()

def messages_to_wav(messages: List[Message], tempo: int, sample_rate: int, 
                    ticks_per_beat:int, out_file:str, soundfont_path:str):
    """ Generates an audible wav file from list of messages.

    """
    temp_file = 'custom_output.mid'

    # Converts MIDI into a track
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track0.append(
        MetaMessage(
            'time_signature', numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
        ),
    )
    track0.append(MetaMessage('end_of_track', time=1))
    track = MidiTrack()
    mid.tracks.append(track)
    for i in messages:
        track.append(i)

    track.append(MetaMessage('end_of_track', time=1))
    mid.save(temp_file)

    # using the default sound font in 44100 Hz sample rate
    fs = FluidSynth(soundfont_path, sample_rate=sample_rate)
    fs.midi_to_audio(temp_file, out_file)


def spectrogram_to_wav(spectrogram, output_filename, sample_rate=16e3, hop_length=128, n_fft=2048):
    # Spectrogram into a wave
    spectrogram = np.array(spectrogram) # shape N_mels, N_seq
    S_linear = librosa.feature.inverse.mel_to_stft(
        librosa.db_to_amplitude(spectrogram), sr=sample_rate, n_fft=n_fft,
    )
    y = librosa.griffinlim(S_linear, n_fft=n_fft,  hop_length=hop_length)
    sf.write(output_filename, y, samplerate=int(sample_rate))


def visualise_spectrogram(S_dB, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000,
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

def sanity_check(batch, dp):
    import librosa

    from muse.vis import spectrogram_to_wav
    x,y = batch
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x[0].cpu().numpy().T, sr=16e3, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('output/temp.png')

    spectrogram_to_wav(x[0].cpu().numpy().T, 'output/spectrogram.wav')

    # This is for sanity checking the 
    messages = dp.tok.res_to_messages(dp.tok.detokenise(y[0]))

    for m in messages:
        print(m)

    messages_to_wav(
            messages, tempo=500000, sample_rate=dp.ap.sampling_rate,
            ticks_per_beat=384, out_file='output/out.wav', soundfont_path = '/root/musica/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2'
        )        