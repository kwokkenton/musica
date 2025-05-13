import os

import librosa
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from midi2audio import FluidSynth
from mido import Message, MetaMessage, MidiFile, MidiTrack
from torch.utils.data import DataLoader, Dataset


def make_spectrogram(audio_filename, target_sr=16e3, n_fft=2048, hop_length=128, n_mels=512):
    """_summary_

    Args:
        audio_filename (_type_): _description_
        target_sr (int, optional): . Defaults to 16e3.
        n_fft (int, optional): _description_. Defaults to 2048.
        hop_length (int, optional): _description_. Defaults to 128.
        n_mels (int, optional): _description_. Defaults to 512.

    Returns:
        _type_: _description_
    """
    y, sr = librosa.load(audio_filename, sr = None, mono=True)
    # resample to lower rate
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, 
                                       sr=target_sr, 
                                        n_fft = n_fft, 
                                        hop_length=hop_length, 
                                        n_mels=n_mels, 
                                        fmax=target_sr//2)

    # Convert to decibels (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB, target_sr

class MaestroDataset(Dataset):
    def __init__(self, df_root):
        self.dir = df_root
        self.df = pd.read_csv(os.path.join(df_root, 'maestro-v2.0.0.csv'))
        self.df = self.df[self.df['year'].isin([2008, 2009])]

        self.n_fft = 2048
        self.hop_length = 128
        self.n_mels = 256
        self.target_sr = 16e3

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        midi_filename = os.path.join(self.dir, self.df.iloc[index]['midi_filename'])
        audio_filename = os.path.join(self.dir, self.df.iloc[index]['audio_filename'])

        S_dB = make_spectrogram(audio_filename, self.target_sr, self.n_fft, self.hop_length, self.n_mels)
        midi_file = mido.MidiFile(midi_filename)

        return torch.tensor(S_dB), midi_file

    
class Tokeniser:
    def __init__(self, ):
        self.time_ms_max  = 60000
        self.time_max  = 6000
        self.velocity_max = 128
        self.note_max = 128
    
    @staticmethod
    def process_midi(midi: MidiFile):
        """_summary_
        # In: Librosa midi file

        Args:
            midi (_type_): _description_

        Returns:
            Out: Absolute times, in ticks (torch.int32)
            Msgs (list)
        """

        times = []
        msgs = []
        for i, track in enumerate(midi.tracks):
            abs_time = 0
            # remove initial track, which is metadata
            if i ==0:
                continue
            for msg in track:
                # time is a timedelta, which is to be incremented
                abs_time += msg.time
                if msg.type in ['note_on', 'note_off']:
                    msgs.append(msg)
                    times.append(round(abs_time/10) * 10)
                else:
                    continue
        times=  torch.tensor(times, dtype=torch.int32)
        return times, msgs
    
    @staticmethod
    def collect_indices_between(times, start_time, end_time):
        return [i for i, t in enumerate(times) if start_time <= t <= end_time] 

    @staticmethod
    def get_midi_between(midi: MidiFile, start_index=None, end_index=None, start_time=None, end_time=None):
        """Obtains midi messages between two indices or times.

        Args:
            midi (_type_): _description_
            start_index (_type_, optional): _description_. Defaults to None.
            end_index (_type_, optional): _description_. Defaults to None.
            start_time (_type_, optional): _description_. Defaults to None.
            end_time (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # First process entire MIDI file
        times, msgs = Tokeniser.process_midi(midi)

        # And then slice
        if (start_index is not None) and (end_index is not None):
            if end_index > len(times) - 1:
                raise ValueError
            idxs = [start_index, end_index]
        elif (start_time is not None) and (end_time is not None):
            # Obtain indices between the provided times
            idxs = Tokeniser.collect_indices_between(times, start_time, end_time)
        else: 
            raise ValueError(f'Start_index:{start_index}')
        return times[idxs[0]:idxs[-1]], msgs[idxs[0]:idxs[-1]]
    
    def tokenise_midi(self, time_ms, velocity, note):
        # Converts single message into triplets of tokens
        assert time_ms >= 0 and time_ms < self.time_ms_max
        assert velocity >= 0 and velocity < self.velocity_max
        assert note >0 and note < self.note_max
        # To nearest 10 ms
        time = time_ms // 10
        return torch.tensor([time, self.time_max + velocity, self.time_max + self.velocity_max + note], dtype=torch.int32)
    
    def process_chunk(self, times, msgs, start_time):
        chunk = torch.tensor([], dtype=torch.int16)
        for t, m in zip(times, msgs):
            chunk = torch.cat([chunk, self.tokenise_midi(t-start_time, m.velocity, m.note)])
        return chunk
    
    def detokenise_midi(self, token_sequence, start_time_ms = 0):
        """
        Convert a flat token sequence into a list of MIDI messages and times.
        token_sequence: 1D tensor of shape (N,) where N is a multiple of 3.
        Returns: list of (absolute_time_ms, mido.Message)
        """
        assert token_sequence.ndim == 1 and token_sequence.size(0) % 3 == 0

        messages = []
        time_ms = 0
        velocity = 64  # Default velocity
        prev_time_ms = 0
        for i in range(0, len(token_sequence), 3):
            t_token = token_sequence[i].item()
            v_token = token_sequence[i + 1].item()
            n_token = token_sequence[i + 2].item()

            # Decode absolute time (quantized to 10ms)
            time_ms = t_token * 10

            # Decode velocity
            velocity = v_token - self.time_max
            if velocity < 0 or velocity >= self.velocity_max:
                raise ValueError(f"Invalid velocity token: {v_token}")

            # Decode note
            note = n_token - self.time_max - self.velocity_max 
            if note <= 0 or note >= self.note_max:
                raise ValueError(f"Invalid note token: {n_token}")

            # Determine message type
     
            msg = Message('note_on', note=note, velocity=velocity, time=time_ms - prev_time_ms)
            prev_time_ms = time_ms

            messages.append((int(start_time_ms + time_ms), msg))

        return messages

def visualise_spectrogram(S_dB, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    return


def collate_fn(spectrogram: torch.Tensor, midi_file: MidiFile):
    # Get batch size, say N
    # Get spectrogram
    # Choose N starting points

    N=4
    max_len = 512
    sampling_rate = 16e3
    hop_length = 128
    spectrogram_frames = []
    chunks = []
    tok = Tokeniser()


    spectrogram_start_idxs = torch.randint(low=0, high=spectrogram.shape[1]- max_len, size=(N,))
    spectrogram_start_idxs = torch.tensor([0, 512, 1024, 1536])

    # Times are in seconds, convert to ticks
    start_times = spectrogram_start_idxs * hop_length/sampling_rate * 1000
    # Get sampling length of say 512
    end_times = (spectrogram_start_idxs + max_len) * hop_length/sampling_rate * 1000


    for i in spectrogram_start_idxs:
        spectrogram_frames.append(spectrogram[:, i:i+max_len])

    for t0, t1 in zip(start_times, end_times):
        times, messages = tok.get_midi_between(midi_file, start_time=t0, end_time = t1)
        chunks.append(tok.process_chunk(times, messages, start_time=t0))

    return spectrogram_frames, chunks

def save_out_spectrogram_and_midi(spectrogram, messages, tempo=500000, sample_rate = 16e3, ticks_per_beat = 384):
    """ For verification, saves out generated MIDI

    # 500,000 µs = 120 BPM
    """

    soundfont = '/Users/kenton/projects/mlx-institute/musica/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2'
    temp_file = 'custom_output.mid'

    # Converts MIDI into a track
    mid = MidiFile(ticks_per_beat = ticks_per_beat)
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4,
                        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    track = MidiTrack()
    mid.tracks.append(track)

    for i in tok.detokenise_midi(messages):
        track.append(i[1])

    track.append(MetaMessage('end_of_track', time=1))
    mid.save(temp_file)

    # using the default sound font in 44100 Hz sample rate
    fs = FluidSynth(soundfont, sample_rate = sample_rate, gain=0.75)
    fs.midi_to_audio(temp_file, 'output_midi.wav')

    # Now do spectrogram into a wave
    S_linear = librosa.feature.inverse.mel_to_stft(librosa.db_to_amplitude(spectrogram), sr=sr, n_fft=n_fft)
    y = librosa.griffinlim(S_linear, n_fft=n_fft,  hop_length=hop_length)
    sf.write("output_spectrogram.wav", y, samplerate=int(sr))
    
if __name__ == "__main__":
    midi_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    wav_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.wav'


    # # Load the MIDI file

    tok = Tokeniser()
    # # times, msgs = tok.get_midi_between(midi_file, start_time=10000, end_time = 20000)
    # times, msgs = tok.get_midi_between(midi_file, start_index=500, end_index = 1024)
    
    # for t, m in zip(times[0:10], msgs[0:10]):
    #     print(t, m)
    # messages = tok.detokenise_midi(tok.process_chunk(times, msgs), times[0])
    # for m in messages[0:10]:
    #     print(m)


    # visualise_spectrogram(spectrogram, sr)
    # plt.savefig('temp.png')
    # print(spectrogram.shape)
    # print(librosa.get_duration(S=spectrogram, n_fft=2048, hop_length=128, sr=sr))
    n_fft = 2048
    hop_length = 128
    # ds = MaestroDataset()
    spectrogram, sr = make_spectrogram(wav_path, n_mels=512)
    midi_file = mido.MidiFile(midi_path)
    s, m = collate_fn(spectrogram, midi_file)

    times, messages = tok.process_midi(midi_file)

    for i, (time, message) in enumerate(zip(times, messages)):
        print(time, message)
        if i ==10:
            break

    print(tok.detokenise_midi(m[0]))


    
    # save_out_spectrogram_and_midi(s[0], m[0], tempo=500000, )