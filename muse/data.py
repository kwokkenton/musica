import logging
import os
from copy import deepcopy

import librosa
import mido
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from mido import Message, MetaMessage, MidiFile, MidiTrack
from torch.utils.data import DataLoader, Dataset

from muse import vis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

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
        
        return torch.tensor(S_dB), Tokeniser.process_midi(midi_file)

    
class Tokeniser:
    def __init__(self, ):
        self.time_ms_max  = 60000
        self.time_max  = 6000
        self.velocity_max = 128
        self.note_max = 128
    
    @staticmethod
    def process_midi(midi: MidiFile, tempo = 500000):
        """ Preprocesses midi file

        midi tempo is microseconds per beat

        Args:
            midi (_type_): _description_
            tempo: microseconds per beat

        Returns:
            Out: Absolute times, in milliseconds (torch.int32)
            
        """

        ticks = []
        # msgs = []
        notes = []
        velocities = []

        ticks_per_beat = deepcopy(midi.ticks_per_beat)
        logger.info(f'Ticks per beat{ticks_per_beat}')

        # milliseconds is ticks * microseconds per beat * 1/ticks_per_beat / 1000

        for i, track in enumerate(midi.tracks):
            abs_ticks = 0
            # remove initial track, which is metadata
            if i ==0:
                continue
            for msg in track:
                # time is a timedelta, which is to be incremented
                abs_ticks += msg.time
                assert msg.time >= 0, f"Negative delta time detected: {msg.time}, msg={msg}"
                if msg.type in ['note_on', 'note_off']:
                    # msg.time = 0
                    # msgs.append(Message('note_on', note=msg.note, velocity=msg.velocity))
                    notes.append(msg.note)
                    velocities.append(msg.velocity)
                    ticks.append(abs_ticks)
                else:
                    continue

        # Abs times are in milliseconds
        abs_times = Tokeniser.ticks_to_time_ms(torch.tensor(ticks), tempo, ticks_per_beat) 
        abs_times = torch.round(abs_times/10)*10
        # abs_times =  abs_times.to(torch.int32)
        notes = torch.tensor(notes)
        velocities = torch.tensor(velocities)
        res = torch.stack([abs_times, notes, velocities], axis=1).to(torch.int32)
        return res
    
    @staticmethod
    def ticks_to_time_ms(ticks, tempo, ticks_per_beat):
        return ticks* tempo/ticks_per_beat / 1000
    
    @staticmethod
    def time_ms_to_ticks(time, tempo, ticks_per_beat):
        time = torch.tensor(time, dtype=torch.float32)
        return time * ticks_per_beat * 1000/ tempo
    
    @staticmethod
    def collect_indices_between(times, start_time, end_time):
        return [i for i, t in enumerate(times) if start_time <= t <= end_time] 

    @staticmethod
    def get_midi_between(res: torch.Tensor, start_time_ms: int, end_time_ms: int):
        """Obtains midi messages between two indices or times.

        Args:
            midi (_type_): _description_
            start_time (_type_, optional): _description_. Defaults to None.
            end_time (_type_, optional): _description_. Defaults to None.

        Returns:
            processed_res
        """
        assert type(res) is torch.Tensor
        # And then slice
        times = res[:,0] 
        start_idx = (times > start_time_ms).nonzero(as_tuple=False)
        # Not inclusive, so we use normal indexing
        end_idx = (times > end_time_ms).nonzero(as_tuple=False) 
        assert len(start_idx) > 0
        assert len(end_idx) > 0

        start_idx = start_idx[0].item()
        end_idx = end_idx[0].item()

        # Not figured out what happens with this yet (length must be > 0)
        if end_idx <= start_idx:
            raise NotImplementedError
        return res[start_idx:end_idx]
    
    def tokenise_midi(self, abs_rel_time:int, velocity:int, note:int):
        # Converts single message into triplets of tokens
        assert abs_rel_time >= 0 and abs_rel_time < self.time_ms_max
        assert velocity >= 0 and velocity < self.velocity_max
        assert note >0 and note < self.note_max

        # To nearest 10 ms
        time = int(abs_rel_time // 10)

        return torch.tensor([time, self.time_max + velocity, self.time_max + self.velocity_max + note], dtype=torch.int32)
    
    def process_chunk(self, res: torch.Tensor, start_time_chunk:int):

        # abs_times and msgs are already of the relevant lengths
        res[:, 0] += int(start_time_chunk)
        res[:, 0] = res[:, 0] // 10

        res[:, 1] += self.time_max
        res[:,2] += self.time_max + self.velocity_max

        chunk = res.flatten()
        chunk = chunk.to(dtype = torch.int32)

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
    
    def res_to_messages(self, res, start_time_ms = 0, tempo = 500000, ticks_per_beat=384):
        messages = []
        prev_tick = Tokeniser.time_ms_to_ticks(start_time_ms, tempo, ticks_per_beat)


        for i in res:
            tick = Tokeniser.time_ms_to_ticks(i[0], tempo, ticks_per_beat)
            tick_delta = int(tick - prev_tick)
            prev_tick = tick
            assert tick_delta >= 0, f"Negative delta time detected: {tick_delta}"
            msg = Message('note_on', note=int(i[1]), velocity=int(i[2]), time= tick_delta)
            messages.append(msg)
        return messages
    


def collate_fn(spectrogram: torch.Tensor, res: torch.Tensor):
    """
    Get base midi file first

    # Get batch size, say N
    # Get spectrogram
    # Choose N starting points
    
    """
    N=4
    max_len = 512
    sampling_rate = 16e3
    hop_length = 128

    # Batching
    spectrogram_frames = []
    chunks = []
    tok = Tokeniser()

    spectrogram_start_idxs = torch.randint(low=0, high=spectrogram.shape[1]- max_len, size=(N,))
    spectrogram_start_idxs = torch.tensor([0, 512, 1024, 1536])

    # Times are in milliseconds
    start_times_ms = spectrogram_start_idxs * hop_length/sampling_rate * 1000
    # Get sampling length of say 512
    end_times_ms = (spectrogram_start_idxs + max_len) * hop_length/sampling_rate * 1000

    # Make spectrogram frames
    for i in spectrogram_start_idxs:
        spectrogram_frames.append(spectrogram[:, i:i+max_len])

    #
    for t0_ms, t1_ms in zip(start_times_ms, end_times_ms):
        res_between = tok.get_midi_between(res, t0_ms, t1_ms)
        chunks.append(tok.process_chunk(res_between, start_time_chunk=t0_ms))

    return spectrogram_frames, chunks


    
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
    midi_processed = tok.process_midi(midi_file)
    print(midi_processed)
    res = tok.get_midi_between(midi_processed, 1000, 6000)
    messages = tok.res_to_messages(res)
    from muse.vis import messages_to_wav
    for m in messages:
        print(m)

    spectrograms, inputs = collate_fn(spectrogram, midi_processed)
    print(spectrograms, inputs)
    # messages_to_wav(messages, tempo = 500000, sample_rate = 16e3, ticks_per_beat = 384, out_file = 'out.wav')
    # for i, (time, message) in enumerate(zip(times, messages)):
    #     print(time, message)
    #     if i ==10:
    #         break

    # print(tok.detokenise_midi(m[0]))


    
    # save_out_spectrogram_and_midi(s[0], m[0], tempo=500000, )