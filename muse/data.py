import logging
import os
from copy import deepcopy

import librosa
import mido
import numpy as np
import pandas as pd
import torch
from mido import Message, MetaMessage, MidiFile, MidiTrack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from muse.vis import messages_to_wav, spectrogram_to_wav

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


class AudioProcessor:
    def __init__(self):
        self.target_sr = 16e3
        self.n_fft = 2048
        self.hop_length = 128
        self.n_mels = 512
        self.sampling_rate = self.target_sr

    def make_spectrogram(self, audio_filename):
        """Makes log-Mel Spectrogram.

        Args:
            audio_filename (_type_): _description_
            target_sr (int, optional): . Defaults to 16e3.
            n_fft (int, optional): _description_. Defaults to 2048.
            hop_length (int, optional): _description_. Defaults to 128.
            n_mels (int, optional): _description_. Defaults to 512.

        Returns:
            spectrogram : shape (n_mels, clip length)
        """
        y, sr = librosa.load(audio_filename, sr=None, mono=True)
        # resample to lower rate
        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.target_sr//2,
        )

        # Convert to decibels (log scale)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    def spectrogram_idx_to_time_ms(self, idx):
        return idx * self.hop_length/self.sampling_rate * 1000


class Tokeniser:
    def __init__(self):
        # Paper uses 6e4
        self.time_ms_max = 1e4
        # This is in 10s of ms
        self.time_max = int(self.time_ms_max // 10)
        self.velocity_max = 128
        self.note_max = 128

        self.bos_id = self.time_max + 128+128
        self.eos_id = self.bos_id + 1
        self.pad_id = self.bos_id + 2

        self.vocab_size = int(self.pad_id + 1)

    @staticmethod
    def process_midi(midi: MidiFile, tempo=500000):
        """ Preprocesses midi file

        midi tempo is microseconds per beat

        Args:
            midi (_type_): _description_
            tempo: microseconds per beat

        Returns:
            Out: Absolute times, in milliseconds (torch.int32)
        """
        ticks = []
        notes = []
        velocities = []

        ticks_per_beat = deepcopy(midi.ticks_per_beat)
        logger.info(f'Ticks per beat{ticks_per_beat}')

        # milliseconds is ticks * microseconds per beat * 1/ticks_per_beat / 1000

        for i, track in enumerate(midi.tracks):
            abs_ticks = 0
            # remove initial track, which is metadata
            if i == 0:
                continue
            for msg in track:
                # time is a timedelta, which is to be incremented
                abs_ticks += msg.time
                assert msg.time >= 0, f'Negative delta time detected: {msg.time}, msg={msg}'
                if msg.type in ['note_on', 'note_off']:
                    notes.append(msg.note)
                    # Binary note on/ note off classification
                    velocities.append(127 if msg.velocity > 0 else 0)
                    # velocities.append(msg.velocity)
                    ticks.append(abs_ticks)
                else:
                    continue

        # Abs times are in milliseconds
        abs_times = Tokeniser.ticks_to_time_ms(
            torch.tensor(ticks), tempo, ticks_per_beat,
        )
        abs_times = torch.round(abs_times/10)*10
        # abs_times =  abs_times.to(torch.int32)
        notes = torch.tensor(notes)
        velocities = torch.tensor(velocities)
        res = torch.stack(
            [abs_times, notes, velocities],
            axis=1,
        ).to(torch.int32)

        assert torch.min(res) >= 0
        # assert torch.max(res_between) < self.tok.vocab_size

        assert torch.min(res[1:, 0] - res[:-1, 0]
                         ) >= 0, 'Times are not ascending'

        return res

    @staticmethod
    def ticks_to_time_ms(ticks, tempo, ticks_per_beat):
        return ticks * tempo/ticks_per_beat / 1000

    @staticmethod
    def time_ms_to_ticks(time, tempo, ticks_per_beat):
        if type(time) is torch.Tensor:
            time = time.to(torch.float32)
        else:
            time = float(time)
        return time * ticks_per_beat * 1000 / tempo

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
        times = res[:, 0]
        start_idx = (times > start_time_ms).nonzero(as_tuple=False)
        # Not inclusive, so we use normal indexing
        end_idx = (times > end_time_ms).nonzero(as_tuple=False)
        assert len(start_idx) > 0
        assert len(end_idx) > 0, ''

        start_idx = start_idx[0].item()
        end_idx = end_idx[0].item()

        # Not figured out what happens with this yet (length must be > 0)
        if end_idx <= start_idx:
            logger.info(
                f'No music between time {start_time_ms}, {end_time_ms}.')
            return torch.empty(0)
        else:
            # Clone or else downstream processing is affected
            return res[start_idx:end_idx].clone()

    def process_chunk(self, res: torch.Tensor, start_time_chunk: int):
        """ Tokenises N instructions into


        1. Turn into times relative to the chunk's start time
        1. Turns times into 10s of ms
        1. Adds <bos> and <eos>

        # Divide by 10 for tokenisation

        Args:
            res (torch.Tensor): MIDI instructions in an array (N,3)
            start_time_chunk (int): Start time in ms

        Returns:
            chunk: Shape (3 N + 2, )
        """
        # Empty tensor
        if res.numel() == 0:
            # raise NotImplementedError
            chunk = torch.tensor([self.bos_id, self.eos_id], dtype=torch.int32)
        else:

            # Turn to time relative to the start time
            time_processed = res[:, 0] // 10 - int(start_time_chunk//10)
            # Make sure that the time makes sense
            assert torch.min(time_processed) >= 0
            assert torch.max(
                time_processed,
            ) < self.time_max, f'{start_time_chunk//10, time_processed}'

            res[:, 0] = time_processed
            res[:, 1] += self.time_max
            res[:, 2] += self.time_max + self.velocity_max

            chunk = res.flatten()
            chunk = chunk.to(dtype=torch.int32)
            # Add <bos> and <eos>
            chunk = torch.cat([torch.tensor([self.bos_id]),
                              chunk, torch.tensor([self.eos_id])])

        assert torch.max(chunk) < self.vocab_size
        assert torch.min(chunk) >= 0
        return chunk

    def detokenise(self, tokens):
        res = tokens[tokens != self.pad_id]
        res = res[res != self.bos_id]
        res = res[res != self.eos_id]

        res = res.reshape(-1, 3)
        res[:, 0] = res[:, 0] * 10
        res[:, 1] -= self.time_max
        res[:, 2] -= self.time_max + self.velocity_max
        return res

    @staticmethod
    def res_to_messages(res, start_time_ms=0, tempo=500000, ticks_per_beat=384):
        messages = []
        prev_tick = Tokeniser.time_ms_to_ticks(
            start_time_ms, tempo, ticks_per_beat,
        )

        for i in res:
            tick = Tokeniser.time_ms_to_ticks(i[0], tempo, ticks_per_beat)
            tick_delta = int(tick - prev_tick)
            prev_tick = tick
            assert tick_delta >= 0, f'Negative delta time detected: {tick_delta}'
            msg = Message(
                'note_on', note=int(
                    i[1],
                ), velocity=int(i[2]), time=tick_delta,
            )
            messages.append(msg)
        return messages


class DataProcessor:
    def __init__(self, batch_size=4):

        self.tok = Tokeniser()
        self.ap = AudioProcessor()
        # Get sampling length of say 512

        self.max_enc_len = 500
        self.max_dec_len = 1024
        self.N = batch_size

        self.make_spectrogram = self.ap.make_spectrogram
        self.process_midi = self.tok.process_midi

    def __call__(self, wav_path: str, midi_path: str):
        midi_file = mido.MidiFile(midi_path)
        spectrogram = self.make_spectrogram(wav_path)
        midi_tensor = self.process_midi(midi_file)
        return spectrogram, midi_tensor

    def collate_fn(self, batch: list):
        """
        Get base midi file first

        # Get batch size, say N
        # Get spectrogram
        # Choose N starting points

        """
        spectrogram, res = batch[0]
        # Batching
        spectrogram_frames = []
        chunks = []

        # Hack 2x so that fewer bugs occur
        spectrogram_start_idxs = torch.randint(
            low=0, high=spectrogram.shape[1] - 2 * self.max_enc_len, size=(self.N,),
        )
        # spectrogram_start_idxs = torch.tensor([0, 8000])

        # Times are in milliseconds
        start_times_ms = self.ap.spectrogram_idx_to_time_ms(
            spectrogram_start_idxs)
        # fix length to self.max_enc_len for now
        end_times_ms = self.ap.spectrogram_idx_to_time_ms(
            spectrogram_start_idxs + self.max_enc_len)

        # Make spectrogram frames
        for i in spectrogram_start_idxs:
            spectrogram_frames.append(
                torch.tensor(spectrogram[:, i:i+self.max_enc_len]),
            )

        # These are absolute times
        for t0_ms, t1_ms in zip(start_times_ms, end_times_ms):
            res_between = self.tok.get_midi_between(res, t0_ms, t1_ms)
            # No negative times, notes or anything
            assert torch.min(res_between) >= 0
            # assert torch.max(res_between) < self.tok.vocab_size
            if len(res_between) > 1:
                assert torch.min(
                    res_between[1:, 0] - res_between[:-1, 0]) >= 0, 'Times are not ascending'

            chunks.append(self.tok.process_chunk(
                res_between, start_time_chunk=t0_ms))

        # Permute so it is N, seq_len, num_mels or num_features
        return torch.stack(spectrogram_frames).permute(0, 2, 1), pad_sequence(chunks, batch_first=True, padding_value=self.tok.pad_id)


class MaestroDataset(Dataset):
    def __init__(self, data_dir: str, path_to_csv: str, dp, train=True):
        super().__init__()
        # self.dir = df_root
        self.dir = data_dir
        self.dp = dp

        self.df = pd.read_csv(path_to_csv)
        # Limit to 10 lines
        self.df = self.df.iloc[0:2]
        # Filter
        self.df.loc[0:1, 'split'] = 'train'
        self.df.loc[1:2, 'split'] = 'valid'

        split = 'train' if train else 'valid'
        self.df = self.df[self.df['split'] == split]
        print(self.df)
        # self.df = self.df[self.df['year'].isin([2008, 2009])]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        logger.info(f'Loading {self.df.iloc[index]["midi_filename"]}.')
        midi_filename = os.path.join(
            self.dir, self.df.iloc[index]['midi_filename'],
        )
        audio_filename = os.path.join(
            self.dir, self.df.iloc[index]['audio_filename'],
        )

        # Returns entire file
        S_dB, midi_file = self.dp(audio_filename, midi_filename)
        return S_dB, midi_file


class MaestroDatasetSingle(Dataset):
    def __init__(self, wav_path, midi_path, dp: DataProcessor):
        super().__init__()
        self.midi_filename = midi_path
        self.audio_filename = wav_path
        self.dp = dp

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Returns entire file
        S_dB, midi_file = self.dp(self.audio_filename, self.midi_filename)
        return S_dB, midi_file


if __name__ == '__main__':
    midi_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    wav_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.wav'

    dp = DataProcessor(batch_size=4)
    train_ds = MaestroDatasetSingle(wav_path, midi_path, dp)
    # Create spectrogram
    spectrograms, inputs = dp.collate_fn([train_ds[0]])

    # visualise_spectrogram(spectrogram, sr)
    # plt.savefig('temp.png')
    # print(spectrogram.shape)
    # print(librosa.get_duration(S=spectrogram, n_fft=2048, hop_length=128, sr=sr))
    # ds = MaestroDataset()

    messages = dp.tok.res_to_messages(dp.tok.detokenise(inputs[2]))

    for m in messages:
        print(m)

    messages_to_wav(
        messages, tempo=500000, sample_rate=dp.ap.sampling_rate,
        ticks_per_beat=384, out_file='out.wav',
    )
    spectrogram_to_wav(spectrograms[2])
